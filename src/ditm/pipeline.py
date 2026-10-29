from ditm import (
    config,
    CorpusComposite,
    Corpus,
    EmailPreprocessor,
    TranscriptPreprocessor,
    EmailDbLoader,
    TranscriptDbLoader,
)
from ditm import viz
from ditm.nlp_filters import STOPWORDS, NER
from ditm.dates import datewindow
from ditm import params
from ditm.util import make_runhash, get_or_make, timeit

from dataclasses import asdict
import pickle

import numpy as np

import multiprocessing
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

import logging
logger = logging.getLogger(__name__)
from concurrent.futures.process import BrokenProcessPool

from pathlib import Path

def pipeline(
    corpus_spec: params.CorpusSpec,
    umap_params: params.UmapParams,
    hdbscan_params: params.HdbscanParams,
    vectorizer_params: params.VectorizerParams,
    cache_dir: Path | None = None
):
    """

    cache_dir: Directory where to save cached objects for testing sweeps.
        Leave as None to not store large unused objects to disk.
    
    Returns:
        metrics:       pd.Series
        artifactpaths: dict
    """
    startlines = [
        f"{type(p).__name__}.{k} = {v}"
        for p in (corpus_spec, umap_params, hdbscan_params, vectorizer_params)
        for k, v in asdict(p).items()
    ]
    logger.info("Commencing run with parameters:\n%s", "\n".join(startlines))

    for d in (config.OUTPUTDIR, config.VIZDIR, cache_dir):
        if d is not None:
            d.mkdir(parents = True, exist_ok = True)

    def cache_filepath(pipeline_stage, runhash, file_extension):
        """ Builds a filename for cached objects if cache_dir is specified """
        return (
            None if cache_dir is None
            else cache_dir / f"{pipeline_stage}_{runhash}.{file_extension}"
        )


    # creating dates before hash generation
    datespec = datewindow(display_years = corpus_spec.years)

    # function to be used with cache_path to save numpy enbeddings
    save_npy = lambda obj, f: np.save(f, obj)

    runhash = make_runhash( # note this call is duplicated in flow.py
        corpus_spec, umap_params, hdbscan_params, vectorizer_params,
        datespec, sorted(STOPWORDS), sorted(NER)
    )
    corpus_hash = make_runhash(
        corpus_spec, datespec, sorted(STOPWORDS), sorted(NER)
    )
    embedding_hash = make_runhash(corpus_hash, str(config.EMBEDDING_MODEL))
    umap_hash = make_runhash(embedding_hash, umap_params)

# corpus

    @timeit
    def build_corpus():
        """ Produce corpus documents
        Encapsulated into a function so that it may be skipped if cached
        or called by ditm.util.get_or_make
        """
        corpus = CorpusComposite([
            Corpus(
                loader = EmailDbLoader(),
                preprocessor = EmailPreprocessor(),
                datewindow = datespec,
                min_tokens = corpus_spec.min_tokens_email,
            ),
            Corpus(
                loader = TranscriptDbLoader(),
                preprocessor = TranscriptPreprocessor(),
                datewindow = datespec,
                min_tokens = corpus_spec.min_tokens_transcript,
            )
        ])
        corpus.load()
        corpus.preprocess_docs()
        return corpus

    corpus = get_or_make(
        path = cache_filepath(
            pipeline_stage = "corpus",
            runhash = corpus_hash,
            file_extension = "pickle"
        ),
        make_fn = build_corpus,
        save_fn = pickle.dump,
        load_fn = pickle.load,
    )
    documents = list(corpus.documents)


# models
    logger.info("Initializing UMAP model")
    umap_model = UMAP(
        n_neighbors      = umap_params.n_neighbors,
        n_components     = umap_params.n_components,
        min_dist         = umap_params.min_dist,
        metric           = umap_params.metric,
        random_state     = umap_params.random_state,
        init             = umap_params.init,
        verbose          = umap_params.verbose,
    )

    logger.info("Initializing HDBSCAN model")
    hdbscan_model = HDBSCAN(
        min_cluster_size          = hdbscan_params.min_cluster_size,
        min_samples               = hdbscan_params.min_samples,
        cluster_selection_epsilon = hdbscan_params.cluster_selection_epsilon,
        cluster_selection_method  = hdbscan_params.cluster_selection_method,
        prediction_data           = hdbscan_params.prediction_data,
        core_dist_n_jobs          = multiprocessing.cpu_count() - 1,
    )

    logger.info("Initializing CountVectorizer")
    vectorizer_model = CountVectorizer(
        ngram_range = vectorizer_params.ngram_range,
        min_df      = vectorizer_params.min_df,
        max_df      = vectorizer_params.max_df,
    )


# Generate SentenceTransformer embeddings
    embedding_model = SentenceTransformer(str(config.EMBEDDING_MODEL.resolve()))

    embeddings = get_or_make(
        path = cache_filepath(
            pipeline_stage = "embeddings",
            runhash = embedding_hash,
            file_extension = "npy"
        ),
        make_fn = lambda: embedding_model.encode(
            inputs = documents,
            convert_to_numpy = True,   # default, quiets linter
            convert_to_tensor = False, # default, quiets linter
        ),
        save_fn = save_npy,
        load_fn = np.load,
    )
    assert(len(embeddings) == len(documents))


# Reduce dimensionality for clustering
    def reduce():
        try:
            return umap_model.fit_transform(embeddings)
        except MemoryError:
            logger.exception("MemoryError at UMAP reduction")
            raise

    reduced_embeddings = get_or_make(
        path = cache_filepath(
            pipeline_stage = "umap",
            runhash = umap_hash,
            file_extension = "npy",
        ),
        make_fn = reduce,
        save_fn = save_npy,
        load_fn = np.load,
    )

    assert(len(reduced_embeddings) == len(embeddings))

# contructing topicmodel
    logger.info("Initializing BERTopic")
    topic_model = BERTopic(
        embedding_model = embedding_model,
        # This UMAP model is a no-op because embeddings are already reduced above
        umap_model = BaseDimensionalityReduction(),
        hdbscan_model = hdbscan_model,
        vectorizer_model = vectorizer_model,
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words = True),
        # Representation only affects displayed topic labels (words)
        representation_model = MaximalMarginalRelevance(diversity = .2),
        calculate_probabilities = False,
        verbose = True,
    )


# Fit the model
    try:
        logger.info("Fitting BERTopic")
        _, _ = topic_model.fit_transform(
                documents = documents,
                embeddings = reduced_embeddings
        )
    except BrokenProcessPool:
        logger.exception("min_cluster_size too high")
        raise
    except OSError:
        logger.exception("")
        raise
    except MemoryError:
        logger.exception("Memory error: "
            "lower time frame of query "
            "or lower min_samples or free memory")
        raise
    except:
        logger.exception("It broke. Dunno why.")
        raise


# show me the pictures
    artifactpaths = {}
    metrics = {}

# timeviz
    logger.info("Creating timescale visualization")
    baselines: list[viz.Baseline] = ["rolling", "growing"]
    for baseline in baselines:
        timeviz_path = config.VIZDIR / f"timeviz_{baseline}_{runhash}.html"
        timechange = viz.topic_changes(
            topic_model.topics_,
            list(corpus.timestamps),
            alpha0 = 500, # stops zeros from breaking math, little effect otherwise
            k = corpus_spec.baseline_quarters,
            freq = "Q",
            direction = "both",
            level = .998, # Threshold of chance. Main reduce-output lever.
            n_perm = 3000,
            baseline = baseline, # type: ignore
        )
        words = viz.bertopic_words(topic_model)
        timechange.heatmap(words).write_html(timeviz_path)
        artifactpaths[f"timeviz_{baseline}_{runhash}"] = timeviz_path

# bar plot
    topic_sizes_path = config.VIZDIR / f"topic_sizes_{runhash}.html"
    logger.info("Creating topic-count visualization")
    viz.topic_sizes(
        topic_model.get_topic_info()
    ).write_html(topic_sizes_path)
    artifactpaths['topic_sizes'] = topic_sizes_path

# corpus data
    topic_info_path = config.OUTPUTDIR / f"topic_info_{runhash}.csv"
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(topic_info_path, index = False)
    artifactpaths['topic_info'] = topic_info_path

    metrics = (
        topic_info
        .loc[topic_info.Topic != -1, :]
        .Count
        .describe()
        .astype(int)
    )
    metrics.index = metrics.index.str.replace(
        r'(\d+)%', r'perc_\1', regex = True
    )
    metrics["doc_total_count"] = len(documents)
    metrics["min_cluster_size_proportion"] = hdbscan_params.min_cluster_size / len(documents)

    corpus_data_path = config.OUTPUTDIR / f"corpus_data_{runhash}.parquet"
    (
        corpus
        .frame()
        .assign(Topic = topic_model.topics_)
        .to_parquet(corpus_data_path)
    )
    artifactpaths['corpus_data'] = corpus_data_path

    return metrics, artifactpaths

