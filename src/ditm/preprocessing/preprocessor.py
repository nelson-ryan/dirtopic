import abc
import spacy
from .. import config
from ..import nlp_filters
from ..util import timeit
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from ..config import logging
from ..contenttype import ContentType
import multiprocessing
from typing import ClassVar
logger = logging.getLogger(__name__)


class Preprocessor(abc.ABC):
    content_type: ClassVar[ContentType]

    try:
        NLP = spacy.load(config.SPACY_MODEL, exclude = [ "parser" ])
        NLP.add_pipe("sentencizer")
    except OSError:
        raise ImportError(f"SpaCy model {config.SPACY_MODEL} not found.")

    CUSTOM_STOPWORDS = nlp_filters.STOPWORDS
    NER_FILTER = nlp_filters.NER
    

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


    @timeit
    def lemmatize(self,
        df: pd.DataFrame,
        remove_stop = True,
        extra_stopwords: frozenset | set = frozenset(),
        split_sents = True,
        retain_unlemmatized = True,
    ) -> pd.DataFrame:
        """ Lemmatization using SpaCy pipeline.

        remove_stop -- flag to retain stopwords only when lemmatizing stopwords
        extra_stopwords -- an external list of stopwords for exclusion
        split_sents -- whether to split each document into sentence-level docs

        Splitting on sentences is seemingly a common practise and mentioned in
        BERTopic documentation. It will certainly have effects on the clustering
        behavior, which may need to be accounted for with UMP and HDBSCAN
        parameters.
        Most crucially, however, is that it carries potential to EXPLODE
        computation time during clustering.


        returns : new df with timestamps and tokenized documents
        The index of the output df matches the original
        """

        # Quick lemmatization on stopword list so that we're not comparing
        # lemmas against raw tokens
        # And bring in ALL the stops
        lemmatized_stopwords = {
            stop.lemma_ for doc in Preprocessor.NLP.pipe(
                extra_stopwords |
                Preprocessor.CUSTOM_STOPWORDS |
                ENGLISH_STOP_WORDS
            )
            for stop in doc
        }

        def _is_keeper(t) -> bool:
            """ Here to save duplication in tokenization
            Conditions under which a token is kept in the output doc """
            return (
                not (remove_stop and t.is_stop)
                and not (remove_stop and t.lemma_ in lemmatized_stopwords)
                and not t.is_punct
                and len(t.text) > 2
                and t.is_alpha
                and not t.ent_type_ in Preprocessor.NER_FILTER
                and t.pos_ not in ("PROPN",)
            )

        logger.info("Starting lemmatize")

        docs = Preprocessor.NLP.pipe(
            df['document'],
            batch_size=64,
            n_process = multiprocessing.cpu_count() - 1
        )

        lemmatized_docs = []
        texts = []

        def _lemmas(doc):
            return " ".join(t.lemma_ for t in doc if _is_keeper(t))

        # tokenization/lemmatization step
        for doc in docs:
            # either individual sentences or entire source texts
            # Note: long docs are a problem for SentenceTransformers downstream
            doc_spans = list(doc.sents) if split_sents else [doc]
            lemmatized_docs.append([_lemmas(d) for d in doc_spans])

            if retain_unlemmatized:
                texts.append([d.text for d in doc_spans])

        doc_cols = ['document'] # names of document columns, exists for untokenized
        df['document'] = pd.array(
            lemmatized_docs,
            dtype = object # prevents breakage on empty datapull
        )
        if retain_unlemmatized:
            df['document_untokenized'] = pd.array(texts, dtype=object)
            doc_cols.append('document_untokenized')

        # each new doc in its own row, crucially keeping original indices
        df = df.explode(doc_cols, ignore_index = False)

        # index for individual docs bc original index is no longer unique
        df['sent_idx'] = df.groupby(level=0).cumcount()

        # Also clean out docs that are empty after processing
        df = df.loc[
            df['document'].notna() &
            df['document'].str.strip().ne("")
        ]

        logger.info("Completed lemmatize")
        return df


    @abc.abstractmethod
    def build_stopwords(
        self,
        rawdf: pd.DataFrame,
        extra_stopwords: frozenset = frozenset(),
    ) -> frozenset:
        """ The list of stopwords is passed to BERTopic's 'vectorizer' component,
            here CountVectorizer.

            The whole list is composed from several sources:
            sklearn's ENGLISH_STOP_WORDS, a list of names pulled from Corpus
            email addresses, and an adhoc list of manual stopwords (which may
            need to fall out via c-tf-idf or some other means of sequestering
            a semi-supervised cluster that is later excluded).
        """
        raise NotImplementedError


    @abc.abstractmethod
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame: ...


