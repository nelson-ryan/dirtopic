import filecmp
from pathlib import Path
import itertools
from datetime import datetime
from urllib.parse import urlparse, unquote
from dataclasses import asdict

from ditm.pipeline import pipeline
from ditm.util import make_runhash
from ditm import (
    config,
    params,
    datewindow
)
from ditm.nlp_filters import STOPWORDS, NER

import mlflow

import logging
logger = logging.getLogger(__name__)
from ditm.logs import setup_logging

if __name__ == "__main__":
    logpath = setup_logging()

    # range of parameters to test
    sweeps = {
        # UMAP
        "n_neighbors" : [
            15,
        ],
        "n_components" : [
            5
        ],
        "init" : [
            "pca",
            # "spectral"
        ],
        # HDBSCAN
        "min_cluster_size" : [
            1800,
            # 2000,
            # 3000
        ],
        "min_samples" : [
            10,
            50,
        ],
        "cluster_selection_epsilon" : [
            0.0,
            # 0.1
        ],
        "cluster_selection_method" : [
            "leaf"
        ],
        "years" : [
            9,
        ],
        # "min_tokens_email" : [ 3 ],
        # "min_tokens_transcript" : [ 4 ],
    }

    mlflow.set_experiment(
        f"ditm_cluster_testing_{datetime.now().strftime("%Y%m%d")}"
    )

    # iterate all combinations
    permutations = itertools.product( *sweeps.values() )

    for permutation in permutations:

        sweep = {
            k : v for k, v in
            zip(sweeps.keys(), permutation)
        }

        corpus_spec = params.CorpusSpec(
            years = sweep["years"],
            # min_tokens_email = sweep['min_tokens_email'],
            # min_tokens_transcript = sweep['min_tokens_transcript'],
        )
        datespec = datewindow(display_years = corpus_spec.years)
        umap_params = params.UmapParams(
            n_components = sweep['n_components'],
            n_neighbors = sweep['n_neighbors'],
            init = sweep['init'],
        )
        hdbscan_params = params.HdbscanParams(
            min_cluster_size = sweep['min_cluster_size'],
            min_samples = sweep['min_samples'],
            cluster_selection_epsilon = sweep['cluster_selection_epsilon'],
            cluster_selection_method = sweep['cluster_selection_method'],
        )
        vectorizer_params = params.VectorizerParams(
        )
        runhash = make_runhash( # note this call is duplicated in pipeline.py
            corpus_spec, umap_params, hdbscan_params, vectorizer_params,
            datespec, sorted(STOPWORDS), sorted(NER)
        )

        with mlflow.start_run(run_name=f"test_{runhash}"):

            mlflow.log_params({
                k : v
                for p in (corpus_spec, umap_params, hdbscan_params, vectorizer_params)
                for k, v in asdict(p).items()
            })

            try:
                metrics, artifactpaths = pipeline(
                    corpus_spec = corpus_spec,
                    umap_params = umap_params,
                    hdbscan_params = hdbscan_params,
                    vectorizer_params = vectorizer_params,
                    cache_dir = config.TMPDIR
                )
            except Exception as e:
                mlflow.set_tags({
                    "failure_type" : type(e).__name__,
                    "failure_detail" : repr(e)[:250],
                    "failure_cause" : repr(e.__cause__)[:250],
                })
                mlflow.end_run(status = "FAILED")
                continue

            mlflow.log_metrics(metrics.to_dict())

            for artifact, path in artifactpaths.items():
                # was in a function in flow, could be again
                mlflow.log_artifact(str(path.resolve()))
                # get new artifact path
                copied_uri = mlflow.get_artifact_uri(path.name)
                uri_parsed = urlparse(copied_uri)
                copied_destpath = Path(unquote(uri_parsed.path))
                # compare copy and original, delete original if match
                if filecmp.cmp(path, copied_destpath, shallow = False):
                    path.unlink() # delete
                else:
                    logger.warning(f"No match not deleted for {path.resolve()}")
