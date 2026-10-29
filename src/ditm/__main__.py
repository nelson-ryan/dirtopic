from ditm.pipeline import pipeline
from ditm import params

from dataclasses import asdict
import filecmp
from urllib.parse import urlparse, unquote
from datetime import datetime

from pathlib import Path

import logging
logger = logging.getLogger(__name__)
from ditm.logs import setup_logging

import mlflow


if __name__ == "__main__":

    setup_logging()

    # Use defaults in main
    corpus_spec = params.CorpusSpec()
    umap_params = params.UmapParams()
    hdbscan_params = params.HdbscanParams()
    vectorizer_params = params.VectorizerParams()

    mlflow.set_experiment("ditm")
    with mlflow.start_run(
        run_name = f"ditm_{datetime.now().strftime("%Y%m%d")}"
    ):

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
                cache_dir = None
            )
        except Exception as e:
            mlflow.set_tags({
                "failure_type" : type(e).__name__,
                "failure_detail" : repr(e)[:250],
                "failure_cause" : repr(e.__cause__)[:250],
            })
            mlflow.end_run(status = "FAILED")
            raise

        mlflow.log_metrics(metrics.to_dict())

        for artifactpath, path in artifactpaths.items():
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
