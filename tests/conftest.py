
import pytest

import ditm
from ditm import (
    config,
    datewindow,
    DateWindow
)
import logging
logger = logging.getLogger()


@pytest.fixture()
def csv_loaded():
    c = ditm.Corpus(
        loader = ditm.CSVLoader(path = config.CSV_PATH),
        preprocessor = ditm.EmailPreprocessor(),
        datewindow = datewindow(display_years = 1),
    )
    c.load()     # calls via Loader, validates
    c._data = c._data.iloc[:100,:]
    return c

@pytest.fixture
def csv_processed(csvcorpus_loaded):
    c = csvcorpus_loaded
    c.preprocess_docs()
    return c


@pytest.fixture
def emaildb_loaded():
    c = ditm.Corpus(
        loader = ditm.EmailDbLoader(),
        preprocessor = ditm.EmailPreprocessor(),
        datewindow = datewindow(display_years = 1),
    )
    c.load()
    c._data = c._data.iloc[:10,:]
    return c

@pytest.fixture
def emaildb_processed(emaildb_loaded):
    c = emaildb_loaded
    c.preprocess_docs()
    return c


@pytest.fixture
def transcriptdb_loaded():
    c = ditm.Corpus(
        loader = ditm.TranscriptDbLoader(),
        preprocessor = ditm.TranscriptPreprocessor(),
        datewindow = datewindow(display_years = 1),
    )
    c.load()
    c._data = c._data.iloc[:10,:]
    return c

@pytest.fixture
def transcriptdb_processed(transcriptdb_loaded):
    c = transcriptdb_loaded
    c.preprocess_docs()
    return c


@pytest.fixture
def mlflow_tmp(tmp_path):
    """ Tests an isolated, ephemeral mlflow instance, not authentication """
    import mlflow
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp_id = mlflow.create_experiment(
        "pytest test",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    yield tmp_path, exp_id
    mlflow.set_tracking_uri(prev_uri)

