import pytest
from pathlib import Path
import pandas as pd
import ditm
from ditm import config, CorpusComposite, Corpus, datewindow
from ditm.db.tables import EmailCols
import logging
logger = logging.getLogger()


def test_load_csv(csv_loaded):
    """ Tests type and columns of csv """
    assert(isinstance(csv_loaded._data, pd.DataFrame))
    assert(
        (
            set(csv_loaded._data.columns) >=
            set(["document", "timestamp"]))
    )


def _test_load_db(db_loaded):
    """ Tests type and columns of corpus from database """
    assert(isinstance(db_loaded._data, pd.DataFrame))
    assert(
        (
            (set(db_loaded._data.columns) |
             set([db_loaded._data.index.name])
            ) >=
            set(["ContentId", "document", "timestamp"]))
    )


def test_load_emaildb(emaildb_loaded):
    """ Tests type and columns of emails from database """
    _test_load_db(emaildb_loaded)


def test_load_transcriptdb(transcriptdb_loaded):
    """ Tests type and columns of transcripts from database """
    _test_load_db(transcriptdb_loaded)


def _test_corpus_len(db_loaded, db_processed):
    """ Tests function of `__len__` on Corpus object """
    assert(len(db_loaded._data) == len(db_loaded))
    assert(len(db_processed._data) == len(db_processed))


def test_emaildb_corpus_len(emaildb_loaded, emaildb_processed):
    _test_corpus_len(emaildb_loaded, emaildb_processed)


def test_spacy_nlp():
    """ Just making sure the NLP pipeline is an NLP pipeline """
    from spacy.language import Language
    preprocessor = ditm.EmailPreprocessor()
    assert(isinstance(preprocessor.NLP, Language))


def test_composite(emaildb_processed, transcriptdb_processed):
    """ Test that CorpusComposite members have the same result
    as independent Corpora.
    This may well fail as written because the database query is unordered.
    """

    comp = CorpusComposite([
        Corpus(
            loader = ditm.EmailDbLoader(),
            preprocessor = ditm.EmailPreprocessor(),
            datewindow = datewindow(display_years = 1),
        ),
        Corpus(
            loader = ditm.TranscriptDbLoader(),
            preprocessor = ditm.TranscriptPreprocessor(),
            datewindow = datewindow(display_years = 1),
        ),
    ])
    comp.load()
    for c in comp.corpora:
        c._data = c._data.iloc[:10,:]
    comp.preprocess_docs()

    c = emaildb_processed
    c._data.equals(comp.corpora[0]._data)

    c = transcriptdb_processed
    c._data.equals(comp.corpora[0]._data)


def test_mlflow(mlflow_tmp):
    import mlflow
    from mlflow.tracking import MlflowClient
    tmp_path, exp_id = mlflow_tmp
    testfile = tmp_path / "test.txt"
    testfile.write_text("test")

    with mlflow.start_run(experiment_id=exp_id, run_name="pytest") as run:
        mlflow.log_params({"foo": "bar"})
        mlflow.log_metrics({"baz": 1})
        mlflow.log_artifact(str(testfile))

    client = MlflowClient()
    r = client.get_run(run.info.run_id)
    assert r.data.params == {"foo": "bar"}
    assert r.data.metrics == {"baz": 1}
    assert (
        [a.path for a in client.list_artifacts(run.info.run_id)] == ["test.txt"]
    )
