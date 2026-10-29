from .loader import Loader
from ..util import timeit
from ..db.tables import EmailCols
from ..contenttype import ContentType
from ..dates import DateWindow
from pathlib import Path
import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class CSVLoader(Loader):
    """ Loads email data from a specified CSV file.
    Used in early development while the database was being prepared
    """

    content_type = ContentType.email

    def __init__(self, *args, path: str | os.PathLike, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = Path(path)
        self.df: pd.DataFrame | None = None

    def __repr__(self):
        return f"{type(self).__name__}({self.path!r})"


    @timeit
    def _load(self, datewindow: DateWindow) -> pd.DataFrame:
        """Read in the corpus as a csv file.
        Data validation takes place in Preprocessor.
        """
        if self.df is not None:
            return self.df
        logger.info(f"Loading corpus from {self.path}")
        self.df = pd.read_csv(self.path)

        column_correction = {
            # mapping any csv column names (keys) to match db names (values)
            "Date.Display"       : EmailCols.timestamp,
            "Date.Received"      : EmailCols.timestamp,
            "Header.Subject"     : EmailCols.emailsubject,
            "Body.Text"          : EmailCols.bodytext,
        }
        addressfields = [
            "Address.Recipients", "Address.To", "Address.Sender"
        ]

        self.df = self.df.rename(columns = column_correction)
        self.df[EmailCols.timestamp] = pd.to_datetime(
            self.df[EmailCols.timestamp]
        )

        self.df = (
            self.df
            .loc[
                (self.df[EmailCols.timestamp] >= datewindow.data_start) &
                (self.df[EmailCols.timestamp] <= datewindow.end),
            :]
        )

        return self.df

    @timeit
    def _normalize_dataframe(self) -> pd.DataFrame:

        if self.df is None:
            raise TypeError("Can't normalize nothing")

        self.df['document'] = (
            self.df[EmailCols.emailsubject]
            .str.cat(
                self.df[EmailCols.bodytext].astype(str),
                sep = ". ",
                na_rep = "",
            )
        )
        self.df['timestamp'] = (
            pd.to_datetime(
                self.df[EmailCols.timestamp],
                errors = "coerce"
            )
        )

        return self.df
