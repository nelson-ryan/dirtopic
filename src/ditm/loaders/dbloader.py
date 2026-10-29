from .loader import Loader
from ..config import DB_URL
from ..dates import DateWindow
from ..db import (
    get_engine,
    emailmessages,
    externalsystems,
    contentexports,
)
from ..db.tables import (
    EmailCols,
    TranscriptCols,
    ContentCols,
    ExternalSystemsCols,
    ContentExportCols,
    ExternalSystemName
)
from ..util import timeit
from ..contenttype import ContentType
from pathlib import Path
import abc
from functools import cached_property
import pandas as pd
from mssql_python import connect
from sqlalchemy import select, and_, Table
import logging
logger = logging.getLogger(__name__)


# TODO take connection string as input
class DbLoader(Loader):
    """ Defines query structure (using SQLAlchemy) for database.
    Subclasses specify only the table and columns that are queried.
    Tables and columns are defined in ..db.tables
    Subclasses must define _normalize_dataframe to set document and timestamp
    columns, and set ContentId as index.
    """

    table: Table
    tablecols: type[EmailCols] | type[TranscriptCols]


    def __init__(self, *args, db_url: str = DB_URL, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_url: str = db_url
        self.database: str = db_url.split("?")[0].split("/")[-1]
        self.engine = get_engine(db_url)
        self.df: pd.DataFrame | None = None
        # This flag allows suppressing writing the export to db
        self._log_export = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.db_url!r})"


    @timeit
    def _load(self, datewindow: DateWindow) -> pd.DataFrame:
        """Read in the corpus from the database.
        Sets ContentId as DataFrame index.
        Data validation takes place in Preprocessor.
        """
        if self.df is not None:
            return self.df

        logger.info(f"Loading corpus from {self.database}")

        # Define query:
        query = (
            # Reads columns as specified in tablecols, defined in db.tables
            select(*[self.table.c[col] for col in self.tablecols])
            .where(
                self.table.c[self.tablecols.timestamp] >= datewindow.data_start,
                self.table.c[self.tablecols.timestamp] < datewindow.end,
            )
        )

        self.df = pd.read_sql(
            sql = query,
            con = self.engine,
        )

        if self._log_export:
            self._write_contentexports()

        self.df.set_index(self.tablecols.contentid, inplace = True)
        return self.df


    @cached_property
    def _externalsystemsid(self):
        """ Gets the ExternalSystemId based on the SystemName.
        """
        with self.engine.connect() as conn:
            query = (
                select(
                    externalsystems.c[ExternalSystemsCols.externalsystemid]
                ).where(
                    externalsystems.c[ExternalSystemsCols.systemname] ==
                    ExternalSystemName
                )
            )
            id = conn.execute(query).scalar_one()
        return id


    @timeit
    def _write_contentexports(self) -> None:
        logger.info("Writing ContentExports to database")

        if self.df is None:
            raise RuntimeError("Cannot write export without having exported.")
        export_df = pd.DataFrame({
            ContentExportCols.contentid : self.df[self.tablecols.contentid],
            ContentExportCols.externalsystemid  : self._externalsystemsid,
            ContentExportCols.exportstatus : "Success",
            ContentExportCols.exportdateutc : pd.to_datetime("now", utc = True),
        })
        export_df.to_sql(
            name = contentexports.name,
            con = self.engine,
            if_exists = "append",
            index = False
        )

    # These make the Loader picklable
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['engine']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.engine = get_engine(self.db_url)
