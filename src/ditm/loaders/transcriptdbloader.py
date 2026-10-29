from . import DbLoader
from ..db.tables import transcripts, TranscriptCols
from ..util import timeit
from ..contenttype import ContentType
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class TranscriptDbLoader(DbLoader):
    table = transcripts
    tablecols = TranscriptCols

    content_type = ContentType.transcript

    @timeit
    def _normalize_dataframe(self):
        """ No special treatment for transcripts;
        content body goes in 'document' column
        """
        if self.df is None:
            raise TypeError(
                f"Can't normalize nothing; no data in {type(self).__name__}"
            )

        self.df = self.df.rename(
            columns = { self.tablecols.contentbody : "document"}
        )
        self.df['timestamp'] = pd.to_datetime(
            self.df.pop(self.tablecols.timestamp),
            errors = "coerce"
        )
        return self.df
