from . import DbLoader
from ..db.tables import emailmessages, EmailCols
from ..util import timeit
from ..contenttype import ContentType
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class EmailDbLoader(DbLoader):
    table = emailmessages
    tablecols = EmailCols

    content_type = ContentType.email

    @timeit
    def _normalize_dataframe(self) -> pd.DataFrame:
        """ To include subject lines and body text alike,
        combine both into a single column separated by a period.
        Downstream, Preprocessor.lemmatize() splits them as sparate docs.
        """
        if self.df is None:
            raise TypeError(
                f"Can't normalize nothing; no data in {type(self).__name__}"
            )

        self.df['document'] = (
            self.df.pop(EmailCols.emailsubject)
            .str.cat(
                self.df.pop(EmailCols.bodytext),
                sep = ". ",
                na_rep = "",
            )
        )
        self.df['timestamp'] = (
            pd.to_datetime(
                self.df.pop(EmailCols.timestamp),
                errors = "coerce"
            )
        )
        return self.df
