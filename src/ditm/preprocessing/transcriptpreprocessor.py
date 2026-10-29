import re
import pandas as pd
from .preprocessor import Preprocessor
from ..db.tables import TranscriptCols
import logging
from ..util import timeit
from ..contenttype import ContentType
logger = logging.getLogger(__name__)

class TranscriptPreprocessor(Preprocessor):

    content_type = ContentType.transcript

    # GO pattern matches GO lines
    GORE = re.compile(r"^.*?{.*?<GO>}$", re.MULTILINE)
    # Operator pattern matches operator lines and everything else after
    GOOP = re.compile(r"^\s*Operator\s*$.*", re.MULTILINE | re.DOTALL)

    @timeit
    def build_stopwords(
        self,
        rawdf: pd.DataFrame,
        extra_stopwords: frozenset = frozenset(),
    ) -> frozenset:
        """ No-op """
        return frozenset(
            set() | extra_stopwords
        )


    @timeit
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Replace any non-contentful patterns
        """
        logger.info("Starting clean_text")

        # Split the transcript into speaker-turns, leaving out the first bit
        df['document'] = df['document'].str.split(
            pat = self.GORE,
            regex = True,
        ).str[1:-1]
        # explode creates new rows for each speaker-turn
        df = df.explode('document', ignore_index = False)
        # Remove 'Operator' turns from those
        df['document'] = df['document'].str.replace(
            pat = self.GOOP,
            repl = "",
            regex = True,
        )
        # clear empties
        df = df.loc[
            df['document'].notna() &
            df['document'].str.strip().ne("")
        ]
        return df
