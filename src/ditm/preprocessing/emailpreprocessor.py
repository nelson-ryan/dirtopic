import re
from .. import config
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from .preprocessor import Preprocessor
from ..db.tables import EmailCols
from ..util import timeit
from ..contenttype import ContentType


class EmailPreprocessor(Preprocessor):
    """
    """
    content_type = ContentType.email

    addressfields = [ "ToAddresses", "FromAddress" ]

    _RE = {
        # reply
        re.compile(r"(^>.*$|^-{3,}.*$|^On .+wrote:.*$)",
            re.MULTILINE | re.IGNORECASE
        ) : " ",
        # html
        re.compile(r"<[^>]+>") : " ",
        # url
        re.compile(r"https?://\S+") : " ",
        # email headers
        re.compile(r"^(?:From|To|Sent|Subject):.*?$", re.IGNORECASE) : " ",
    }

    _SIG_MARKERS = [
        "--", "sent from my", "unsubscribe", "regards,", "best,", "thanks,",
        "©",
    ]
    SIG_REGEX = re.compile(
        # multiline regex identifies any match at the start of the line
        # wildcard at the end matches all following text
        r"^(?:" + r"|".join(re.escape(s) for s in _SIG_MARKERS) + ").*",
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    @timeit
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Replace regex patterns.
        """
        logger.info("Starting clean_text")

        df['document'] = (
            df['document']
            .str.lower()
            .str.strip()
            .str.replace(pat = self._RE, regex = True)
        )

        # Remove lines of a given document after any signature marker
        # delete everything in the text after a signature marker
        df['document'] = df['document'].str.replace(
            pat = self.SIG_REGEX,
            repl = "",
            regex = True,
        )

        return df


    @timeit
    def build_stopwords(
        self,
        rawdf: pd.DataFrame,
        extra_stopwords: frozenset = frozenset()
    ) -> frozenset:
        """ Emails extract from and to addresses to include in stopwords
            This was seemingly motivated by an earlier verions of
            the dataset because full names were included in the email
            address format, which is no longer the case.
            See bertopic_claude.py for that coe
            This may be defunct now. For now, it is a no-op.
        """
        #
        # addresses = (
        #     pd.concat([
        #         rawdf[addressfield]
        #         for addressfield in self.addressfields
        #         if addressfield in rawdf.columns
        #     ])
        #     .dropna()
        #     .str.split(";")
        #     .explode()
        #     .str.strip()
        #     .unique()
        # )
        return frozenset(
            set() | extra_stopwords
        )
