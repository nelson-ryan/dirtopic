from ..util import timeit
from ..dates import DateWindow, datewindow
from .. import params
from .. import config
from .corpuscomponent import CorpusComponent
from ..loaders import Loader
from ..preprocessing import Preprocessor
from typing import Iterable, Self
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class Corpus(CorpusComponent):
    """ A composite "Leaf," this represents one corpus type and source

    loader -- a Loader to load the corpus data from its source
    preprocessor -- a Preprocessor to appropriately clean and extract
    data based on the corpus type (e.g. emails versus transcripts).
    """

    def __init__(
        self,
        loader: Loader,
        preprocessor: Preprocessor,
        *,
        datewindow: DateWindow = datewindow(),
        min_tokens: int = 2,
    ):

        self.loader = loader
        self.preprocessor = preprocessor
        if loader.content_type != preprocessor.content_type:
            msg = (
                f"Loader/Preprocessor mismatch in {type(self).__name__}: "
                f"{type(self.loader).__name__} is "
                f"{self.loader.content_type}; "
                f"{type(self.preprocessor).__name__} is "
                f"{self.preprocessor.content_type}"
            )
            logger.error("msg")
            raise TypeError(msg)
        self._data: pd.DataFrame | None = None
        self.stopwords: frozenset[str] = frozenset()

        self.datewindow = datewindow
        self.min_tokens = min_tokens

        self._loaded: bool = False
        self._lemmatized: bool = False
        self._cleaned: bool = False

    def __len__(self) -> int:
        if self._data is not None:
            return len(self._data)
        return 0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"({self.loader!r}, "
            f"{self.preprocessor!r}, "
            f"years = {self.datewindow!r}, "
            f"min_tokens = {self.min_tokens})"
        )


    @timeit
    def load(self):
        """ Load the data as defined in the Loader.
        """
        self._data = self.loader.load(datewindow = self.datewindow)
        self.loader.df = None # TODO Change Loaders to return, not save, the df
        self._loaded = True
        self._cleaned = False
        self._lemmatized = False
        if self._data.empty:
            logger.warning("Corpus empty")
        return self


    @timeit
    def build_stopwords(self,
        extra_stopwords: frozenset[str] = frozenset()
    ) -> frozenset[str]:

        if not self._loaded:
            raise RuntimeError("Must load before building stopwords")
        self.stopwords = self.preprocessor.build_stopwords(
            rawdf = self._data,
            extra_stopwords = extra_stopwords
        )
        return self.stopwords


    @timeit
    def preprocess_docs(
        self,
        extra_stopwords: frozenset[str] = frozenset(),
        split_sents: bool = True,
    ) -> Self:

        logger.info(f"Starting Corpus preprocessing {self!r}")
        if not self._loaded or self._data is None:
            raise ValueError("Must load before processing documents")
        self._data = self.preprocessor.clean_text(df = self._data)
        self._cleaned = True
        self._data = self.preprocessor.lemmatize(
            df = self._data,
            remove_stop = True,
            extra_stopwords = self.stopwords | extra_stopwords,
            split_sents = split_sents,
        )
        self._data['n_tokens'] = self._data['document'].str.count(" ") + 1
        self._lemmatized = True
        logger.info(
            f"Processed {type(self).__name__} "
            f"'{self.preprocessor.content_type}'; Total docs: {len(self)}"
        )
        logger.info(
            f"Doc count at or above {self.min_tokens} min_tokens: "
            f"{self._n_at_min_tokens()} "
        )
        return self

    def _min_token_mask(self) -> pd.Series:
        """ """
        if not (self._cleaned and self._lemmatized) or self._data is None:
            raise RuntimeError("Cannot access data before processing")
        else:
            return self._data['n_tokens'] >= self.min_tokens

    def _n_at_min_tokens(self) -> int:
        return int(sum(self._min_token_mask()))

    @property
    def documents(self) -> Iterable[str]:
        if self._data is not None:
            yield from self._data.loc[self._min_token_mask(), 'document']


    @property
    def documents_untokenized(self) -> Iterable[str]:
        if self._data is not None:
            yield from self._data.loc[self._min_token_mask(), 'document_untokenized']


    @property
    def timestamps(self) -> Iterable[pd.Timestamp]:
        if self._data is not None:
            yield from self._data.loc[self._min_token_mask(), 'timestamp']

    def frame(self) -> pd.DataFrame:
        """ Output DataFrame
        """
        if self._data is None:
            raise RuntimeError("Cannot access data before processing")
        return self._data.loc[
            self._min_token_mask(), self._data.columns != 'n_tokens'
        ]
