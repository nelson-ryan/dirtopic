import abc
from typing import Iterable, Self
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class CorpusComponent(abc.ABC):
    """ Base class for corpora
    """

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def load(self) -> Self: ...

    @abc.abstractmethod
    def preprocess_docs(
        self,
        extra_stopwords: frozenset[str] = frozenset(),
        split_sents: bool = True,
    ) -> Self: ...

    @abc.abstractmethod
    def _n_at_min_tokens(self) -> int: ...

    @abc.abstractmethod
    def build_stopwords(
        self,
        extra_stopwords: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        ...

    @property
    @abc.abstractmethod
    def timestamps(self) -> Iterable[pd.Timestamp]: ...

    @property
    @abc.abstractmethod
    def documents(self) -> Iterable[str]: ...

    @property
    @abc.abstractmethod
    def documents_untokenized(self) -> Iterable[str]: ...

    @abc.abstractmethod
    def frame(self) -> pd.DataFrame: ...
