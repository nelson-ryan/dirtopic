import abc
import pandas as pd
from typing import final
from ..dates import DateWindow
from ..contenttype import ContentType
import logging
logger = logging.getLogger(__name__)

class Loader(abc.ABC):
    """ Abstract class for loading corpora from disparate sources,
    e.g local CSVs, Databases."""

    content_type: ClassVar[ContentType]

    def __init__(self):
        self.df: pd.DataFrame | None = None


    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @final # this is what is called and should not be overriden by subclasses
    def load(self, datewindow: DateWindow) -> pd.DataFrame:
        self._load(datewindow = datewindow)
        self._normalize_dataframe()
        # check that all data is after start
        # this ensures that the filtering is done in each subclass' ._load()
        if not (self.df['timestamp'] >= datewindow.data_start).all():
            msg = f"Unfiltered data from {self!r}\nREQUIRED ACTION: update {
                type(self).__name__
            }._load() to filter on date field"
            logger.error(msg)
            raise ValueError(msg)
        return self.df

    @abc.abstractmethod
    def _load(self, datewindow: DateWindow) -> pd.DataFrame: # subclasses define this
        ...

    @abc.abstractmethod
    def _normalize_dataframe(self) -> pd.DataFrame:
        """ At a minimum, this must append a 'document' column and a 'timestamp'
        column to the dataframe """
        ...
