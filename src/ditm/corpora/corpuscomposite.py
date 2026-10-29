
from ..util import timeit
from .corpuscomponent import CorpusComponent
from typing import Iterable, Self
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class CorpusComposite(CorpusComponent):
    """ Composite: a list of CorpusComponents (Corpora)
        with the same methods as the leaf.
        BERTopic/REITopic can use this directly without worrying about
        internal structure (one corpus or several)
    """

    def __init__(self, corpora: list[CorpusComponent]):
        self.corpora = corpora
        if len({c.datewindow for c in corpora}) != 1:
            raise ValueError("Corpora have inconsistent time windows")
        self.stopwords: frozenset[str] = frozenset()

    def __len__(self) -> int:
        return sum(len(c) for c in self.corpora)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.corpora!r})"


    @timeit
    def load(self) -> Self:
        for c in self.corpora:
            c.load()
        if all(c._data.empty for c in self.corpora):
            msg = "All corpora data imports returns no rows"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @timeit
    def build_stopwords(self, extra_stopwords: frozenset[str] = frozenset()
    ) -> frozenset[str]:
        merged = set()
        for c in self.corpora:
            merged |= c.build_stopwords()
        self.stopwords = frozenset(merged)
        return self.stopwords

    @timeit
    def preprocess_docs(
        self,
        extra_stopwords: frozenset[str] = frozenset(),
        split_sents: bool = True,
    ) -> Self:
        for c in self.corpora:
            c.preprocess_docs(
                extra_stopwords = extra_stopwords | self.stopwords,
                split_sents = split_sents,
            )
        logger.info(
            f"Processed {type(self).__name__}; Total docs: {len(self)}\n"
        )
        logger.info(
            f"Doc total count at or above min_tokens: {self._n_at_min_tokens()}"
        )
        return self

    def _n_at_min_tokens(self) -> int:
        return sum(c._n_at_min_tokens() for c in self.corpora)

    @property
    def timestamps(self) -> Iterable[pd.Timestamp]:
        for c in self.corpora:
            yield from c.timestamps

    @property
    def documents(self) -> Iterable[str]:
        for c in self.corpora:
            yield from c.documents

    @property
    def documents_untokenized(self) -> Iterable[str]:
        for c in self.corpora:
            yield from c.documents_untokenized

    def frame(self) -> pd.DataFrame:
        return pd.concat([
            c.frame() for c in self.corpora
        ])
