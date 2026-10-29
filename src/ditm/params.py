from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen = True)
class CorpusSpec():
    years: int = 7
    # number of quarters for timescale comparison
    baseline_quarters: int = 12
    min_tokens_email: int = 2
    min_tokens_transcript: int = 4

@dataclass(frozen = True)
class UmapParams():
    n_neighbors: int   = 15
    n_components: int  = 5
    min_dist: float    = 0.0
    metric: str        = "cosine"
    random_state: int  = 42
    init: str          = "pca"
    verbose: bool      = True


@dataclass(frozen = True)
class HdbscanParams():
    min_cluster_size: int            = 2000
    min_samples: int                 = 10
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str    = "leaf"
    prediction_data: bool            = False

@dataclass(frozen = True)
class VectorizerParams():
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int                  = 2
    max_df: float                = .3 # surely has no effect unless very low,
                                      # but then would exclude meaningful words
