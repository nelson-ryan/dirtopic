"""
Changes in topic prominence over time, with small topics on an equal footing.

Each period is compared with a baseline made of the periods before it, using
the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi &
Quinn 2008, "Fightin' Words", section 3.5.1). Topics play the role of words;
the current period and its baseline play the role of the two groups.

Two numbers per (period, topic):
  z      Evidence that the change is more than noise. Used ONLY to decide
         whether a change is flagged. The threshold comes from shuffled data
         and keeps the chance of any false flag anywhere in the table at
         1 - level (default 5%).
  ratio  Size of the change: roughly how many times more (>1) or less (<1)
         prominent the topic is than in its baseline. Used for ranking and
         coloring, so a small topic that triples outranks a big topic that
         grows 10%.

Usage, with a fitted BERTopic model:

    changes = topic_changes(
        model.topics_,
        timestamps,
        alpha0 = 500,
        k = 12,
    )
    words = bertopic_words(model)
    changes.flagged(words)                             # DataFrame, biggest first
    changes.heatmap(words).write_html("changes.html")

Only model.topics_ and the timestamps are needed to rerun this later; the
model itself need not be saved (words are only for labels).
"""
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

import numpy as np
import pandas as pd

from ..util import timeit

Direction = Literal["rise", "fall", "both"]
Baseline = Literal["rolling", "growing"]

TopicLabels = Union[Sequence[int], np.ndarray]
Timestamps = Union[Sequence[Any], np.ndarray, pd.Series, pd.Index]
TopicWords = Mapping[int, str]


@timeit
def topic_changes(
    topics: TopicLabels,
    timestamps: Timestamps,
    *,
    alpha0: float,
    direction: Direction = "rise",
    freq: str = "Q",
    baseline: Baseline = "rolling",
    k: int = 12,
    include_outliers: bool = False,
    n_perm: int = 1000,
    level: float = 0.95,
    seed: int = 1337,
    progress: bool = True,
) -> "TopicChanges":
    """
    Score every (period, topic) against its baseline and compute the
    threshold for flagging.

    Parameters
    ----------
    topics
        One topic label per document (e.g. BERTopic's model.topics_).
    timestamps
        One timestamp per document, same order as `topics`.
    alpha0
        Prior strength. Each topic's estimate is pulled toward its share of
        the period plus baseline by the fraction alpha0 / (n + alpha0),
        where n is the number of documents in the period. The pull is the
        same fraction for big and small topics; it mostly affects periods
        with few documents.
        At our scale, it primarily serves to prevent 0 counts from breaking math.
    direction
        Which changes to look for:
          "rise"  increases only
          "fall"  decreases only
          "both"  either
        The threshold guards only against false flags in this direction,
        so "rise" gives a threshold no higher than "both" (sometimes equal,
        sometimes lower).
    freq
        pandas period frequency ("M" = calendar month, "W" = week, ...).
        Periods with no documents still count as periods.
    baseline
        "rolling"  the k periods immediately before
        "growing"  every period before; the first k periods are still left
                   unscored so early estimates rest on enough history
    k
        Baseline length ("rolling") or minimum history ("growing").
    include_outliers
        Whether BERTopic's outlier label -1 counts as a topic. If False,
        outlier documents are dropped before anything is counted.
    n_perm
        Number of shuffles used to set the threshold. Each shuffle recounts
        labels only (the topic model is never refit). More shuffles give a
        more stable threshold; the per-shuffle results are kept in
        .null_maxima so their spread can be inspected.
    level
        0.95 -> 5% chance of any false flag in the whole table.
    seed
        Random seed, for reproducibility.
    progress
        Print progress through the shuffles.
    """
    _check_choice(direction, Direction, "direction")
    _check_choice(baseline, Baseline, "baseline")

    p, t, periods, labels = _encode(
        topics,
        timestamps,
        freq,
        include_outliers,
        k,
    )
    P, T = len(periods), len(labels)

    counts = _count(p, t, P, T)
    base = _baseline(counts, baseline, k)
    z, delta = _z(counts, base, alpha0)

    if baseline == "rolling":
        n_base = np.full(P, float(k))
    else:
        n_base = np.arange(P, dtype = float)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        base_avg = base / n_base[:, None]

    # Threshold: shuffling topic labels across documents removes any link
    # between topic and time but keeps
    #   - every period's size
    #   - every topic's total
    # The largest score in each shuffled table is what pure chance can
    # produce; `level` of those maxima fall below the threshold.
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_perm)
    step = max(n_perm // 10, 1)
    for i in range(n_perm):
        c = _count(p, rng.permutation(t), P, T)
        zp, _ = _z(c, _baseline(c, baseline, k), alpha0)
        maxima[i] = np.nanmax(_directed(zp, direction))
        if progress and (i + 1) % step == 0:
            print(f"shuffle {i + 1}/{n_perm}", flush = True)

    def frame(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(a, index = periods, columns = labels)

    return TopicChanges(
        z = frame(z),
        ratio = frame(np.exp(delta)),
        counts = frame(counts),
        baseline_avg = frame(base_avg),
        threshold = float(np.quantile(maxima, level)),
        direction = direction,
        null_maxima = maxima,
        settings = dict(
            alpha0 = alpha0,
            freq = freq,
            baseline = baseline,
            k = k,
            include_outliers = include_outliers,
            n_perm = n_perm,
            level = level,
            seed = seed,
        ),
    )


def bertopic_words(model: Any, n_words: int = 5) -> Dict[int, str]:
    """{topic: "word, word, ..."} from a fitted BERTopic model, for labels."""
    return {
        t: ", ".join(w for w, _ in ws[:n_words])
        for t, ws in model.get_topics().items()
        if t != -1
    }


@dataclass
class TopicChanges:
    """
    Result of topic_changes(). Tables are periods (rows) x topics (columns).

    z             evidence score; decides flagging only
    ratio         size of change vs baseline (2.0 = twice as prominent)
    counts        documents in the period
    baseline_avg  average documents per baseline period

    NaN in z and ratio means not scored, which happens for:
      - the first k periods
      - periods with no documents
      - baselines with no documents
      - topics absent from both the period and its baseline
    """
    z: pd.DataFrame
    ratio: pd.DataFrame
    counts: pd.DataFrame
    baseline_avg: pd.DataFrame
    threshold: float
    direction: Direction
    null_maxima: np.ndarray
    settings: Dict[str, Any]

    def mask(self) -> pd.DataFrame:
        """True where a change is flagged (NaN cells are never flagged)."""
        flag = _directed(self.z.to_numpy(), self.direction) > self.threshold
        return pd.DataFrame(
            flag,
            index = self.z.index,
            columns = self.z.columns,
        )

    def flagged(self, words: Optional[TopicWords] = None) -> pd.DataFrame:
        """One row per flagged change, largest change first."""
        pi, ti = np.where(self.mask().to_numpy())
        df = pd.DataFrame({
            "period": self.z.index[pi].astype(str),
            "topic": self.z.columns[ti],
            "change": self.ratio.to_numpy()[pi, ti],
            "docs": self.counts.to_numpy()[pi, ti].astype(int),
            "baseline_avg_docs": self.baseline_avg.to_numpy()[pi, ti],
            "z": self.z.to_numpy()[pi, ti],
        })
        if words is not None:
            df.insert(2, "words", [words.get(t, "") for t in df.topic])
        size = _directed(np.log(df.change.to_numpy()), self.direction)
        order = np.argsort(-size, kind = "stable")
        return df.iloc[order].reset_index(drop = True)

    @timeit
    def heatmap(
        self,
        words: Optional[TopicWords] = None,
        max_topics: Optional[int] = None,
    ) -> Any:
        """
        Topics x periods, returned as a plotly Figure.

        Cell colors:
          - red          flagged rise; deeper = larger (log scale)
          - blue         flagged fall; deeper = larger (log scale)
          - white        scored but not flagged
          - grey         not scored

        Cells are separated by darker grey gridlines.

        Rows are topics with at least one flagged change, largest change at
        the top. Hover shows details for every cell.
        """
        import plotly.graph_objects as go

        # no reason to plot the initial baseline periods, which are always grey
        k = self.settings["k"]
        periods = self.z.index[k:]

        flag = self.mask().to_numpy()[k:]
        z = self.z.to_numpy()[k:]
        ratio = self.ratio.to_numpy()[k:]
        counts = self.counts.to_numpy()[k:]
        avg = self.baseline_avg.to_numpy()[k:]
        log2r = np.log2(ratio)

        size = np.where(flag, _directed(log2r, self.direction), -np.inf)
        best = size.max(axis = 0)
        order = [
            j for j in np.argsort(-best, kind = "stable")
            if flag[:, j].any()
        ]
        if not order:
            raise ValueError("No flagged changes to plot.")
        if max_topics:
            order = order[:max_topics]

        topics = self.z.columns[order]
        x = [str(q) for q in periods]
        if words:
            y = [f"{t}: {words.get(t, '')[:40]}" for t in topics]
        else:
            y = [str(t) for t in topics]

        hover: List[List[str]] = []
        for j, t in zip(order, topics):
            label = f"<b>Topic {t}</b>"
            if words:
                label += f"<br>{words.get(t, '')}"
            row: List[str] = []
            for i, q in enumerate(x):
                if np.isnan(z[i, j]):
                    row.append(f"{label}<br>{q}: not scored")
                    continue
                note = "" # " (FLAGGED)" if flag[i, j] else ""
                row.append(
                    f"{label}"
                    f"<br>{q}: {ratio[i, j]:.2f}x baseline{note}"
                    f"<br>{int(counts[i, j])} docs "
                    f"(baseline avg {avg[i, j]:.1f})"
                    f"<br>z = {z[i, j]:.1f}"
                )
            hover.append(row)

        # Layer 1: white = scored, grey = not scored. Carries all hover text.
        bg = np.isnan(z[:, order]).T.astype(float)
        fig = go.Figure(go.Heatmap(
            z = bg,
            x = x,
            y = y,
            zmin = 0,
            zmax = 1,
            showscale = False,
            colorscale = [
                [0, "#ffffff"],
                [1, "#d9d9d9"],
            ],
            text = hover,
            hovertemplate = "%{text}<extra></extra>",
        ))

        # Layer 2: flagged cells only. color scales start at a visible
        # tint so a small flagged change is never mistaken for white.
        vals = np.where(flag, log2r, np.nan)[:, order].T
        hi = max(float(np.nanmax(np.abs(vals))), 0.1)
        light_red, dark_red = "#fcbba1", "#99000d"
        dark_blue, light_blue = "#08306b", "#c6dbef"
        scale: List[List[Any]]
        if self.direction == "rise":
            scale = [
                [0, light_red],
                [1, dark_red],
            ]
            lo, top = 0.0, hi
        elif self.direction == "fall":
            scale = [
                [0, dark_blue],
                [1, light_blue],
            ]
            lo, top = -hi, 0.0
        else:
            scale = [
                [0, dark_blue],
                [0.5, light_blue],
                [0.5, light_red],
                [1, dark_red],
            ]
            lo, top = -hi, hi

        candidate_ratios = (
            1 / 16,
            1 / 8,
            1 / 4,
            1 / 2,
            2 / 3,
            4 / 5,
            1,
            1.25,
            1.5,
            2,
            4,
            8,
            16,
        )
        ticks = [
            r for r in candidate_ratios
            if lo - 1e-9 <= np.log2(r) <= top + 1e-9
        ]
        fig.add_trace(go.Heatmap(
            z = vals,
            x = x,
            y = y,
            zmin = lo,
            zmax = top,
            colorscale = scale,
            hoverinfo = "skip",
            colorbar = dict(
                title = "Change vs<br>baseline",
                tickvals = [np.log2(r) for r in ticks],
                ticktext = [f"{r:.2g}x" for r in ticks],
            ),
        ))

        # Gridlines: real lines on cell boundaries, drawn above both layers.
        # On category axes, cell i spans i - 0.5 to i + 0.5.
        grid_color = "#a6a6a6"
        grid = dict(
            color = grid_color,
            width = 1,
        )
        # for b in np.arange(len(x) + 1) - 0.5:
        #     fig.add_shape(
        #         type = "line",
        #         x0 = b,
        #         x1 = b,
        #         y0 = -0.5,
        #         y1 = len(y) - 0.5,
        #         line = grid,
        #         layer = "above",
        #     )
        for b in np.arange(len(y) + 1) - 0.5:
            fig.add_shape(
                type = "line",
                x0 = -0.5,
                x1 = len(x) - 0.5,
                y0 = b,
                y1 = b,
                line = grid,
                layer = "above",
            )

        s = self.settings
        title = (
            f"Flagged: {
                "rises/falls" if self.direction == "both" else f"{self.direction}s"
            } in topic prominence; "
            f"|z| threshold {self.threshold:.2f}, "
            f"relative to {s['baseline']} baseline of {s['k']} preceding periods, "
            f"alpha0={s['alpha0']:g}"
        )
        fig.update_layout(
            title = title,
            height = max(400, 20 * len(y) + 150),
            plot_bgcolor = "white",
            xaxis = dict(
                type = "category",
                title = "Period",
                showgrid = False,
            ),
            yaxis = dict(
                type = "category",
                autorange = "reversed",
                title = "Topic",
                showgrid = False,
            ),
        )
        return fig


# ---- internals ------------------------------------------------------------

def _check_choice(value: str, choices: Any, name: str) -> None:
    """Raise if `value` is not one of the options of a Literal type."""
    options = get_args(choices)
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


def _directed(z: np.ndarray, direction: Direction) -> np.ndarray:
    """Orient scores so that 'larger' means 'more of what we look for'."""
    if direction == "rise":
        return z
    if direction == "fall":
        return -z
    return np.abs(z)


def _encode(
    topics: TopicLabels,
    timestamps: Timestamps,
    freq: str,
    include_outliers: bool,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, pd.PeriodIndex, np.ndarray]:
    """
    Convert labels and timestamps to integer indices.

    Returns:
      - period index of each document
      - topic index of each document
      - every period in the data's range, including empty ones
      - topic labels, in the order of the topic indices
    """
    labels_in = np.asarray(topics)
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(labels_in) != len(ts):
        raise ValueError(f"{len(labels_in)} topics vs {len(ts)} timestamps")
    if ts.isna().any():
        raise ValueError("timestamps contain missing values")
    if not include_outliers:
        keep = labels_in != -1
        labels_in, ts = labels_in[keep], ts[keep]
    per = ts.to_period(freq)
    periods = pd.period_range(per.min(), per.max(), freq = freq)
    if k < 1 or k >= len(periods):
        raise ValueError(
            f"k={k} leaves no scorable periods "
            f"(data spans {len(periods)} periods)"
        )
    labels, t_idx = np.unique(labels_in, return_inverse = True)
    return periods.get_indexer(per), t_idx, periods, labels


def _count(
    p: np.ndarray,
    t: np.ndarray,
    n_periods: int,
    n_topics: int,
) -> np.ndarray:
    """Documents per (period, topic), as a periods x topics float array."""
    flat = np.bincount(
        p * n_topics + t,
        minlength = n_periods * n_topics,
    )
    return flat.reshape(n_periods, n_topics).astype(float)


def _baseline(counts: np.ndarray, kind: Baseline, k: int) -> np.ndarray:
    """Baseline counts per period; NaN rows for the first k periods."""
    cum = np.vstack([
        np.zeros((1, counts.shape[1])),
        counts.cumsum(axis = 0),
    ])
    base = np.full_like(counts, np.nan)
    i = np.arange(k, counts.shape[0])      # cum[i] = sum of periods 0..i-1
    if kind == "rolling":
        base[i] = cum[i] - cum[i - k]
    else:
        base[i] = cum[i]
    return base


def _z(
    y: np.ndarray,
    yb: np.ndarray,
    alpha0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monroe et al. log-odds difference (delta) and z-score, current counts y
    vs baseline counts yb. Returns (z, delta), NaN where not estimable.
    """
    n = y.sum(axis = 1, keepdims = True)     # documents in current period
    nb = yb.sum(axis = 1, keepdims = True)   # documents in baseline
    with np.errstate(divide = "ignore", invalid = "ignore"):
        a = alpha0 * (y + yb) / (n + nb)     # prior pseudo-counts
        delta = (
            np.log((y + a) / (n + alpha0 - y - a))
            - np.log((yb + a) / (nb + alpha0 - yb - a))
        )
        z = delta / np.sqrt(1 / (y + a) + 1 / (yb + a))
    bad = ~np.isfinite(z) | (n == 0) | (nb == 0)
    z[bad] = np.nan
    delta[bad] = np.nan
    return z, delta
