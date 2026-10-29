from bertopic import BERTopic
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud
import seaborn as sns
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
import pandas as pd
from ditm.config import logging
from ditm import config

# -- 4. Build human-readable topic labels ------------------------------------
def make_topic_labels(topic_model: BERTopic) -> dict[int, str]:
    labels = {}
    for tid in topic_model.get_topics():
        if tid == -1:
            labels[tid] = "Outlier / Noise"
            continue
        words = [w for w, _ in topic_model.get_topic(tid)[:4]]
        labels[tid] = " · ".join(words)
    return labels

# -- 5. Heatmap: topics × time ------------------------------------------------
def plot_heatmap(df: pd.DataFrame, topic_labels: dict, path: str):
    df2 = df[df["topic"] != -1].copy()
    df2["period"] = df2["Date.Received"].dt.to_period(config.TIME_BUCKET)

    # pick top N topics by total email count
    top_topics = (
        df2["topic"].value_counts().head(config.TOP_N_TOPICS).index.tolist()
    )
    df2 = df2[df2["topic"].isin(top_topics)]

    pivot = (
        df2.groupby(["period", "topic"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = pivot.index.astype(str)
    pivot.columns = [topic_labels.get(c, str(c)) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(14, len(pivot) * 0.7), max(8, len(pivot.columns) * 0.5)))
    sns.heatmap(
        pivot.T,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="#e0e0e0",
        annot=True,
        fmt="d",
        ax=ax,
        cbar_kws={"label": "Email Count"},
    )
    ax.set_title("Email Topic Trends Over Time", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Heatmap saved → {path}")

# -- 6. WordClouds ------------------------------------------------------------
def plot_wordclouds(topic_model: BERTopic, topic_labels: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    palettes = list(mcolors.TABLEAU_COLORS.values())

    for tid, label in topic_labels.items():
        if tid == -1:
            continue
        word_weights = dict(topic_model.get_topic(tid))
        if not word_weights:
            continue

        color = palettes[tid % len(palettes)]
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            color_func=lambda *a, **kw: color,
            max_words=60,
            prefer_horizontal=0.85,
        ).generate_from_frequencies(word_weights)

        fname = f"{out_dir}topic_{tid:02d}.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {tid}: {label}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()

    logging.info(f"  WordClouds saved → {out_dir}")

# -- 7. Excel export ----------------------------------------------------------
HEADER_FILL   = PatternFill("solid", fgColor="2D4A8A")
HEADER_FONT   = Font(color="FFFFFF", bold=True)
ALT_FILL      = PatternFill("solid", fgColor="EEF2FA")

def _style_header(ws):
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

def _autofit(ws):
    for col in ws.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=10)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 50)

def export_excel(df: pd.DataFrame, topic_model: BERTopic, topic_labels: dict, path: str):
    wb = openpyxl.Workbook()

    # -- Sheet 1: per-email detail ------------------------------------------
    ws1 = wb.active
    ws1.title = "Email Details"
    cols = ["Date.Received", "Address.To", "Header.Subject",
            "topic", "topic_label", "topic_probability"]
    ws1.append(cols)
    _style_header(ws1)

    for i, row in df.iterrows(): # Do not use iterrows
        ws1.append([
            row["Date.Received"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["Date.Received"]) else "",
            str(row.get("Address.To", "")),
            str(row.get("Header.Subject", "")),
            int(row["topic"]),
            row["topic_label"],
            round(float(row["topic_probability"]), 4),
        ])
        if i % 2 == 0:
            for cell in ws1[i + 2]:
                cell.fill = ALT_FILL

    ws1.freeze_panes = "A2"
    _autofit(ws1)

    # -- Sheet 2: topic summary ---------------------------------------------
    ws2 = wb.create_sheet("Topic Summary")
    ws2.append(["Topic ID", "Label", "Top Keywords", "Email Count", "% of Total"])
    _style_header(ws2)

    total = len(df[df["topic"] != -1])
    for tid, label in sorted(topic_labels.items()):
        if tid == -1:
            continue
        count   = int((df["topic"] == tid).sum())
        pct     = round(count / total * 100, 1) if total else 0
        keywords = ", ".join(w for w, _ in topic_model.get_topic(tid)[:10])
        ws2.append([tid, label, keywords, count, pct])

    ws2.freeze_panes = "A2"
    _autofit(ws2)

    # -- Sheet 3: time-series pivot -----------------------------------------
    ws3 = wb.create_sheet("Topic Trends")
    df3  = df[df["topic"] != -1].copy()
    df3["period"] = df3["Date.Received"].dt.to_period(config.TIME_BUCKET).astype(str)
    pivot = (
        df3.groupby(["period", "topic_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ws3.append(list(pivot.columns))
    _style_header(ws3)
    for row in pivot.itertuples(index=False):
        ws3.append(list(row))
    _autofit(ws3)

    wb.save(path)
    logging.info(f"  Excel saved → {path}")
