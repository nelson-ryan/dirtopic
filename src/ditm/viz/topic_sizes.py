import pandas as pd
import plotly.graph_objects as go

def topic_sizes(
    topic_info: pd.DataFrame, 
    n_words: int = 5,
) -> go.Figure:
    info = (
        topic_info.loc[
            topic_info["Topic"] != -1
        ]
        .sort_values("Count", ascending = False)
        .reset_index(drop = True)
    )
    words = info["Representation"].map(
        lambda w: ", ".join(w[:n_words])
    )
    fig = go.Figure(go.Bar(
        x = info.index + 1,
        y = info.Count,
        customdata = list(zip(info.Topic, words)),
        hovertemplate = (
            "Topic %{customdata[0]}<br>"
            "%{customdata[1]}<br>"
            "Documents: %{y}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title = "Document counts per topic",
        xaxis_title = "Topic (rank by size)",
        yaxis_title = "Document count",
    )
    return fig
