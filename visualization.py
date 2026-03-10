"""
Module 4: Education Visualization Dashboard
Interactive charts for learning path visualization using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# ── Source platform icons/colors ──
SOURCE_COLORS = {
    "YouTube": "#FF0000",
    "Coursera": "#0056D2",
    "Udemy": "#A435F0",
    "Khan Academy": "#14BF96",
    "freeCodeCamp": "#0A0A23",
    "edX": "#02262B",
    "MIT OCW": "#A31F34",
    "Google": "#4285F4",
    "Stanford": "#8C1515",
    "W3Schools": "#04AA6D",
    "GeeksforGeeks": "#2F8D46",
    "Kaggle": "#20BEFF",
    "Hugging Face": "#FFD21E",
    "Real Python": "#3776AB",
}

LEVEL_COLORS = {
    "Beginner": "#4CAF50",
    "Intermediate": "#FF9800",
    "Advanced": "#F44336",
}


def create_source_distribution_chart(resources: pd.DataFrame) -> go.Figure:
    """Pie chart showing resource distribution by platform."""
    if resources.empty:
        return _empty_figure("No resources to visualize")

    source_counts = resources["source"].value_counts().reset_index()
    source_counts.columns = ["Source", "Count"]

    colors = [SOURCE_COLORS.get(s, "#888") for s in source_counts["Source"]]

    fig = px.pie(
        source_counts,
        values="Count",
        names="Source",
        title="📊 Resources by Platform",
        color_discrete_sequence=colors,
        hole=0.4,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCC"),
        height=400,
    )
    return fig


def create_level_distribution_chart(resources: pd.DataFrame) -> go.Figure:
    """Bar chart showing resource count per difficulty level."""
    if resources.empty:
        return _empty_figure("No resources to visualize")

    level_order = ["Beginner", "Intermediate", "Advanced"]
    level_counts = resources["level"].value_counts().reindex(level_order, fill_value=0).reset_index()
    level_counts.columns = ["Level", "Count"]

    colors = [LEVEL_COLORS.get(l, "#888") for l in level_counts["Level"]]

    fig = px.bar(
        level_counts,
        x="Level",
        y="Count",
        title="📈 Resources by Difficulty Level",
        color="Level",
        color_discrete_map=LEVEL_COLORS,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCC"),
        height=400,
        showlegend=False,
    )
    return fig


def create_learning_path_chart(resources: pd.DataFrame, topic: str) -> go.Figure:
    """Funnel/timeline chart showing learning path: Beginner → Intermediate → Advanced."""
    if resources.empty:
        return _empty_figure("No resources to visualize")

    level_order = ["Beginner", "Intermediate", "Advanced"]
    path_data = []

    for level in level_order:
        level_resources = resources[resources["level"] == level]
        count = len(level_resources)
        titles = level_resources["description"].head(3).tolist()
        summary = " | ".join(titles) if titles else "No resources"
        path_data.append({
            "Stage": f"{level} ({count} resources)",
            "Count": max(count, 1),
            "Resources": summary,
        })

    df_path = pd.DataFrame(path_data)

    fig = go.Figure(go.Funnel(
        y=df_path["Stage"],
        x=df_path["Count"],
        textinfo="value+text",
        text=df_path["Resources"],
        marker=dict(
            color=[LEVEL_COLORS["Beginner"], LEVEL_COLORS["Intermediate"], LEVEL_COLORS["Advanced"]],
        ),
        connector=dict(line=dict(color="#555", width=2)),
    ))

    fig.update_layout(
        title=f"🗺️ Learning Path: {topic}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCC"),
        height=450,
    )
    return fig


def create_topic_heatmap(all_resources: pd.DataFrame) -> go.Figure:
    """Heatmap of resource availability across topics and levels."""
    if all_resources.empty:
        return _empty_figure("No resources to visualize")

    level_order = ["Beginner", "Intermediate", "Advanced"]
    topics = sorted(all_resources["topic"].unique())

    matrix = []
    for topic in topics:
        row = []
        for level in level_order:
            count = len(all_resources[
                (all_resources["topic"] == topic) & (all_resources["level"] == level)
            ])
            row.append(count)
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=level_order,
        y=topics,
        colorscale="Viridis",
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 14},
    ))

    fig.update_layout(
        title="🔥 Resource Availability Heatmap",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCC"),
        height=450,
        xaxis_title="Difficulty Level",
        yaxis_title="Topic",
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with a message."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#888"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    return fig
