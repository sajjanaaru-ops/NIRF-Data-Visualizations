# dynamic_charts.py
"""
Interactive Plotly chart engine for the NIRF Graph Builder.

Every function takes an already-filtered DataFrame and returns a Plotly figure
(the scatter also returns a stats dict). Functions never raise on empty or
sparse input: they return a figure with a friendly message instead.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_THEME = "plotly_white"
PALETTE = ["#0071e3", "#34c759", "#ff9f0a", "#af52de", "#ff375f", "#5ac8fa", "#a2845e", "#ffcc00",
           "#5856d6", "#30b0c7", "#8e8e93"]
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, INK_3, GRID = "#1d1d1f", "#6e6e73", "#ececf0"
PARAMS = ["TLR", "RPC", "GO", "OI", "Perception"]
SCORE_COLS = ["Score"] + PARAMS

METRIC_LABELS = {
    "Score": "Overall Score",
    "TLR": "Teaching, Learning & Resources (TLR)",
    "RPC": "Research & Professional Practice (RPC)",
    "GO": "Graduation Outcomes (GO)",
    "OI": "Outreach & Inclusivity (OI)",
    "Perception": "Perception (PR)",
    "Rank_Num": "Rank",
}
SHORT_LABELS = {"Score": "Score", "TLR": "TLR", "RPC": "RPC", "GO": "GO",
                "OI": "OI", "Perception": "Perception", "Rank_Num": "Rank"}


def label(metric):
    return METRIC_LABELS.get(metric, metric)


def empty_figure(message, height=420):
    """A blank figure with a centred message (used instead of raising)."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=15, color=INK_3, family=FONT),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(template=CHART_THEME, height=height,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def _base_layout(fig, title, height):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK, family=FONT, weight=600)),
        template=CHART_THEME,
        height=height,
        margin=dict(l=10, r=30, t=60, b=30),
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        legend=dict(font=dict(color=INK_3)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_3),
                     title_font=dict(color=INK_3))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_3),
                     title_font=dict(color=INK_3))
    return fig


def _unique_labels(frame):
    """Institution labels that stay unique when two institutes share a name."""
    names = frame["Name"].astype(str)
    dup = names.duplicated(keep=False)
    labels = names.where(~dup, names + " (" + frame["City"].astype(str) + ")")
    dup2 = labels.duplicated(keep=False)
    return labels.where(~dup2, labels + " #" + (frame.groupby(labels).cumcount() + 1).astype(str))


def _shorten(text, n=48):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


# ------------------------------------------------------------------------------
# 1. Top N bar chart
# ------------------------------------------------------------------------------
def create_top_n_bar_chart(filtered_df, metric="Score", top_n=10, color_mode="state", multi_year="mean"):
    """Horizontal bar chart of the top N institutions.

    multi_year: how to combine several years for one institution —
                "mean" (average across years) or "latest" (most recent year only).
    color_mode: "state" (one colour per state) or "value" (shade by value).
    """
    if filtered_df.empty or metric not in filtered_df.columns:
        return empty_figure("No data matches the current filters")

    data = filtered_df.dropna(subset=[metric])
    if data.empty:
        return empty_figure(f"No published {SHORT_LABELS.get(metric, metric)} values in this selection")

    n_years = data["Year"].nunique()
    if n_years > 1:
        if multi_year == "latest":
            data = data.sort_values("Year").groupby(["Name", "State"], as_index=False).tail(1)
            agg_note = "most recent year"
        else:
            data = (data.groupby(["Name", "State"], as_index=False)
                        .agg(**{metric: (metric, "mean"), "City": ("City", "first"),
                                "Years": ("Year", "nunique"), "Year": ("Year", "max"),
                                "Rank": ("Rank", "last")}))
            agg_note = "average across years"
    else:
        agg_note = None

    top = data.sort_values(metric, ascending=False).head(top_n).copy()
    top["Institution"] = _unique_labels(top).map(_shorten)
    order = top["Institution"].tolist()[::-1]  # best at top

    hover = {c: True for c in ["State", "City", "Year", "Rank"] if c in top.columns}
    if "Years" in top.columns:
        hover["Years"] = True

    common = dict(x=metric, y="Institution", orientation="h", text=metric,
                  hover_data=hover, template=CHART_THEME)
    if color_mode == "state":
        fig = px.bar(top, color="State", color_discrete_sequence=PALETTE, **common)
        fig.update_layout(legend=dict(title="State", orientation="v", y=1, x=1.02))
    else:
        fig = px.bar(top, color=metric, color_continuous_scale=["#b9d8fb", "#0071e3"], range_color=[top[metric].min() * 0.9, top[metric].max()], **common)
        fig.update_coloraxes(showscale=False)

    fig.update_yaxes(categoryorder="array", categoryarray=order, title="")
    fig.update_xaxes(title=label(metric), range=[0, top[metric].max() * 1.12])
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False,
                      marker_line_width=0)

    title = f"Top {len(top)} institutions by {label(metric)}"
    if agg_note:
        title += f" <span style='font-size:12px;color:#64748b'>({agg_note})</span>"
    return _base_layout(fig, title, max(420, 30 * len(top) + 120))


# ------------------------------------------------------------------------------
# 2. Time series
# ------------------------------------------------------------------------------
def create_time_series_chart(df, selected_institutions, metric="Rank_Num"):
    """Year-by-year trajectory of selected institutions."""
    if not selected_institutions:
        return empty_figure("Pick at least one institution to see its trend")

    trend_df = df[df["Name"].isin(selected_institutions)].sort_values("Year")
    trend_df = trend_df.dropna(subset=[metric])
    if trend_df.empty:
        return empty_figure(f"The selected institutions have no published {SHORT_LABELS.get(metric, metric)} "
                            "in these years (rank-band entries have no exact value)")

    trend_df = trend_df.assign(Institution=trend_df["Name"].map(lambda s: _shorten(s, 40)))
    fig = px.line(trend_df, x="Year", y=metric, color="Institution", markers=True,
                  hover_data={"Rank": True, "Score": ":.2f", "State": True, "Institution": False},
                  hover_name="Name", color_discrete_sequence=PALETTE, template=CHART_THEME)
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8))

    if metric == "Rank_Num":
        fig.update_yaxes(autorange="reversed", title="Rank (1 = best)")
    else:
        fig.update_yaxes(title=label(metric))
    fig.update_xaxes(dtick=1, title="")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, title=""),
                      hovermode="closest")
    years = trend_df["Year"]
    return _base_layout(fig, f"{label(metric)} over time ({years.min()}–{years.max()})", 480)


# ------------------------------------------------------------------------------
# 3. Scatter
# ------------------------------------------------------------------------------
def create_bivariate_scatter(filtered_df, x_var="RPC", y_var="Score", show_trendline=True, color_var="State"):
    """Scatter plot with a single overall OLS trendline and correlation stats."""
    scatter_df = filtered_df.dropna(subset=[x_var, y_var]).copy()
    if len(scatter_df) < 3:
        return empty_figure("Not enough institutions with published scores for a scatter plot"), {}

    if x_var == y_var:
        return empty_figure("Choose two different metrics for the X and Y axes"), {}

    stats_summary = {"sample_size": len(scatter_df), "pearson_r": None, "spearman_rho": None}
    if scatter_df[x_var].std() > 0 and scatter_df[y_var].std() > 0:
        stats_summary["pearson_r"] = scatter_df[x_var].corr(scatter_df[y_var])
        stats_summary["spearman_rho"] = scatter_df[x_var].corr(scatter_df[y_var], method="spearman")

    color = None
    if color_var == "State":
        top_states = scatter_df["State"].value_counts().head(8).index
        scatter_df["Colour"] = scatter_df["State"].where(scatter_df["State"].isin(top_states), "Other states")
        color = "Colour"
    elif color_var == "Year":
        scatter_df["Colour"] = scatter_df["Year"].astype(str)
        color = "Colour"

    fig = px.scatter(
        scatter_df, x=x_var, y=y_var, color=color, hover_name="Name",
        hover_data={"State": True, "Year": True, "Rank": True, "Colour": False} if color else
                   {"State": True, "Year": True, "Rank": True},
        trendline="ols" if show_trendline else None, trendline_scope="overall",
        trendline_color_override="#ff375f",
        color_discrete_sequence=PALETTE, template=CHART_THEME,
        category_orders={"Colour": sorted(scatter_df["Colour"].unique(), key=lambda s: (s == "Other states", s))}
        if color else None,
    )
    fig.update_traces(selector=dict(mode="markers"),
                      marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color="white")))
    fig.update_xaxes(title=label(x_var))
    fig.update_yaxes(title=label(y_var))
    fig.update_layout(legend=dict(title=color_var if color else ""))

    r = stats_summary["pearson_r"]
    title = f"{SHORT_LABELS.get(y_var)} vs {SHORT_LABELS.get(x_var)}"
    if r is not None:
        title += f" <span style='font-size:12px;color:#64748b'>(r = {r:+.2f})</span>"
    return _base_layout(fig, title, 500), stats_summary


# ------------------------------------------------------------------------------
# 4. Distribution
# ------------------------------------------------------------------------------
def create_distribution_chart(filtered_df, metric="Score", group_by="None", chart_kind="box", max_groups=12):
    """Box, violin or histogram of a metric, optionally grouped by Year or State."""
    plot_df = filtered_df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return empty_figure(f"No published {SHORT_LABELS.get(metric, metric)} values in this selection")

    group = None
    order = None
    if group_by == "Year":
        plot_df["Group"] = plot_df["Year"].astype(str)
        order = sorted(plot_df["Group"].unique())
        group = "Group"
    elif group_by == "State":
        top_states = plot_df["State"].value_counts().head(max_groups).index
        plot_df = plot_df[plot_df["State"].isin(top_states)]
        plot_df["Group"] = plot_df["State"]
        order = plot_df.groupby("Group")[metric].median().sort_values(ascending=False).index.tolist()
        group = "Group"

    cat_orders = {"Group": order} if group else None
    if chart_kind in ("box", "violin"):
        fn = px.box if chart_kind == "box" else px.violin
        kwargs = dict(points="outliers") if chart_kind == "box" else dict(box=True, points=False)
        fig = fn(plot_df, x=group, y=metric, color=group, hover_name="Name", category_orders=cat_orders,
                 color_discrete_sequence=PALETTE, template=CHART_THEME, **kwargs)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=group_by if group else "")
        fig.update_yaxes(title=label(metric))
    else:
        fig = px.histogram(plot_df, x=metric, color=group, nbins=30, opacity=0.75, barmode="overlay",
                           category_orders=cat_orders, color_discrete_sequence=PALETTE, template=CHART_THEME)
        fig.update_xaxes(title=label(metric))
        fig.update_yaxes(title="Number of institutions")
        fig.update_layout(legend=dict(title=group_by if group else ""))

    title = f"Distribution of {label(metric)}"
    if group:
        title += f" by {group_by}"
        if group_by == "State" and filtered_df["State"].nunique() > max_groups:
            title += f" <span style='font-size:12px;color:#64748b'>(top {max_groups} states by count)</span>"
    return _base_layout(fig, title, 480)


# ------------------------------------------------------------------------------
# 5. Radar
# ------------------------------------------------------------------------------
def create_radar_comparison(df_year, selected_institutions, year, benchmark_df=None):
    """Radar chart comparing up to 5 institutions on the five NIRF parameters."""
    radar_df = df_year[df_year["Name"].isin(selected_institutions)].dropna(subset=PARAMS, how="all")
    if radar_df.empty:
        return empty_figure(f"Pick institutions with published scores in {year}")

    theta = PARAMS + [PARAMS[0]]
    fig = go.Figure()
    if benchmark_df is not None and not benchmark_df.dropna(subset=PARAMS, how="all").empty:
        avg = benchmark_df[PARAMS].mean().fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(r=avg + [avg[0]], theta=theta, name="Category average",
                                      line=dict(color="#94a3b8", dash="dash", width=2), fill=None))

    labels = _unique_labels(radar_df).tolist()
    for i, (_, row) in enumerate(radar_df.iterrows()):
        values = [row[p] if pd.notnull(row[p]) else 0 for p in PARAMS]
        colour = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=theta, fill="toself", opacity=0.55,
                                      name=_shorten(labels[i], 40), line=dict(color=colour, width=2),
                                      hovertemplate="%{theta}: %{r:.1f}<extra>" + _shorten(labels[i], 40) + "</extra>"))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
                      legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5))
    return _base_layout(fig, f"Parameter profile, {year}", 520)


# ------------------------------------------------------------------------------
# 6. Correlation heatmap
# ------------------------------------------------------------------------------
def create_correlation_heatmap(filtered_df, method="pearson"):
    """Pairwise correlation matrix of the score and five parameters."""
    cols = [c for c in SCORE_COLS if c in filtered_df.columns and filtered_df[c].notna().sum() >= 3]
    if len(cols) < 2:
        return empty_figure("Not enough published scores to compute correlations")

    corr = filtered_df[cols].corr(method=method, min_periods=3)
    corr.index = [SHORT_LABELS[c] for c in corr.index]
    corr.columns = [SHORT_LABELS[c] for c in corr.columns]

    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, template=CHART_THEME)
    fig.update_coloraxes(colorbar=dict(title="r"))
    n = int(filtered_df[cols].dropna(how="all").shape[0])
    return _base_layout(fig, f"How the NIRF parameters move together "
                             f"<span style='font-size:12px;color:#64748b'>({method.title()}, n = {n})</span>", 480)


# ------------------------------------------------------------------------------
# 7. State aggregation
# ------------------------------------------------------------------------------
def create_state_aggregation_chart(filtered_df, top_n=10, agg_mode="count", min_count=3):
    """Number of ranked institutions, or average score, per state."""
    if filtered_df.empty or "State" not in filtered_df.columns:
        return empty_figure("No data matches the current filters")

    if agg_mode == "count":
        state_data = (filtered_df.groupby("State")["Name"].nunique()
                      .sort_values(ascending=False).head(top_n).reset_index())
        state_data.columns = ["State", "Institutions"]
        value_col, fmt = "Institutions", "%{text}"
        title = f"Top {len(state_data)} states by number of ranked institutions"
    else:
        scored = filtered_df.dropna(subset=["Score"])
        if scored.empty:
            return empty_figure("No published scores in this selection")
        grp = scored.groupby("State").agg(avg=("Score", "mean"), n=("Name", "nunique"))
        grp = grp[grp["n"] >= min_count]
        if grp.empty:
            return empty_figure(f"No state has at least {min_count} scored institutions — lower the minimum")
        state_data = grp.sort_values("avg", ascending=False).head(top_n).reset_index()
        state_data.columns = ["State", "Average score", "Institutions"]
        value_col, fmt = "Average score", "%{text:.1f}"
        title = f"Top {len(state_data)} states by average score (≥ {min_count} institutions)"

    order = state_data["State"].tolist()[::-1]
    fig = px.bar(state_data, x=value_col, y="State", orientation="h", text=value_col, color=value_col,
                 color_continuous_scale=["#b9d8fb", "#0071e3"], hover_data=list(state_data.columns), template=CHART_THEME)
    fig.update_traces(texttemplate=fmt, textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(categoryorder="array", categoryarray=order, title="")
    fig.update_xaxes(title=value_col, range=[0, state_data[value_col].max() * 1.12])
    return _base_layout(fig, title, max(420, 30 * len(state_data) + 120))
