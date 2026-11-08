# graph_builder.py
"""
Custom Graph Builder panel (Level 2: ad-hoc adaptive visualisation).

The left panel walks the user through three steps:
  1. What to see   - pick a chart by goal, with a one-line explanation
  2. Which data    - category, year / year range, states
  3. Customise     - only the options relevant to the chosen chart
The right panel shows a summary of the active filters, KPIs, the chart,
data-coverage notes and the underlying records with CSV export.
"""

import pandas as pd
import streamlit as st

import dynamic_charts as dc
import ui

P = "gb_"  # session-state key prefix for every builder widget

CHARTS = {
    "top": ("🏆 Top institutions", "Rank the leading institutions by overall score or any single parameter."),
    "trend": ("📈 Trends over time", "Follow how chosen institutions' rank or scores changed year to year."),
    "scatter": ("🔗 Compare two metrics", "Check whether two parameters go together, with a trendline and correlation."),
    "dist": ("📦 Score spread", "See how scores are distributed, overall or split by year or state."),
    "radar": ("🕸️ Institution profiles", "Compare up to 5 institutions on all five NIRF parameters for one year."),
    "corr": ("🧮 Parameter correlations", "Find out which NIRF parameters tend to move together."),
    "state": ("🗺️ States", "See which states have the most, or the best-scoring, ranked institutions."),
}
NEEDS_SCORES = {"top", "scatter", "dist", "radar", "corr"}
METRICS = ["Score", "TLR", "RPC", "GO", "OI", "Perception"]


# ------------------------------------------------------------------------------
# Session-state helpers
# ------------------------------------------------------------------------------
def _reset():
    for k in [k for k in st.session_state.keys() if str(k).startswith(P)]:
        del st.session_state[k]


def _ensure(key, valid, default):
    """Make sure a stored widget value is still valid for the current options."""
    if st.session_state.get(key) not in valid:
        st.session_state[key] = default


def _ensure_list(key, valid, default):
    """Keep only still-valid picks; fall back to default if nothing survives."""
    valid = set(valid)
    current = st.session_state.get(key)
    if current is None:
        st.session_state[key] = default
        return
    kept = [v for v in current if v in valid]
    st.session_state[key] = kept if (kept or not current) else default


def _pretty_category(c):
    return "State Public University" if c == "STATEPUBLICUNIVERSITY" else c


def _step(n, text):
    st.markdown(f"<div class='gb-step'><span class='gb-num'>{n}</span>{text}</div>", unsafe_allow_html=True)


def _top_ranked(frame, n):
    """Names of the n best-ranked institutions in the latest year of `frame`."""
    ranked = frame.dropna(subset=["Rank_Num"])
    if ranked.empty:
        return sorted(frame["Name"].unique())[:n]
    latest = ranked[ranked["Year"] == ranked["Year"].max()]
    return latest.sort_values("Rank_Num")["Name"].drop_duplicates().head(n).tolist()


CSS = """
<style>
.gb-step {font-weight: 650; font-size: 0.95rem; color: #1d1d1f; margin: 1.3rem 0 0.5rem 0;
          display: flex; align-items: center; letter-spacing: -0.01em;}
.gb-num {display:inline-flex; align-items:center; justify-content:center; width:1.5rem; height:1.5rem;
         border-radius:50%; background:#0071e3; color:#fff; font-size:0.78rem; font-weight:700; margin-right:0.6rem;}
.gb-head {font-size: 1.15rem; font-weight: 700; letter-spacing: -0.015em; color: #1d1d1f;}
.gb-title {font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; color: #1d1d1f; margin-bottom: 0.3rem;}
.gb-desc {font-size: 0.86rem; color: #6e6e73; background: #f5f5f7; border-radius: 12px; padding: 0.6rem 0.8rem;
          margin-top: 0.3rem; line-height: 1.45;}
.st-key-gb_chart [data-testid="stButtonGroup"] button p { font-size: 0.85rem; }
</style>
"""


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
def render_graph_builder(df):
    st.markdown(CSS, unsafe_allow_html=True)
    cfg_col, out_col = st.columns([1.15, 2.6], gap="large")

    with cfg_col:
        with st.container(key="card_gb_cfg"):
            head_l, head_r = st.columns([1.6, 1], vertical_alignment="center")
            head_l.markdown("<div class='gb-head'>Chart settings</div>", unsafe_allow_html=True)
            head_r.button("Reset", on_click=_reset, width="stretch", icon=":material/restart_alt:",
                          help="Restore every option in this panel to its default")

            # ---- Step 1: chart --------------------------------------------------
            _step(1, "What do you want to see?")
            if st.session_state.get(P + "chart") not in CHARTS:  # pills can be de-selected
                st.session_state[P + "chart"] = st.session_state.get(P + "last_chart", "top")
            chart = st.pills("Chart", options=list(CHARTS), format_func=lambda k: CHARTS[k][0],
                             key=P + "chart", label_visibility="collapsed") or "top"
            st.session_state[P + "last_chart"] = chart
            st.markdown(f"<div class='gb-desc'>{CHARTS[chart][1]}</div>", unsafe_allow_html=True)

            # ---- Step 2: data ---------------------------------------------------
            _step(2, "Which data?")
            categories = sorted(df["Category"].dropna().unique())
            _ensure(P + "category", categories, "Overall" if "Overall" in categories else categories[0])
            category = st.selectbox("Ranking category", categories, key=P + "category",
                                    format_func=_pretty_category)
            cat_df = df[df["Category"] == category]
            years = sorted(int(y) for y in cat_df["Year"].unique())

            if chart == "trend":
                year_mode = "range"
            elif chart == "radar":
                year_mode = "single"
            elif len(years) > 1:
                _ensure(P + "year_mode", ["single", "range"], "single")
                year_mode = st.segmented_control(
                    "Years", ["single", "range"], key=P + "year_mode",
                    format_func={"single": "One year", "range": "Range of years"}.get,
                    help="Use a range to pool several years together") or "single"
            else:
                year_mode = "single"

            if len(years) == 1:
                year_from = year_to = years[0]
                st.caption(f"📅 {_pretty_category(category)} rankings were only published for **{years[0]}**.")
            elif year_mode == "single":
                _ensure(P + "year", years, years[-1])
                year_from = year_to = st.select_slider("Year", options=years, key=P + "year")
            else:
                rng = st.session_state.get(P + "years")
                if not (isinstance(rng, (tuple, list)) and len(rng) == 2 and all(y in years for y in rng)):
                    st.session_state[P + "years"] = (years[0], years[-1])
                year_from, year_to = st.select_slider("Year range", options=years, key=P + "years")

            year_df = cat_df[cat_df["Year"].between(year_from, year_to)]
            state_opts = sorted(year_df["State"].dropna().unique())
            _ensure_list(P + "states", state_opts, [])
            states = st.multiselect("States", state_opts, key=P + "states",
                                    placeholder="All states — type to pick specific ones")
            work_df = year_df[year_df["State"].isin(states)] if states else year_df
            scored_df = work_df.dropna(subset=["Score"])

            # ---- Step 3: chart-specific options -------------------------------
            _step(3, "Customise")
            opts = _chart_options(chart, work_df, scored_df, year_from, year_to)

    with out_col:
        with st.container(key="card_gb_out"):
            _render_output(chart, opts, category, cat_df, work_df, scored_df, year_from, year_to, states)


def _metric_select(label, key, default="Score", options=METRICS, help=None):
    _ensure(key, options, default)
    return st.selectbox(label, options, key=key, format_func=dc.label, help=help)


def _chart_options(chart, work_df, scored_df, year_from, year_to):
    o = {}
    multi_year = year_to > year_from

    if chart == "top":
        o["metric"] = _metric_select("Rank institutions by", P + "top_metric")
        _ensure(P + "top_n", list(range(5, 51, 5)), 10)
        o["top_n"] = st.slider("How many institutions", 5, 50, step=5, key=P + "top_n")
        _ensure(P + "top_color", ["state", "value"], "state")
        o["color_mode"] = st.segmented_control("Bar colour", ["state", "value"], key=P + "top_color",
                                               format_func={"state": "By state", "value": "By value"}.get) or "state"
        if multi_year:
            _ensure(P + "top_agg", ["mean", "latest"], "mean")
            o["multi_year"] = st.segmented_control(
                "Combine years using", ["mean", "latest"], key=P + "top_agg",
                format_func={"mean": "Average", "latest": "Most recent year"}.get,
                help="Each institution appears once: averaged over the range, or its latest year") or "mean"

    elif chart == "trend":
        _ensure(P + "trend_metric", ["Rank_Num"] + METRICS, "Rank_Num")
        o["metric"] = st.selectbox("Track", ["Rank_Num"] + METRICS, key=P + "trend_metric", format_func=dc.label)
        names = sorted(work_df["Name"].dropna().unique())
        _ensure_list(P + "trend_insts", names, _top_ranked(work_df, 5))
        o["institutions"] = st.multiselect(
            "Institutions", names, key=P + "trend_insts", max_selections=10,
            placeholder="Type a name, e.g. 'Madras'…",
            help="Defaults to the top 5 of the latest year. Up to 10 institutions.")

    elif chart == "scatter":
        c1, c2 = st.columns(2)
        with c1:
            o["x"] = _metric_select("X axis", P + "sc_x", "RPC")
        with c2:
            o["y"] = _metric_select("Y axis", P + "sc_y", "Score")
        if o["x"] == o["y"]:
            st.warning("Pick two different metrics.")
        colour_opts = ["State", "Year", "None"] if multi_year else ["State", "None"]
        _ensure(P + "sc_color", colour_opts, "State")
        o["color"] = st.segmented_control("Colour points by", colour_opts, key=P + "sc_color") or "None"
        _init_bool(P + "sc_trend", True)
        o["trend"] = st.toggle("Show trendline", key=P + "sc_trend")

    elif chart == "dist":
        o["metric"] = _metric_select("Metric", P + "dist_metric")
        group_opts = ["None", "Year", "State"] if multi_year else ["None", "State"]
        _ensure(P + "dist_group", group_opts, "Year" if multi_year else "None")
        o["group"] = st.segmented_control("Split by", group_opts, key=P + "dist_group") or "None"
        _ensure(P + "dist_kind", ["box", "violin", "histogram"], "box")
        o["kind"] = st.segmented_control("Style", ["box", "violin", "histogram"], key=P + "dist_kind",
                                         format_func=str.title) or "box"

    elif chart == "radar":
        names = sorted(scored_df["Name"].dropna().unique())
        _ensure_list(P + "radar_insts", names, _top_ranked(scored_df, 3))
        o["institutions"] = st.multiselect("Institutions (up to 5)", names, key=P + "radar_insts",
                                           max_selections=5, placeholder="Type a name…")
        _init_bool(P + "radar_bench", True)
        o["benchmark"] = st.toggle("Show category average", key=P + "radar_bench",
                                   help="Dashed outline = average of all scored institutions in this selection")

    elif chart == "corr":
        _ensure(P + "corr_method", ["pearson", "spearman"], "pearson")
        o["method"] = st.segmented_control(
            "Method", ["pearson", "spearman"], key=P + "corr_method", format_func=str.title,
            help="Pearson = linear relationship; Spearman = rank-based, robust to outliers") or "pearson"

    elif chart == "state":
        _ensure(P + "st_mode", ["count", "mean"], "count")
        o["mode"] = st.segmented_control("Measure", ["count", "mean"], key=P + "st_mode",
                                         format_func={"count": "No. of institutions", "mean": "Average score"}.get) or "count"
        _ensure(P + "st_n", list(range(5, 31, 5)), 10)
        o["top_n"] = st.slider("How many states", 5, 30, step=5, key=P + "st_n")
        if o["mode"] == "mean":
            _ensure(P + "st_min", list(range(1, 11)), 3)
            o["min_count"] = st.slider("Minimum institutions per state", 1, 10, key=P + "st_min",
                                       help="Hides states whose average rests on very few institutions")
    return o


def _init_bool(key, default):
    if not isinstance(st.session_state.get(key), bool):
        st.session_state[key] = default


def _render_output(chart, o, category, cat_df, work_df, scored_df, year_from, year_to, states):
    title, _ = CHARTS[chart]
    st.markdown(f"<div class='gb-title'>{title}</div>", unsafe_allow_html=True)

    years_txt = str(year_from) if year_from == year_to else f"{year_from}–{year_to}"
    states_txt = "All states" if not states else (", ".join(states) if len(states) <= 3 else f"{len(states)} states")
    chips = [_pretty_category(category), years_txt, states_txt]
    ui.chips(chips)

    # KPIs (custom cards wrap long names instead of truncating them)
    leader = _top_ranked(work_df, 1)
    leader = leader[0] if leader else "—"
    ui.stat_cards([
        {"label": "Institutions", "value": f"{work_df['Name'].nunique():,}"},
        {"label": "With published scores", "value": f"{len(scored_df):,}", "sub": "rank-band entries have none"},
        {"label": "Average score", "value": f"{scored_df['Score'].mean():.1f}" if not scored_df.empty else "—"},
        {"label": f"#1 in {work_df['Year'].max() if not work_df.empty else '—'}", "value": leader, "small": True},
    ], compact=True)
    st.write("")

    # Scores missing entirely for this slice → explain instead of an empty chart
    if chart in NEEDS_SCORES and scored_df.empty:
        st.info(f"**{_pretty_category(category)}** has no published parameter scores for this selection "
                "(only ranks or rank bands). Try **🗺️ States** or **📈 Trends over time** (rank), "
                "or pick another category / year.", icon="ℹ️")
        _data_expander(work_df, category, years_txt)
        return

    stats = None
    if chart == "top":
        fig = dc.create_top_n_bar_chart(scored_df, o["metric"], o["top_n"], o["color_mode"], o.get("multi_year", "mean"))
    elif chart == "trend":
        fig = dc.create_time_series_chart(work_df, o["institutions"], o["metric"])
    elif chart == "scatter":
        fig, stats = dc.create_bivariate_scatter(scored_df, o["x"], o["y"], o["trend"], o["color"])
    elif chart == "dist":
        fig = dc.create_distribution_chart(scored_df, o["metric"], o["group"], o["kind"])
    elif chart == "radar":
        fig = dc.create_radar_comparison(scored_df, o["institutions"], year_from,
                                         benchmark_df=scored_df if o["benchmark"] else None)
    elif chart == "corr":
        fig = dc.create_correlation_heatmap(scored_df, o["method"])
    else:
        fig = dc.create_state_aggregation_chart(work_df, o["top_n"], o["mode"], o.get("min_count", 3))

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False,
                    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": f"nirf_{chart}"}})

    # Context notes under the chart
    if stats and stats.get("pearson_r") is not None:
        r = stats["pearson_r"]
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        st.caption(f"📐 Pearson r = **{r:+.3f}** ({strength} {direction}), Spearman ρ = **{stats['spearman_rho']:+.3f}**, "
                   f"n = {stats['sample_size']} records.")
    unscored = len(work_df) - len(scored_df)
    if chart in NEEDS_SCORES and unscored > 0:
        st.caption(f"ℹ️ {unscored:,} rank-band records (101+) have no published scores and are not plotted.")
    if chart == "trend" and o["metric"] == "Rank_Num":
        st.caption("ℹ️ Gaps mean the institution was outside the exact-rank list (rank band) or unranked that year. "
                   "Some institutions changed their official name between years and appear as separate entries.")

    _data_expander(work_df, category, years_txt)


def _data_expander(work_df, category, years_txt):
    with st.expander(f"📋 View the {len(work_df):,} underlying records & download CSV"):
        cols = [c for c in ["Year", "Category", "Rank", "Name", "City", "State", "Score",
                            "TLR", "RPC", "GO", "OI", "Perception"] if c in work_df.columns]
        table = work_df[cols].sort_values(["Year", "Score"], ascending=[False, False]).reset_index(drop=True)
        st.dataframe(table, width="stretch", height=300, hide_index=True)
        st.download_button("📥 Download CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"nirf_{category}_{years_txt}.csv".replace("–", "-"), mime="text/csv")
