# views/overview.py — landing page
import streamlit as st

import data
import dynamic_charts as dc
import ui

df, df_overall, *_ = data.get_data()

latest = int(df_overall["Year"].max())
leader = df_overall[df_overall["Year"] == latest].sort_values("Rank_Num")["Name"].iloc[0]
top100 = df_overall[(df_overall["Year"] == latest) & (df_overall["Rank_Num"] <= 100)]
top_state = top100["State"].value_counts()

# ---- Hero --------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">NIRF · 2017 – {latest}</div>
  <div class="hero-title">India's higher-education rankings, <span class="grad">beautifully explored.</span></div>
  <div class="hero-sub">Nine years of National Institutional Ranking Framework data across
  {df['Category'].nunique()} categories. Build your own charts, walk through the 40 assignment
  questions, or simply ask.</div>
</div>
""", unsafe_allow_html=True)

_, c1, c2, _ = st.columns([2, 1.1, 1.1, 2])
with c1.container(key="cta_primary"):
    st.page_link("views/builder.py", label="Build a chart", icon=":material/insights:", width="stretch")
with c2.container(key="cta_secondary"):
    st.page_link("views/assistant.py", label="Ask the AI", icon=":material/auto_awesome:", width="stretch")
st.write("")

# ---- Headline numbers --------------------------------------------------------
ui.stat_cards([
    {"label": "Records", "value": f"{len(df):,}", "sub": "institution-year entries"},
    {"label": "Years covered", "value": f"{df['Year'].nunique()}", "sub": f"{int(df['Year'].min())} – {latest}"},
    {"label": "Ranking categories", "value": f"{df['Category'].nunique()}", "sub": "Overall, Engineering, Medical…"},
    {"label": f"#1 Overall in {latest}", "value": leader, "small": True, "sub": f"{top_state.index[0]} leads the top 100 "
                                                                                 f"with {top_state.iloc[0]} institutions"},
])

# ---- Three ways to explore ---------------------------------------------------
ui.section("Three ways to explore")
tiles = [
    ("📊", "#e8f2ff", "Graph Builder", "Pick a chart, a category and years. Compare institutions, parameters and "
                                       "states in seconds.", "views/builder.py", "Open builder"),
    ("📚", "#eefbf1", "Annexure A", "All 40 prescribed questions across 8 analytical themes, each with its chart "
                                    "and finding.", "views/annexure.py", "Browse questions"),
    ("✨", "#f6effe", "Ask AI", "Ask in plain English. Gemini looks up the real numbers and draws a chart for "
                               "you.", "views/assistant.py", "Start a conversation"),
]
for col, (icon, bg, title, text, page, cta) in zip(st.columns(3, gap="medium"), tiles):
    with col:
        with st.container(key=f"card_tile_{title.replace(' ', '_')}"):
            st.markdown(f"<div class='tile-icon' style='background:{bg}'>{icon}</div>"
                        f"<div class='tile-title'>{title}</div><div class='tile-text'>{text}</div>",
                        unsafe_allow_html=True)
            st.page_link(page, label=cta, icon=":material/arrow_forward:")

# ---- Highlights --------------------------------------------------------------
ui.section(f"{latest} at a glance")
left, right = st.columns(2, gap="medium")
with left:
    with st.container(key="card_hl_top"):
        ui.card_title(f"Top 10 Overall, {latest}", "Overall NIRF score out of 100")
        fig = dc.create_top_n_bar_chart(df_overall[df_overall["Year"] == latest].dropna(subset=["Score"]),
                                        "Score", 10, "value")
        fig.update_layout(title_text="", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with right:
    with st.container(key="card_hl_states"):
        ui.card_title("Where the top 100 are", f"Institutions in the {latest} Overall top 100, by state")
        fig = dc.create_state_aggregation_chart(top100, top_n=10, agg_mode="count")
        fig.update_layout(title_text="", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

ui.footer()
