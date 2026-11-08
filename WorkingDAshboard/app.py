# app.py
"""
NIRF Insights — Interactive Analytics Dashboard (2017 – 2025)

Entry point / router. Each section is its own page, reachable from the top
navigation bar, and only the selected page is rendered:

  Overview       views/overview.py   headline numbers, highlights, shortcuts
  Graph Builder  views/builder.py    Level 2: ad-hoc adaptive visualiser
  Annexure A     views/annexure.py   Level 1: the 40 prescribed questions, one at a time
  Ask AI         views/assistant.py  Level 3: Gemini natural-language assistant

Run with:  python -m streamlit run app.py
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

import ui

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NIRF Insights · 2017–2025",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()
st.logo(os.path.join(HERE, "assets", "logo.svg"), size="large")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page("views/builder.py", title="Graph Builder", icon=":material/insights:", url_path="builder"),
    st.Page("views/annexure.py", title="Annexure A", icon=":material/menu_book:", url_path="annexure"),
    st.Page("views/assistant.py", title="Ask AI", icon=":material/auto_awesome:", url_path="assistant"),
]
st.navigation(pages, position="top").run()
