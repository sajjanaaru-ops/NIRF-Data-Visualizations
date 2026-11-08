# ui.py
"""Shared design system: global CSS and small HTML components.

Look & feel: light canvas (#f5f5f7), white rounded cards with soft shadows,
one accent colour (#0071e3), generous whitespace and a clear type scale.
Any container created with a key starting with "card" is styled as a card.
"""

import html

import streamlit as st

ACCENT = "#0071e3"

CSS = """
<style>
:root {
  --ink: #1d1d1f; --ink-2: #424245; --ink-3: #6e6e73; --line: #e3e3e8;
  --card: #ffffff; --canvas: #f5f5f7; --accent: #0071e3; --accent-soft: rgba(0,113,227,0.08);
}
html, body { -webkit-font-smoothing: antialiased; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }
h1, h2, h3, h4 { letter-spacing: -0.022em; color: var(--ink); }
p, li { line-height: 1.6; }

/* ---------- Cards: any keyed container named card* ---------- */
[class*="st-key-card"] {
  background: var(--card);
  border-radius: 22px;
  padding: 1.4rem 1.5rem 1.5rem 1.5rem;
  border: 1px solid rgba(0,0,0,0.05);
  box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 10px 30px rgba(0,0,0,0.05);
}
[class*="st-key-card"] [data-testid="stExpander"] details { border-radius: 14px; }

/* ---------- Page header ---------- */
.ph { margin: 1.6rem 0 1.8rem 0; }
.ph-eyebrow { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
              color: var(--accent); margin-bottom: 0.35rem; }
.ph-title { font-size: clamp(1.9rem, 3.2vw, 2.6rem); font-weight: 750; letter-spacing: -0.03em;
            line-height: 1.1; color: var(--ink); margin: 0; }
.ph-sub { font-size: 1.08rem; color: var(--ink-3); margin-top: 0.6rem; max-width: 760px; line-height: 1.55; }

/* ---------- Hero ---------- */
.hero { padding: 2.2rem 0 1.4rem 0; text-align: center; }
.hero-eyebrow { display: inline-block; font-size: 0.8rem; font-weight: 600; color: var(--accent);
                background: var(--accent-soft); padding: 0.32rem 0.85rem; border-radius: 999px; margin-bottom: 1.1rem; }
.hero-title { font-size: clamp(2.3rem, 5.2vw, 4rem); font-weight: 800; letter-spacing: -0.04em;
              line-height: 1.04; color: var(--ink); margin: 0 auto; max-width: 900px; }
.hero-title .grad { background: linear-gradient(90deg, #0a84ff 0%, #5e5ce6 55%, #bf5af2 100%);
                    -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { font-size: 1.2rem; color: var(--ink-3); max-width: 680px; margin: 1.1rem auto 0 auto; line-height: 1.55; }

/* ---------- Stat cards ---------- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 0.4rem 0 0.4rem 0; }
.stat { background: var(--card); border-radius: 18px; padding: 1.05rem 1.2rem;
        border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 1px 2px rgba(0,0,0,0.03); min-width: 0; }
.stats.compact { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.stats.compact .stat { background: var(--canvas); box-shadow: none; border: none; padding: 0.8rem 1rem; border-radius: 14px; }
.stat-label { font-size: 0.8rem; font-weight: 500; color: var(--ink-3); margin-bottom: 0.25rem; }
.stat-value { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; color: var(--ink);
              line-height: 1.2; overflow-wrap: anywhere; }
.stats.compact .stat-value { font-size: 1.3rem; }
.stat-value.small { font-size: 1.05rem; line-height: 1.3; }
.stats.compact .stat-value.small { font-size: 0.95rem; }
.stat-sub { font-size: 0.78rem; color: var(--ink-3); margin-top: 0.25rem; line-height: 1.35; }

/* ---------- Feature tiles ---------- */
.tile-icon { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
             font-size: 1.35rem; margin-bottom: 0.9rem; }
.tile-title { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.015em; color: var(--ink); margin-bottom: 0.3rem; }
.tile-text { font-size: 0.95rem; color: var(--ink-3); line-height: 1.5; min-height: 4.4em; }

/* ---------- Section title inside a page ---------- */
.sec { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; color: var(--ink); margin: 2.2rem 0 0.9rem 0; }
.card-title { font-size: 1.05rem; font-weight: 650; color: var(--ink); margin-bottom: 0.15rem; letter-spacing: -0.01em; }
.card-sub { font-size: 0.85rem; color: var(--ink-3); margin-bottom: 0.4rem; }

/* ---------- Chips & pills ---------- */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.2rem 0 0.9rem 0; }
.chip { padding: 0.22rem 0.7rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        background: var(--accent-soft); color: #0058b0; }
.status { display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.85rem; font-weight: 500;
          color: var(--ink-2); background: var(--card); border: 1px solid var(--line);
          padding: 0.3rem 0.8rem; border-radius: 999px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

/* ---------- Insight callout ---------- */
.insight { background: linear-gradient(180deg, rgba(0,113,227,0.06), rgba(0,113,227,0.03));
           border-radius: 16px; padding: 1rem 1.2rem; margin-top: 0.9rem; border: 1px solid rgba(0,113,227,0.12); }
.insight-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
                 color: var(--accent); margin-bottom: 0.3rem; }
.insight-text { font-size: 1rem; color: var(--ink); line-height: 1.55; }

/* ---------- Widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a { font-weight: 500; }
[data-testid="stPageLink"] a { border-radius: 999px; }
[data-testid="stRadio"] label p { line-height: 1.4; }
div[data-baseweb="select"] > div, .stTextInput input { border-radius: 12px !important; }

/* Chat */
[data-testid="stChatMessage"] { background: transparent; padding: 0.6rem 0.2rem; }
[data-testid="stChatMessageContent"] p { line-height: 1.6; }
[data-testid="stBottomBlockContainer"] { background: linear-gradient(180deg, rgba(245,245,247,0) 0%, #f5f5f7 30%); }
[data-testid="stChatInput"] { border-radius: 22px !important; box-shadow: 0 6px 24px rgba(0,0,0,0.08); background: #fff; }

/* Call-to-action links styled as Apple-style pill buttons */
.st-key-cta_primary a, .st-key-cta_secondary a { justify-content: center; padding: 0.7rem 1.2rem !important;
    border-radius: 999px !important; font-weight: 600; transition: all .15s ease; }
.st-key-cta_primary a { background: var(--accent); }
.st-key-cta_primary a p, .st-key-cta_primary a span { color: #fff !important; }
.st-key-cta_primary a:hover { background: #0077ed; box-shadow: 0 6px 18px rgba(0,113,227,0.28); }
.st-key-cta_secondary a { background: var(--card); border: 1px solid var(--line); }
.st-key-cta_secondary a p, .st-key-cta_secondary a span { color: var(--accent) !important; }
.st-key-cta_secondary a:hover { border-color: var(--accent); }
/* Links inside feature tiles */
[class*="st-key-card_tile"] [data-testid="stPageLink"] { margin-top: 0.6rem; }
[class*="st-key-card_tile"] [data-testid="stPageLink"] a { padding-left: 0; }
[class*="st-key-card_tile"] [data-testid="stPageLink"] a p,
[class*="st-key-card_tile"] [data-testid="stPageLink"] a span { color: var(--accent) !important; font-weight: 600; }

/* Never let button labels break mid-word */
.stButton > button p, .stDownloadButton > button p { white-space: nowrap; }

/* Footer */
.foot { text-align: center; color: var(--ink-3); font-size: 0.8rem; margin-top: 3rem; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow, title, subtitle=None):
    sub = f"<div class='ph-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"<div class='ph'><div class='ph-eyebrow'>{eyebrow}</div>"
                f"<div class='ph-title'>{title}</div>{sub}</div>", unsafe_allow_html=True)


def section(title):
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)


def card_title(title, subtitle=None):
    sub = f"<div class='card-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"<div class='card-title'>{title}</div>{sub}", unsafe_allow_html=True)


def stat_cards(items, compact=False):
    """items: list of dicts {label, value, sub?, small?}. Values wrap instead of truncating."""
    cells = []
    for it in items:
        small = " small" if it.get("small") else ""
        sub = f"<div class='stat-sub'>{html.escape(str(it['sub']))}</div>" if it.get("sub") else ""
        cells.append(f"<div class='stat'><div class='stat-label'>{html.escape(str(it['label']))}</div>"
                     f"<div class='stat-value{small}'>{html.escape(str(it['value']))}</div>{sub}</div>")
    st.markdown(f"<div class='stats{' compact' if compact else ''}'>{''.join(cells)}</div>", unsafe_allow_html=True)


def chips(labels):
    st.markdown("<div class='chips'>" + "".join(f"<span class='chip'>{html.escape(str(c))}</span>" for c in labels)
                + "</div>", unsafe_allow_html=True)


def insight(text, label="Finding"):
    st.markdown(f"<div class='insight'><div class='insight-label'>{label}</div>"
                f"<div class='insight-text'>{html.escape(text)}</div></div>", unsafe_allow_html=True)


def footer():
    st.markdown("<div class='foot'>NIRF Insights · National Institutional Ranking Framework data 2017 – 2025 · "
                "Built with Python & Streamlit</div>", unsafe_allow_html=True)
