# views/annexure.py — Level 1: the 40 prescribed questions, one at a time
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import streamlit as st

import data
import questions_visualizer as qv
import ui

CATALOG = qv.QUESTIONS_CATALOG
ALL_IDS = list(CATALOG)
THEMES = list(dict.fromkeys(q["category_name"] for q in CATALOG.values()))
DEFAULT_INST = "Indian Institute of Technology Madras"


def theme_label(name):
    return re.sub(r"^\d+\.\s*", "", name)


def _apply_chart_style():
    """Calm, legible matplotlib defaults that match the rest of the app."""
    plt.rcParams.update({
        "font.family": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlepad": 12,
        "axes.labelsize": 10.5, "axes.labelcolor": "#424245", "text.color": "#1d1d1f",
        "xtick.color": "#6e6e73", "ytick.color": "#6e6e73",
        "axes.edgecolor": "#d2d2d7", "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def _to_png(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def question_png(qid):
    _apply_chart_style()
    return _to_png(CATALOG[qid]["plot_fn"](*data.get_data()))


@st.cache_data(show_spinner=False)
def question_finding(qid, institution=None):
    """Finding text computed from the same data the chart uses (Q1.1 follows the chosen institution)."""
    fn = CATALOG[qid]["finding_fn"]
    try:
        if qid == "1.1" and institution:
            return fn(*data.get_data(), name=institution)
        return fn(*data.get_data())
    except Exception as e:  # never break the page because a sentence could not be built
        return f"Finding unavailable ({e})."


@st.cache_data(show_spinner=False)
def search_text(qid):
    q = CATALOG[qid]
    return " ".join([q["title"], q.get("note") or "", question_finding(qid)]).lower()


@st.cache_data(show_spinner=False)
def institution_trend_png(name):
    """Q1.1 is interactive: rank progression of any chosen institution."""
    _apply_chart_style()
    df_overall = data.get_data()[1]
    inst = df_overall[df_overall["Name"] == name].sort_values("Year")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(inst["Year"], inst["Rank_Num"], marker="o", markersize=8, color="#0071e3", linewidth=2.5)
    for x, y in zip(inst["Year"], inst["Rank_Num"]):
        if y == y:  # not NaN
            ax.annotate(f"#{int(y)}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=9, color="#424245")
    ax.invert_yaxis()
    ax.set_title(f"{name}: Overall rank, 2017 – 2025")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_xticks(range(2017, 2026))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # ranks are whole numbers
    ranks = inst["Rank_Num"].dropna()
    if not ranks.empty:
        ax.set_ylim(ranks.max() + max(1, 0.15 * ranks.max()), max(0.5, ranks.min() - max(1, 0.15 * ranks.max())))
    return _to_png(fig)


def _go(target):
    st.session_state["ax_q"] = target


# ---- Page --------------------------------------------------------------------
df, df_overall, *_ = data.get_data()

st.markdown("""<style>
.st-key-card_ax_nav [data-testid="stRadio"] [role="radiogroup"] { gap: 0.15rem; }
.st-key-card_ax_nav [data-testid="stRadio"] label { padding: 0.55rem 0.6rem; border-radius: 12px; width: 100%;
    transition: background .12s ease; }
.st-key-card_ax_nav [data-testid="stRadio"] label:hover { background: #f5f5f7; }
.st-key-card_ax_nav [data-testid="stRadio"] label:has(input:checked) { background: rgba(0,113,227,0.08); }
.st-key-card_ax_nav [data-testid="stRadio"] label p { font-size: 0.9rem; }
</style>""", unsafe_allow_html=True)

ui.page_header("Level 1 · Annexure A", "The 40 prescribed questions",
               "Eight analytical themes, five questions each. Pick a question to see its chart and "
               "what the data says.")

nav_col, view_col = st.columns([1, 2.35], gap="large")

with nav_col:
    with st.container(key="card_ax_nav"):
        query = st.text_input("Search", key="ax_search", placeholder="Search questions…",
                              label_visibility="collapsed", icon=":material/search:")
        theme = st.selectbox("Theme", ["All themes"] + THEMES, key="ax_theme",
                             format_func=lambda t: t if t == "All themes" else theme_label(t))

        ids = [k for k in ALL_IDS if theme == "All themes" or CATALOG[k]["category_name"] == theme]
        if query.strip():
            ql = query.strip().lower()
            ids = [k for k in ids if ql in search_text(k) or ql == k]

        st.caption(f"{len(ids)} of {len(ALL_IDS)} questions")
        if ids:
            if st.session_state.get("ax_q") not in ids:
                st.session_state["ax_q"] = ids[0]
            with st.container(height=470, border=False):
                st.radio("Question", ids, key="ax_q", label_visibility="collapsed",
                         format_func=lambda k: f"**{k}** · {CATALOG[k]['title']}")

with view_col:
    with st.container(key="card_ax_view"):
        if not ids:
            st.markdown("<div style='text-align:center;padding:4rem 1rem;color:#6e6e73'>"
                        "<div style='font-size:2.4rem'>🔍</div><div style='font-size:1.1rem;font-weight:600;"
                        "color:#1d1d1f;margin-top:.5rem'>No matching questions</div>"
                        "<div>Try another word or choose “All themes”.</div></div>", unsafe_allow_html=True)
        else:
            qid = st.session_state["ax_q"]
            q = CATALOG[qid]
            pos = ids.index(qid)

            ui.chips([theme_label(q["category_name"]), f"Question {qid}", f"{ALL_IDS.index(qid) + 1} of {len(ALL_IDS)}"])
            st.markdown(f"<div style='font-size:1.45rem;font-weight:700;letter-spacing:-0.02em;line-height:1.3;"
                        f"color:#1d1d1f;margin-bottom:1rem'>{q['title']}</div>", unsafe_allow_html=True)

            if q.get("note"):
                ui.insight(q["note"], label="How we read this question")

            inst = None
            if qid == "1.1":
                names = sorted(df_overall["Name"].dropna().unique())
                if st.session_state.get("ax_inst") not in names:
                    st.session_state["ax_inst"] = DEFAULT_INST if DEFAULT_INST in names else names[0]
                inst = st.selectbox("Institution", names, key="ax_inst",
                                    help="Pick any institution to see its Overall rank history")
                png = institution_trend_png(inst)
            else:
                try:
                    with st.spinner("Drawing chart…"):
                        png = question_png(qid)
                except Exception as e:
                    png = None
                    st.error(f"This chart could not be drawn: {e}")
            if png:
                st.image(png, width="stretch")

            ui.insight(question_finding(qid, inst))

            st.write("")
            b1, _, b2 = st.columns([1, 2, 1])
            b1.button("← Previous", key="ax_prev", width="stretch", disabled=pos == 0,
                      on_click=_go, args=(ids[pos - 1] if pos > 0 else qid,))
            b2.button("Next →", key="ax_next", width="stretch", type="primary", disabled=pos == len(ids) - 1,
                      on_click=_go, args=(ids[pos + 1] if pos < len(ids) - 1 else qid,))

ui.footer()
