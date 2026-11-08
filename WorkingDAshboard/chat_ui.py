# chat_ui.py
"""Streamlit UI for the Level 3 Gemini-powered NIRF assistant (rendered on its own page)."""

import json

import streamlit as st

import chatbot_engine as ce

SUGGESTIONS = [
    ("🏆", "Who were the top 5 in Overall in 2025?"),
    ("📈", "How has IIT Bombay's rank changed since 2017?"),
    ("🗺️", "Which states have the most top-100 Overall institutions in 2025?"),
    ("🕸️", "Compare IIT Madras, IISc and IIT Delhi on all parameters in 2025"),
    ("🔗", "How strongly does research (RPC) correlate with score in Engineering?"),
    ("🩺", "Best-ranked medical colleges in Tamil Nadu in 2024"),
]

SETUP_HELP = """
**Turn on the assistant in 30 seconds**

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Create a file named `.env` in the project folder (next to `app.py`):

```
GEMINI_API_KEY=your-key-here
```

3. Send a message. No restart needed.

Optional: `GEMINI_MODEL=...` sets the main model (default `{model}`), and
`GEMINI_FALLBACK_MODELS=model-a,model-b` sets the backups tried when it is busy.
"""

CSS = """
<style>
.st-key-chat_empty { text-align: center; padding: 1.2rem 0 0.4rem 0; }
.chat-hello-icon { width: 64px; height: 64px; border-radius: 18px; margin: 0 auto 1rem auto; font-size: 1.9rem;
                   display: flex; align-items: center; justify-content: center;
                   background: linear-gradient(135deg, #0a84ff, #5e5ce6 60%, #bf5af2); color: white; }
.chat-hello-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; color: #1d1d1f; }
.chat-hello-sub { color: #6e6e73; margin: 0.35rem 0 1.4rem 0; }
[class*="st-key-sugg_"] button { justify-content: flex-start; text-align: left; border-radius: 16px !important;
    background: #fff; border: 1px solid #e3e3e8; padding: 0.85rem 1rem; min-height: 3.9rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03); transition: all .15s ease; }
[class*="st-key-sugg_"] button:hover { border-color: #0071e3; box-shadow: 0 4px 14px rgba(0,113,227,0.12); }
.chat-hello-title, .chat-hello-sub { text-align: center; }
[class*="st-key-sugg_"] button > div { justify-content: flex-start; width: 100%; }
[class*="st-key-sugg_"] button p { font-size: 0.92rem; line-height: 1.4; white-space: normal; }
</style>
"""


def _queue(prompt):
    st.session_state["chat_pending"] = prompt


def _clear():
    st.session_state["chat"] = []


def _render_content(i, msg):
    if msg.get("error"):
        st.error(msg["content"], icon="⚠️")
        return
    st.markdown(msg["content"])
    for j, fig in enumerate(msg.get("charts", [])):
        st.plotly_chart(fig, width="stretch", key=f"chat_fig_{i}_{j}", config={"displaylogo": False})
    if msg.get("fallbacks"):
        busy = ", ".join(f"{m} ({r})" for m, r in msg["fallbacks"])
        st.caption(f"↪︎ Answered by backup model **{msg['model']}** · skipped {busy}")
    calls = msg.get("tool_calls") or []
    if calls:
        with st.expander(f"How I got this · {len(calls)} data lookup{'s' if len(calls) != 1 else ''}",
                         icon=":material/manage_search:"):
            for c in calls:
                args = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in c["args"].items())
                st.markdown(f"- `{c['tool']}({args})` → {c['result']}")


def _render_message(i, msg):
    with st.chat_message(msg["role"]):
        _render_content(i, msg)


def _status(key, models):
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        if key:
            backups = f" · +{len(models) - 1} backup{'s' if len(models) > 2 else ''}" if len(models) > 1 else ""
            st.markdown(f"<span class='status' title='Tried in order: {' → '.join(models)}'>"
                        f"<span class='dot' style='background:#34c759'></span>"
                        f"Connected · {models[0]}{backups}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status'><span class='dot' style='background:#ff9f0a'></span>"
                        "Needs an API key</span>", unsafe_allow_html=True)
    with right:
        if st.session_state["chat"]:
            st.button("New chat", on_click=_clear, icon=":material/edit_square:", width="stretch")


def render_chat(df):
    st.markdown(CSS, unsafe_allow_html=True)
    key, models = ce.get_config()
    st.session_state.setdefault("chat", [])

    # Read the input first (st.chat_input at page level is pinned to the bottom),
    # then lay out the conversation so a new question appears in this same run.
    prompt = None
    if key:
        typed = st.chat_input("Ask about rankings, trends, states or parameters…")
        prompt = typed or st.session_state.pop("chat_pending", None)
    history = [m for m in st.session_state["chat"] if not m.get("error")]
    if prompt:
        st.session_state["chat"].append({"role": "user", "content": prompt})

    _status(key, models)

    if not key:
        with st.container(key="card_chat_setup"):
            st.markdown(SETUP_HELP.format(model=ce.DEFAULT_MODEL))
        return

    if not st.session_state["chat"]:
        with st.container(key="chat_empty"):
            st.markdown("<div class='chat-hello-icon'>✦</div>"
                        "<div class='chat-hello-title'>What would you like to know?</div>"
                        "<div class='chat-hello-sub'>Every answer is looked up in the real NIRF data. "
                        "Try one of these to start.</div>", unsafe_allow_html=True)
        cols = st.columns(2, gap="small")
        for n, (icon, text) in enumerate(SUGGESTIONS):
            cols[n % 2].button(f"{icon}  {text}", key=f"sugg_{n}", on_click=_queue, args=(text,), width="stretch")
        return

    st.write("")
    for i, msg in enumerate(st.session_state["chat"]):
        _render_message(i, msg)

    if prompt:
        with st.chat_message("assistant"):
            with st.spinner("Looking that up in the NIRF data…"):
                try:
                    result = ce.ask(prompt, history, df)
                    reply = {"role": "assistant", "content": result["text"],
                             "charts": result["charts"], "tool_calls": result["tool_calls"],
                             "model": result.get("model"), "fallbacks": result.get("fallbacks", [])}
                except ce.AssistantError as e:
                    reply = {"role": "assistant", "content": str(e), "error": True}
            st.session_state["chat"].append(reply)
            _render_content(len(st.session_state["chat"]) - 1, reply)
