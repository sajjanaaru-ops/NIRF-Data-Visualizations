# views/assistant.py — Level 3: Gemini assistant
import streamlit as st

import chat_ui
import data
import ui

df, *_ = data.get_data()

# A narrower reading column suits a conversation better than the full width.
st.markdown("<style>.block-container{max-width:920px;}</style>", unsafe_allow_html=True)

ui.page_header("Level 3 · AI Assistant", "Ask NIRF AI",
               "Ask a question in plain English. Gemini answers by querying the dataset directly, "
               "never from memory, and draws a chart when it helps.")
chat_ui.render_chat(df)
