# data.py
"""Single cached entry point for the dataset, shared by every page."""

import streamlit as st

import questions_visualizer as qv


@st.cache_data(show_spinner="Loading NIRF data…")
def get_data():
    df, df_overall, params, pivot_rank, rank_diff = qv.load_data()
    df = df.copy()
    # Some categories (e.g. Innovation) store states in UPPER CASE ("DELHI");
    # map them onto the canonical spelling so filters and state charts don't split them.
    spellings = df["State"].dropna().str.strip().unique()
    canonical = {s.lower(): s for s in spellings if s.isupper()}
    canonical.update({s.lower(): s for s in spellings if not s.isupper()})  # mixed-case spelling wins
    canonical = {k: (v.title() if v.isupper() else v) for k, v in canonical.items()}
    df["State"] = df["State"].str.strip().str.lower().map(canonical).fillna(df["State"])
    return df, df_overall, params, pivot_rank, rank_diff
