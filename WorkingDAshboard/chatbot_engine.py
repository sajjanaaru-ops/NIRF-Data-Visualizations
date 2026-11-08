# chatbot_engine.py
"""
NIRF AI Query Assistant (Level 3 - Natural Language Interface), powered by Gemini.

How it works
------------
Gemini never sees the raw CSV and never executes generated code. Instead it is
given a small set of *tools* (plain Python functions) that query the pandas
DataFrame and build Plotly charts. Gemini decides which tools to call, the
google-genai SDK runs them (automatic function calling), and Gemini writes the
final answer from the tool results. Every tool call is logged so the UI can
show how an answer was produced.

Configuration (.env next to this file)
--------------------------------------
GEMINI_API_KEY=...            required
GEMINI_MODEL=gemini-3.8-flash optional override
"""

import difflib
import functools
import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd

import dynamic_charts as dc

HERE = os.path.dirname(os.path.abspath(__file__))
# Tried in order. When a model is overloaded, rate-limited, timing out or
# unavailable, the next one answers instead.
DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
DEFAULT_MODEL = DEFAULT_MODELS[0]
REQUEST_TIMEOUT_MS = 90_000      # per model attempt
RETRY_SAME_MODEL_DELAY_S = 1.5   # one quick retry on a 5xx before moving on
METRICS = ["Score", "TLR", "RPC", "GO", "OI", "Perception"]
MAX_ROWS = 50
MAX_CHARTS = 3


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
def _split(value):
    return [m.strip() for m in (value or "").split(",") if m.strip()]


def get_config():
    """Return (api_key, [models in fallback order]).

    Re-reads .env on every call so key/model edits apply without a restart.
      GEMINI_MODEL            primary model (or a comma-separated list)
      GEMINI_FALLBACK_MODELS  comma-separated fallbacks; unset = built-in defaults,
                              set but empty = no fallbacks
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HERE, ".env"), override=True)
    except ImportError:
        pass
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if key.lower().startswith("your") or key == "":
        key = None

    primary = _split(os.getenv("GEMINI_MODEL")) or [DEFAULT_MODEL]
    fallback_env = os.getenv("GEMINI_FALLBACK_MODELS")
    fallbacks = DEFAULT_MODELS if fallback_env is None else _split(fallback_env)
    models = list(dict.fromkeys(primary + fallbacks))  # keep order, drop duplicates
    return key, models


# ------------------------------------------------------------------------------
# Institution name matching (handles IIT / IISc / NIT abbreviations, renamed
# institutes and city spelling variants such as Bangalore / Bengaluru)
# ------------------------------------------------------------------------------
ALIASES = [
    (r"\biiser\b", "indian institute of science education and research"),
    (r"\biisc\b", "indian institute of science"),
    (r"\biiit\b", "indian institute of information technology"),
    (r"\biit\b", "indian institute of technology"),
    (r"\bnit\b", "national institute of technology"),
    (r"\biim\b", "indian institute of management"),
    (r"\baiims\b", "all india institute of medical sciences"),
    (r"\bjnu\b", "jawaharlal nehru university"),
    (r"\bbhu\b", "banaras hindu university"),
    (r"\bamu\b", "aligarh muslim university"),
    (r"\bdu\b", "university of delhi"),
]
CITY_SYNONYMS = {"bengaluru": "bangalore", "chennai": "madras", "mumbai": "bombay",
                 "kolkata": "calcutta", "thiruvananthapuram": "trivandrum"}


def _norm(text):
    text = str(text).lower().replace("&", " and ").replace("`", "'")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [CITY_SYNONYMS.get(t, t) for t in text.split()]
    return " ".join(tokens)


def _expand_query(query):
    q = str(query).lower()
    for pattern, repl in ALIASES:
        q = re.sub(pattern, repl, q)
    return _norm(q)


def match_names(query, names):
    """Return candidate institution names, best match first.

    A name matches when it contains every word of the (alias-expanded) query.
    Names with the fewest extra words win, so "IISc" picks the IISc variants
    rather than every IISER. Falls back to fuzzy matching.
    """
    q_tokens = set(_expand_query(query).split())
    if not q_tokens:
        return []
    scored = []
    for name in names:
        n_tokens = set(_norm(name).split())
        if q_tokens <= n_tokens:
            # fewer extra words first; prefer properly-cased names over ALL-CAPS variants
            scored.append((len(n_tokens - q_tokens), str(name).isupper(), name))
    if scored:
        best = min(s[0] for s in scored)
        return [n for extra, _, n in sorted(scored) if extra <= best + 1][:8]
    normed = {_norm(n): n for n in names}
    close = difflib.get_close_matches(_expand_query(query), list(normed), n=5, cutoff=0.6)
    return [normed[c] for c in close]


# ------------------------------------------------------------------------------
# Helpers shared by the tools
# ------------------------------------------------------------------------------
def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 2)
    return value


def _records(frame, cols=None):
    cols = cols or [c for c in ["Year", "Category", "Rank", "Name", "City", "State", "Score",
                                "TLR", "RPC", "GO", "OI", "Perception"] if c in frame.columns]
    return [{c: _clean(v) for c, v in row.items()} for row in frame[cols].head(MAX_ROWS).to_dict("records")]


class ToolError(Exception):
    pass


def build_tools(df, log, charts):
    """Create the tool functions bound to this DataFrame.

    `log` collects a readable trace of every call; `charts` collects Plotly
    figures created during the turn.
    """
    categories = sorted(df["Category"].unique())

    def resolve_category(category):
        if not category:
            return "Overall"
        c = str(category).strip().lower().replace(" ", "")
        for cat in categories:
            if cat.lower() == c or cat.lower().rstrip("s") == c.rstrip("s"):
                return cat
        synonyms = {"dentistry": "Dental", "medicine": "Medical", "pharma": "Pharmacy", "business": "Management",
                    "mba": "Management", "engg": "Engineering", "technology": "Engineering",
                    "universities": "University", "colleges": "College", "statepublic": "STATEPUBLICUNIVERSITY",
                    "statepublicuniversities": "STATEPUBLICUNIVERSITY", "all": "Overall"}
        if c in synonyms:
            return synonyms[c]
        close = difflib.get_close_matches(c, [x.lower() for x in categories], n=1, cutoff=0.75)
        if close:
            return next(x for x in categories if x.lower() == close[0])
        raise ToolError(f"Unknown category '{category}'. Valid categories: {categories}")

    def resolve_years(cat_df, year_from, year_to):
        years = sorted(int(y) for y in cat_df["Year"].unique())
        lo = int(year_from) if year_from else (int(year_to) if year_to else years[-1])
        hi = int(year_to) if year_to else lo
        lo, hi = min(lo, hi), max(lo, hi)
        if not any(lo <= y <= hi for y in years):
            raise ToolError(f"No data for {lo}-{hi} in this category. Available years: {years}")
        return lo, hi

    def resolve_metric(metric, allow_rank=False):
        lookup = {m.lower(): m for m in METRICS}
        lookup.update({"rank": "Rank_Num", "rank_num": "Rank_Num", "pr": "Perception", "overall": "Score",
                       "score": "Score", "research": "RPC", "teaching": "TLR",
                       "graduation": "GO", "outreach": "OI", "inclusivity": "OI"})
        m = lookup.get(str(metric).strip().lower())
        if m is None or (m == "Rank_Num" and not allow_rank):
            raise ToolError(f"Unknown metric '{metric}'. Use one of {METRICS}" + (" or Rank" if allow_rank else ""))
        return m

    def slice_df(category, year_from=0, year_to=0, states=None):
        cat = resolve_category(category)
        cat_df = df[df["Category"] == cat]
        lo, hi = resolve_years(cat_df, year_from, year_to)
        out = cat_df[cat_df["Year"].between(lo, hi)]
        if states:
            wanted = {s.strip().lower() for s in states}
            out = out[out["State"].str.lower().isin(wanted)]
            if out.empty:
                raise ToolError(f"No institutions for states {states}. Example valid states: "
                                f"{sorted(cat_df['State'].unique())[:15]}")
        return cat, lo, hi, out

    def traced(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            entry = {"tool": fn.__name__, "args": {k: v for k, v in kwargs.items() if v not in (None, "", 0, [])}}
            try:
                result = fn(*args, **kwargs)
            except ToolError as e:
                result = {"error": str(e)}
            except Exception as e:  # never let a tool crash the chat
                result = {"error": f"{type(e).__name__}: {e}"}
            entry["result"] = result.get("error") or result.get("summary") or "ok"
            log.append(entry)
            return result
        return wrapper

    # ---------------------------------------------------------------- tools
    @traced
    def find_institutions(name: str, category: Optional[str] = None) -> dict:
        """Find the exact official name(s) of an institution from a partial name or
        abbreviation (e.g. "IIT Madras", "IISc", "AIIMS Delhi"). Returns matching
        names with the categories and years they appear in. Use this when unsure of
        a name before calling other tools.

        Args:
            name: Full or partial institution name, abbreviations allowed.
            category: Optional ranking category to restrict the search to.
        """
        pool = df if not category else df[df["Category"] == resolve_category(category)]
        found = match_names(name, pool["Name"].unique())
        if not found:
            return {"error": f"No institution matches '{name}'."}
        info = []
        for n in found:
            rows = pool[pool["Name"] == n]
            info.append({"name": n, "state": rows["State"].iloc[0],
                         "categories": sorted(rows["Category"].unique()),
                         "years": sorted(int(y) for y in rows["Year"].unique())})
        return {"summary": f"{len(found)} match(es)", "matches": info}

    @traced
    def get_rankings(category: str = "Overall", year: int = 0, state: Optional[str] = None,
                     name_contains: Optional[str] = None, sort_by: str = "Rank",
                     limit: int = 10, max_rank: int = 0) -> dict:
        """List ranked institutions for one category and year, e.g. the top 10, the
        institutions of one state, or institutions whose name matches a phrase.

        Args:
            category: Ranking category, e.g. Overall, Engineering, Medical, Management.
            year: Ranking year (2017-2025). 0 means the latest year available.
            state: Optional state filter, e.g. "Tamil Nadu".
            name_contains: Optional name filter; abbreviations like IIT/NIT/IIM work.
            sort_by: Rank (default), Score, TLR, RPC, GO, OI or Perception.
            limit: Maximum rows to return (1-50).
            max_rank: Only institutions with exact rank <= this value (e.g. 100). 0 = no limit.
        """
        cat, y, _, sub = slice_df(category, year, year, [state] if state else None)
        if name_contains:
            sub = sub[sub["Name"].isin(match_names(name_contains, sub["Name"].unique())) |
                      sub["Name"].str.contains(name_contains, case=False, regex=False)]
        if max_rank:
            sub = sub[sub["Rank_Num"] <= max_rank]
        total = len(sub)
        if str(sort_by).lower() == "rank":
            sub = sub.sort_values(["Rank_Num", "Name"], na_position="last")
        else:
            m = resolve_metric(sort_by)
            sub = sub.sort_values(m, ascending=False, na_position="last")
        limit = max(1, min(int(limit or 10), MAX_ROWS))
        return {"summary": f"{cat} {y}: {total} matching, showing {min(limit, total)}",
                "category": cat, "year": y, "total_matching": total,
                "rows_without_scores": int(sub["Score"].isna().sum()),
                "rows": _records(sub.head(limit))}

    @traced
    def institution_history(name: str, category: str = "Overall") -> dict:
        """Year-by-year rank and scores of one institution in a category (2017-2025).
        Includes official name variants (some institutes were renamed between years).

        Args:
            name: Institution name or abbreviation, e.g. "IIT Bombay".
            category: Ranking category (default Overall).
        """
        cat = resolve_category(category)
        pool = df[df["Category"] == cat]
        found = match_names(name, pool["Name"].unique())
        if not found:
            other = match_names(name, df["Name"].unique())
            hint = f" It does appear in: {sorted(df[df['Name'].isin(other)]['Category'].unique())}" if other else ""
            return {"error": f"'{name}' not found in {cat}.{hint}"}
        rows = pool[pool["Name"].isin(found)].sort_values("Year")
        return {"summary": f"{len(rows)} yearly records", "category": cat,
                "matched_names": found, "history": _records(rows)}

    @traced
    def aggregate_stats(group_by: str = "State", metric: str = "count", agg: str = "mean",
                        category: str = "Overall", year_from: int = 0, year_to: int = 0,
                        max_rank: int = 0, limit: int = 15) -> dict:
        """Group institutions and aggregate, e.g. number of institutions per state,
        average score per year, median RPC per state.

        Args:
            group_by: State, Year, City or Category.
            metric: "count" to count distinct institutions, or Score/TLR/RPC/GO/OI/Perception.
            agg: For numeric metrics: mean, median, max, min or std.
            category: Ranking category, or "All" to include every category.
            year_from: First year (0 = latest year).
            year_to: Last year (0 = same as year_from).
            max_rank: Only count institutions with exact rank <= this (e.g. 100 for top 100). 0 = all.
            limit: Max groups returned (1-50).
        """
        gb = str(group_by).strip().title()
        if gb not in ("State", "Year", "City", "Category"):
            raise ToolError("group_by must be State, Year, City or Category")
        if str(category).lower() == "all":
            sub = df
            years = sorted(df["Year"].unique())
            lo = int(year_from) if year_from else int(years[-1])
            hi = int(year_to) if year_to else lo
            sub = sub[sub["Year"].between(min(lo, hi), max(lo, hi))]
            cat = "All categories"
        else:
            cat, lo, hi, sub = slice_df(category, year_from, year_to)
        if max_rank:
            sub = sub[sub["Rank_Num"] <= max_rank]
        if str(metric).lower() == "count":
            res = sub.groupby(gb)["Name"].nunique().rename("institutions")
            res = res.sort_index() if gb == "Year" else res.sort_values(ascending=False)
        else:
            m = resolve_metric(metric)
            if agg not in ("mean", "median", "max", "min", "std"):
                raise ToolError("agg must be mean, median, max, min or std")
            grp = sub.dropna(subset=[m]).groupby(gb)[m]
            res = pd.DataFrame({f"{agg}_{m}": grp.agg(agg), "n": grp.size()})
            res = res.sort_index() if gb == "Year" else res.sort_values(f"{agg}_{m}", ascending=False)
        res = res.head(max(1, min(int(limit or 15), MAX_ROWS))).reset_index()
        return {"summary": f"{len(res)} groups", "category": cat, "years": f"{lo}-{hi}",
                "filter": f"rank <= {max_rank}" if max_rank else "all ranked",
                "rows": _records(res, list(res.columns))}

    @traced
    def correlation(x: str, y: str = "Score", category: str = "Overall",
                    year_from: int = 0, year_to: int = 0) -> dict:
        """Pearson and Spearman correlation between two metrics across institutions.

        Args:
            x: First metric (Score, TLR, RPC, GO, OI, Perception, or Rank).
            y: Second metric (default Score).
            category: Ranking category.
            year_from: First year (0 = latest).
            year_to: Last year (0 = same as year_from).
        """
        mx, my = resolve_metric(x, True), resolve_metric(y, True)
        cat, lo, hi, sub = slice_df(category, year_from, year_to)
        d = sub[[mx, my]].dropna()
        if len(d) < 3 or d[mx].std() == 0 or d[my].std() == 0:
            return {"error": "Not enough scored institutions to compute a correlation."}
        r, rho = d[mx].corr(d[my]), d[mx].corr(d[my], method="spearman")
        return {"summary": f"r={r:.3f}", "category": cat, "years": f"{lo}-{hi}", "x": mx, "y": my,
                "pearson_r": round(float(r), 3), "spearman_rho": round(float(rho), 3), "n": int(len(d))}

    @traced
    def create_chart(chart_type: str, category: str = "Overall", year_from: int = 0, year_to: int = 0,
                     metric: str = "Score", institutions: Optional[list[str]] = None,
                     x_metric: str = "RPC", y_metric: str = "Score", top_n: int = 10,
                     states: Optional[list[str]] = None, state_measure: str = "count") -> dict:
        """Create an interactive chart that is displayed under your answer. Use it
        whenever a visual helps (rankings, trends, comparisons, distributions).

        Args:
            chart_type: One of: top_bar (top N institutions by metric), trend (metric over
                years for given institutions), scatter (x_metric vs y_metric), distribution
                (box plot of metric by year), radar (profile of up to 5 institutions on all
                five parameters for one year), correlation_heatmap, state_bar (states by
                number of institutions or average score).
            category: Ranking category.
            year_from: First year (0 = latest). For trend, defaults to the full 2017-2025 span.
            year_to: Last year (0 = same as year_from).
            metric: Score, TLR, RPC, GO, OI, Perception, or Rank (trend only).
            institutions: Institution names/abbreviations for trend or radar.
            x_metric: X axis metric for scatter.
            y_metric: Y axis metric for scatter.
            top_n: Number of bars for top_bar / state_bar (3-50).
            states: Optional list of states to filter to.
            state_measure: For state_bar: "count" or "mean" (average score).
        """
        if len(charts) >= MAX_CHARTS:
            return {"error": f"Chart limit reached ({MAX_CHARTS} per answer)."}
        ct = str(chart_type).strip().lower()
        if ct == "trend" and not year_from and not year_to:
            year_from, year_to = 2017, 2025
        cat_df = df[df["Category"] == resolve_category(category)]
        if ct == "trend" and year_from and year_to:  # clamp a full span to what exists
            ys = cat_df["Year"]
            year_from, year_to = max(int(year_from), ys.min()), min(int(year_to), ys.max())
        cat, lo, hi, sub = slice_df(category, year_from, year_to, states)
        scored = sub.dropna(subset=["Score"])
        top_n = max(3, min(int(top_n or 10), 50))
        yrs = str(lo) if lo == hi else f"{lo}-{hi}"

        def resolve_insts(limit):
            if not institutions:
                raise ToolError("Provide 'institutions' for this chart type.")
            names = []
            for q in institutions:
                m = match_names(q, sub["Name"].unique())
                if not m:
                    raise ToolError(f"'{q}' not found in {cat} {yrs}.")
                names.extend(n for n in m if n not in names)
            return names[:limit]

        if ct == "top_bar":
            fig = dc.create_top_n_bar_chart(scored, resolve_metric(metric), top_n, "state", "mean")
        elif ct == "trend":
            fig = dc.create_time_series_chart(sub, resolve_insts(10), resolve_metric(metric, True))
        elif ct == "scatter":
            fig, _ = dc.create_bivariate_scatter(scored, resolve_metric(x_metric), resolve_metric(y_metric),
                                                 True, "State")
        elif ct == "distribution":
            fig = dc.create_distribution_chart(scored, resolve_metric(metric), "Year" if hi > lo else "None", "box")
        elif ct == "radar":
            one_year = scored[scored["Year"] == hi]
            fig = dc.create_radar_comparison(one_year, resolve_insts(5), hi, benchmark_df=one_year)
        elif ct == "correlation_heatmap":
            fig = dc.create_correlation_heatmap(scored)
        elif ct == "state_bar":
            fig = dc.create_state_aggregation_chart(sub, top_n, "mean" if state_measure == "mean" else "count", 3)
        else:
            raise ToolError("chart_type must be top_bar, trend, scatter, distribution, radar, "
                            "correlation_heatmap or state_bar")
        charts.append(fig)
        return {"summary": f"{ct} chart for {cat} {yrs}", "status": "Chart created and shown below the answer."}

    return [find_institutions, get_rankings, institution_history, aggregate_stats, correlation, create_chart]


# ------------------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------------------
def build_system_prompt(df):
    cats = df.groupby("Category")["Year"].agg(["min", "max"])
    cat_lines = "\n".join(f"- {c}: {int(r['min'])}-{int(r['max'])}" for c, r in cats.iterrows())
    no_scores = [c for c, g in df.groupby("Category") if g["Score"].isna().all()]
    return f"""You are the NIRF Analytics Assistant inside a dashboard about India's National
Institutional Ranking Framework (NIRF) rankings, 2017-2025 ({len(df):,} records).

Data facts:
- Categories and the years they were published:
{cat_lines}
- Each record: Year, Category, Rank, Name, City, State, Score (0-100) and five parameter
  scores (0-100): TLR = Teaching, Learning & Resources; RPC = Research & Professional
  Practice; GO = Graduation Outcomes; OI = Outreach & Inclusivity; Perception (PR).
- Institutions beyond the top ~100 are published only in rank bands (e.g. "101-150")
  with NO scores. Categories with no published scores at all: {no_scores or 'none'}.
- Some institutions changed official names across years; tools return name variants.

Rules:
1. ALWAYS use the tools to get numbers. Never state a rank, score, count or correlation
   that did not come from a tool result in this conversation. If a tool returns an
   error, fix the arguments and retry, or explain what is not available.
2. If the user doesn't specify a year, use the latest year available for that category
   and say which year you used. If no category is given, use Overall.
3. Add a chart with create_chart when it makes the answer clearer (rankings, trends,
   comparisons, distributions). Don't describe the chart's styling; refer to it briefly.
4. Answer in concise Markdown: a direct answer first, then a short list or small table
   of key figures, then (optionally) one sentence of insight. Use institution names as
   they appear in the data.
5. Only answer questions about this NIRF dataset and higher-education rankings. For
   anything else, politely say what you can help with.
"""


# ------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------
class AssistantError(Exception):
    pass


def classify_error(exc):
    """Map an exception to (kind, short reason).

    kind: "auth"    - bad key; every model would fail, so stop
          "network" - no connection; every model would fail, so stop
          "busy"    - overloaded / rate-limited / timed out: try the next model
          "missing" - model not found or not available to this key: try the next model
          "other"   - anything else: try the next model
    """
    msg = str(exc)
    code = getattr(exc, "code", None)
    name = type(exc).__name__
    if code in (401, 403) or any(s in msg for s in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED")):
        return "auth", "API key rejected"
    if "Connect" in name or "getaddrinfo" in msg or "Name or service not known" in msg:
        return "network", "no internet connection"
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        return "busy", "rate limit / quota reached"
    if code in (500, 502, 503, 504) or any(s in msg for s in ("UNAVAILABLE", "overloaded", "high demand", "INTERNAL")):
        return "busy", "overloaded / high demand"
    if "Timeout" in name or "DEADLINE_EXCEEDED" in msg or "timed out" in msg.lower():
        return "busy", "timed out"
    if code == 404 or "NOT_FOUND" in msg:
        return "missing", "model not found"
    return "other", msg[:120]


def _call_model(client, types, model, contents, df):
    """One attempt with one model. Tool log and charts are fresh per attempt so a
    failed attempt never leaks half-finished charts into the answer."""
    log, charts = [], []
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(df),
            tools=build_tools(df, log, charts),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=12),
        ),
    )
    try:
        text = response.text
    except Exception:
        text = None
    return text, charts, log


def ask(question, history, df):
    """Answer `question` given prior chat `history` ([{role, content}]).

    Tries each configured model in order until one succeeds.
    Returns {"text", "charts", "tool_calls", "model", "fallbacks": [(model, reason), ...]}.
    Raises AssistantError with a user-friendly message if no model could answer.
    """
    key, models = get_config()
    if not key:
        raise AssistantError("No Gemini API key found. Add GEMINI_API_KEY to the .env file.")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise AssistantError("The google-genai package is missing. Run: pip install google-genai python-dotenv")

    contents = []
    for m in history[-12:]:  # last few turns are enough context
        if m.get("content"):
            contents.append(types.Content(role="user" if m["role"] == "user" else "model",
                                          parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    skipped = []  # (model, reason) for every model that could not answer
    for model in models:
        for attempt in (1, 2):
            try:
                text, charts, log = _call_model(client, types, model, contents, df)
            except Exception as e:
                kind, reason = classify_error(e)
                if kind == "auth":
                    raise AssistantError("Gemini rejected the API key. Check GEMINI_API_KEY in the .env file.")
                if kind == "network":
                    raise AssistantError("Couldn't reach Gemini. Check your internet connection and try again.")
                # Server-side hiccups often clear in a moment: retry the same model once.
                if attempt == 1 and kind == "busy" and getattr(e, "code", None) in (500, 502, 503, 504):
                    time.sleep(RETRY_SAME_MODEL_DELAY_S)
                    continue
                skipped.append((model, reason))
                break
            if not text:
                text = ("I couldn't finish working that out. Try asking more specifically, "
                        "e.g. naming the category and year.")
            return {"text": text, "charts": charts, "tool_calls": log, "model": model, "fallbacks": skipped}

    detail = "; ".join(f"{m}: {r}" for m, r in skipped)
    if skipped and all(r in ("rate limit / quota reached", "overloaded / high demand", "timed out")
                       for _, r in skipped):
        raise AssistantError(f"All Gemini models are busy right now ({detail}). Please try again in a minute.")
    raise AssistantError(f"No Gemini model could answer ({detail}). Check GEMINI_MODEL / "
                         "GEMINI_FALLBACK_MODELS in .env.")
