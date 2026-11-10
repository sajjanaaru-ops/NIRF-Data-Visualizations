# questions_visualizer.py
# NIRF 2017-2025 Annexure A Visualizations (All 40 Questions)
#
# Every question has a plot function (plot_q_X_Y) and a finding function (finding_q_X_Y).
# Both take the same five data objects and read the same numbers, so the
# text shown under a chart can never disagree with the chart itself.
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '.mplconfig'))
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Global styling
sns.set_theme(style='white', font='sans-serif')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = '#d0d0d0'
plt.rcParams['axes.linewidth'] = 0.8

PARAMS = ['TLR', 'RPC', 'GO', 'OI', 'Perception']
FIRST, LATEST = 2017, 2025
MINUS = '−'
ARROW = '→'
TIE_NOTE = f'Ties broken by {LATEST} overall score (higher first), then name (A–Z).'
TIE_NOTE_STATE = 'Ties broken by state name (A–Z).'
TIERS = ['1–10', '11–25', '26–50', '51–100', '101–150', '151–200']


# ---- Data cleaning -----------------------------------------------------------------------------
_NAME_ALIASES = {
    'indianinstituteofsciencebangalore': 'indianinstituteofscience',
    'indianinstituteofsciencebengaluru': 'indianinstituteofscience',
}


def _name_key(name):
    k = re.sub(r'[^a-z]', '', str(name).lower().replace('&', 'and'))
    return _NAME_ALIASES.get(k, k)


def canonicalize_names(frame):
    """One name per institution across years.

    NIRF spells many institutions differently from year to year ("IIT Delhi" vs
    "Indian Institute of Technology, Delhi"; IISc has three spellings). Rows are linked when they
    share the stable Institute-ID suffix (e.g. "U-0456", used from 2018 onward) or the same
    normalised name within the same state. Each linked group takes its most recent name. A group
    that would end up with two rows in the same year is left unmerged (different campuses).
    """
    frame = frame.copy()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    suffix = frame['Institute ID'].astype(str).str.extract(r'-([A-Z]+-\d+)$')[0]
    nodes = [('k', c, str(s).strip().lower(), _name_key(n))
             for c, s, n in zip(frame['Category'], frame['State'], frame['Name'])]
    for node, cat, suf in zip(nodes, frame['Category'], suffix):
        find(node)
        if isinstance(suf, str):
            parent[find(node)] = find(('s', cat, suf))
    groups = pd.Series([find(n) for n in nodes], index=frame.index)

    per_year = frame.groupby([groups, frame['Year']]).size()
    clashing = set(per_year[per_year > 1].index.get_level_values(0))
    latest_name = frame.assign(_g=groups).sort_values('Year').groupby('_g')['Name'].last()
    merge = ~groups.isin(clashing)
    frame.loc[merge, 'Name'] = groups[merge].map(latest_name)
    return frame


@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nirf_cleaned_2017_2025.csv')
    df = pd.read_csv(csv_path)
    df['State'] = df['State'].replace({'Jammu & Kashmir': 'Jammu and Kashmir'})
    overall_mask = df['Category'] == 'Overall'
    df.loc[overall_mask] = canonicalize_names(df[overall_mask])
    df_overall = df[overall_mask].copy()
    df_overall['Rank_Num'] = pd.to_numeric(df_overall['Rank'], errors='coerce')
    params = list(PARAMS)
    pivot_rank = df_overall.pivot_table(index='Name', columns='Year', values='Rank_Num')
    rank_diff = (pivot_rank[FIRST] - pivot_rank[LATEST]).dropna()
    return df, df_overall, params, pivot_rank, rank_diff


# ---- Shared helpers ----------------------------------------------------------------------------
_ABBREV = [
    ('Indian Institute of Science Education and Research', 'IISER'),
    ('Indian Institute of Science Education & Research', 'IISER'),
    ('Indian Institute of Information Technology', 'IIIT'),
    ('Indian Institute of Technology', 'IIT'),
    ('Indian Institute of Science', 'IISc'),
    ('National Institute of Technology', 'NIT'),
    ('All India Institute of Medical Sciences', 'AIIMS'),
    ('Institute of Higher Education and Research', 'Inst. of Higher Ed. & Research'),
    ('Institute of Medical and Technical Sciences', 'Inst. of Med. & Tech. Sciences'),
    ('Academy of Higher Education', 'Academy of Higher Ed.'),
    ('(Indian School of Mines)', '(ISM)'),
    ('(Banaras Hindu University)', '(BHU)'),
    ('University', 'Univ.'),
]


def short_name(name, max_len=40, abbrev_univ=True):
    s = str(name)
    for long, short in _ABBREV:
        if long == 'University' and not abbrev_univ:
            continue
        s = s.replace(long, short)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if len(s) <= max_len else s[:max_len - 1].rstrip() + '…'


def fname(name):
    """Readable name for finding text: standard acronyms (IIT, NIT...), never truncated."""
    return short_name(name, max_len=500, abbrev_univ=False)


def signed(v, fmt='.1f', unit=''):
    if pd.isna(v):
        return 'n/a'
    if round(v, 6) == 0:
        return f'±{0:{fmt}}{unit}'
    return f"{'+' if v > 0 else MINUS}{abs(v):{fmt}}{unit}"


def move_label(old, new, brackets=True):
    """Rank move text, e.g. '#91 → #23  (+68)'. A smaller rank is better."""
    delta = signed(old - new, ".0f")
    return f'#{int(old)} → #{int(new)}  ({delta})' if brackets else f'#{int(old)} → #{int(new)}, {delta}'


def r_strength(r):
    a = abs(r)
    return ('very strong' if a >= 0.8 else 'strong' if a >= 0.6 else 'moderate' if a >= 0.4
            else 'weak' if a >= 0.2 else 'very weak')


def _latest_scores(df_overall):
    return df_overall[df_overall['Year'] == LATEST].groupby('Name')['Score'].max()


def top_n(frame, col, n=None, ascending=False, df_overall=None):
    """Rows sorted by `col`; ties broken by the latest overall score (higher first), then name (A–Z)."""
    f = frame.copy()
    if 'Name' not in f.columns:
        f = f.rename_axis('Name').reset_index()
    tb = f['Name'].map(_latest_scores(df_overall)) if df_overall is not None else pd.Series(0, index=f.index)
    f['_tb'] = tb.fillna(-1)
    f = f.sort_values([col, '_tb', 'Name'], ascending=[ascending, False, True], kind='mergesort').drop(columns='_tb')
    return f.head(n) if n else f


def top_series(s, n=None, ascending=False, df_overall=None):
    """Series indexed by institution name, sorted with the same tiebreaker."""
    f = top_n(s.rename('_v').rename_axis('Name').reset_index(), '_v', n, ascending, df_overall)
    return f.set_index('Name')['_v']


def top_counts(counts, n=None):
    """Counts indexed by a label (e.g. state), sorted by count desc then label A–Z."""
    f = counts.rename('_v').rename_axis('_k').reset_index().sort_values(['_v', '_k'], ascending=[False, True])
    s = f.set_index('_k')['_v']
    s.index.name = None
    return s.head(n) if n else s


def footnote(text):
    plt.gcf().text(0.01, -0.02, text, fontsize=8, color='#8a8a8e', ha='left', va='top')


def tier_of(rank_band):
    """Overall rank tier from a Rank_Band value ('7', '101-150', ...)."""
    s = str(rank_band).strip()
    if s.startswith('101'):
        return TIERS[4]
    if s.startswith('151'):
        return TIERS[5]
    try:
        r = float(s)
    except ValueError:
        return None
    return TIERS[0] if r <= 10 else TIERS[1] if r <= 25 else TIERS[2] if r <= 50 else TIERS[3]


def _names(items, n=None):
    items = list(items)
    shown = [fname(x) for x in (items[:n] if n else items)]
    if not shown:
        return 'none'
    s = ', '.join(shown[:-1]) + (' and ' if len(shown) > 1 else '') + shown[-1]
    return s + (f' (+{len(items) - n} more)' if n and len(items) > n else '')


def _barh(labels, values, palette, n=None):
    n = n or len(values)
    bars = plt.barh([short_name(x, 46) for x in labels], values,
                    color=sns.color_palette(palette, n_colors=max(n, 1)), height=0.5)
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    return bars


# ================================================================================================
# 1. Time-Series and Trend Analysis
# ================================================================================================
DEFAULT_INSTITUTION = 'Indian Institute of Technology Madras'


def plot_q_1_1(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    inst_df = df_overall[df_overall['Name'] == DEFAULT_INSTITUTION].sort_values('Year')
    plt.figure(figsize=(8, 3.8))
    plt.plot(inst_df['Year'], inst_df['Rank_Num'], marker='o', markersize=8, color='#1f77b4', linewidth=2.5)
    plt.gca().invert_yaxis()
    plt.title('IIT Madras NIRF Overall Ranking (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Year')
    plt.ylabel('Rank (1 = Top)')
    plt.ylim(5, 0.5)
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    return plt.gcf()


def finding_q_1_1(df, df_overall, params, pivot_rank, rank_diff, name=DEFAULT_INSTITUTION):
    inst = df_overall[df_overall['Name'] == name].sort_values('Year')
    ranked = inst.dropna(subset=['Rank_Num'])
    if ranked.empty:
        bands = ', '.join(f"{int(y)}: {b}" for y, b in zip(inst['Year'], inst['Rank_Band']))
        return f'{name} never held an exact Top-100 rank; its rank bands were {bands}.'
    first, last = ranked.iloc[0], ranked.iloc[-1]
    best = ranked['Rank_Num'].min()
    best_years = ranked.loc[ranked['Rank_Num'] == best, 'Year'].astype(int).tolist()
    worst = ranked.loc[ranked['Rank_Num'].idxmax()]
    text = (f"{fname(name)} was ranked #{int(first.Rank_Num)} in {int(first.Year)} and #{int(last.Rank_Num)} "
            f"in {int(last.Year)} (exact ranks in {len(ranked)} of 9 editions). Its best rank was #{int(best)} "
            f"({', '.join(map(str, best_years))}) and its worst was #{int(worst.Rank_Num)} ({int(worst.Year)}).")
    streak = 0
    for r in ranked['Rank_Num'][::-1]:
        if r != last.Rank_Num:
            break
        streak += 1
    if streak > 1:
        text += f' It has held #{int(last.Rank_Num)} for {streak} consecutive editions up to {int(last.Year)}.'
    return text


def _rank_moves(df_overall, rank_diff, ascending):
    moves = top_series(rank_diff, 6, ascending=ascending, df_overall=df_overall)
    piv = df_overall.pivot_table(index='Name', columns='Year', values='Rank_Num')
    return moves, piv


def plot_q_1_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    moves, piv = _rank_moves(df_overall, rank_diff, ascending=False)
    plt.figure(figsize=(9, 3.8))
    bars = _barh(moves.index, moves.values, 'Greens_r', 6)
    plt.title('Greatest Rank Improvement (2017 → 2025)', fontsize=12, fontweight='bold', pad=10)
    for b, name in zip(bars, moves.index):
        plt.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, move_label(piv.at[name, FIRST], piv.at[name, LATEST]),
                 va='center', fontweight='bold', color='#1b5e20')
    plt.xlim(0, moves.max() * 1.45)
    footnote('Institutions holding an exact Top-100 rank in both 2017 and 2025. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_1_2(df, df_overall, params, pivot_rank, rank_diff):
    moves, piv = _rank_moves(df_overall, rank_diff, ascending=False)
    parts = [f"{fname(n)} ({move_label(piv.at[n, FIRST], piv.at[n, LATEST], False)})" for n in moves.index[:3]]
    return (f"Of the {len(rank_diff)} institutions ranked in the Top 100 in both 2017 and 2025, the biggest climbers were "
            f"{parts[0]}, {parts[1]} and {parts[2]}.")


def plot_q_1_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    moves, piv = _rank_moves(df_overall, rank_diff, ascending=True)
    drops = moves.abs()
    plt.figure(figsize=(9, 3.8))
    bars = _barh(drops.index, drops.values, 'Reds_r', 6)
    plt.title('Largest Rank Decline (2017 → 2025)', fontsize=12, fontweight='bold', pad=10)
    for b, name in zip(bars, drops.index):
        plt.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, move_label(piv.at[name, FIRST], piv.at[name, LATEST]),
                 va='center', fontweight='bold', color='#b71c1c')
    plt.xlim(0, drops.max() * 1.45)
    footnote('Institutions holding an exact Top-100 rank in both 2017 and 2025. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_1_3(df, df_overall, params, pivot_rank, rank_diff):
    moves, piv = _rank_moves(df_overall, rank_diff, ascending=True)
    parts = [f"{fname(n)} ({move_label(piv.at[n, FIRST], piv.at[n, LATEST], False)})" for n in moves.index[:3]]
    n_big = int((rank_diff <= -30).sum())
    return (f"The steepest falls between 2017 and 2025 were {parts[0]}, {parts[1]} and {parts[2]}. "
            f"{n_big} institutions dropped 30 or more places.")


def _avg_scores(df_overall):
    scored = df_overall.dropna(subset=['Score'])
    top100 = scored.groupby('Year')['Score'].mean()
    years_scored = scored.groupby('Name')['Year'].nunique()
    cohort_names = years_scored[years_scored == scored['Year'].nunique()].index
    cohort = scored[scored['Name'].isin(cohort_names)].groupby('Year')['Score'].mean()
    return top100, cohort, len(cohort_names)


def plot_q_1_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    top100, cohort, n_cohort = _avg_scores(df_overall)
    plt.figure(figsize=(8.5, 3.9))
    plt.plot(top100.index, top100.values, marker='s', color='#1f77b4', linewidth=2.5, markersize=7,
             label="Each year's Top-100 (published scores)")
    plt.plot(cohort.index, cohort.values, marker='o', color='#ff7f0e', linewidth=2, markersize=6, linestyle='--',
             label=f'Fixed cohort: {n_cohort} institutions in Top-100 all 9 years')
    for s, c in ((top100, '#1f77b4'), (cohort, '#e65100')):
        for x in (s.index[0], s.index[-1]):
            plt.annotate(f'{s[x]:.1f}', (x, s[x]), textcoords='offset points', xytext=(0, 8), ha='center',
                         fontweight='bold', color=c, fontsize=9)
    plt.title('Average NIRF Overall Score (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Year')
    plt.ylabel('Average Score (pts)')
    plt.xticks(range(FIRST, LATEST + 1))
    lo, hi = min(top100.min(), cohort.min()), max(top100.max(), cohort.max())
    plt.ylim(lo - 3, hi + 4)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=False, fontsize=8.5, loc='lower right')
    sns.despine()
    return plt.gcf()


def finding_q_1_4(df, df_overall, params, pivot_rank, rank_diff):
    top100, cohort, n_cohort = _avg_scores(df_overall)
    dips = [int(y) for y in top100.index[1:] if top100[y] < top100[y - 1]]
    dip_txt = f" The only year-on-year dip was in {', '.join(map(str, dips))}." if dips else ''
    return (f"The average score of each year's Top-100 rose from {top100.iloc[0]:.1f} in {FIRST} to {top100.iloc[-1]:.1f} "
            f"in {LATEST} ({signed(top100.iloc[-1] - top100.iloc[0])} pts).{dip_txt} The {n_cohort} institutions ranked "
            f"in every edition rose from {cohort.iloc[0]:.1f} to {cohort.iloc[-1]:.1f} ({signed(cohort.iloc[-1] - cohort.iloc[0])} pts), "
            f"so the rise is not just a change in which institutions are listed.")


def _top10_years(df_overall):
    counts = df_overall[df_overall['Rank_Num'] <= 10]['Name'].value_counts()
    counts = counts[counts >= 5]
    return top_series(counts, df_overall=df_overall), df_overall['Year'].nunique()


def plot_q_1_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    counts, n_years = _top10_years(df_overall)
    plt.figure(figsize=(9, 4))
    bars = _barh(counts.index, counts.values, 'Blues_r', len(counts))
    plt.title(f'Institutions in the Top 10 for at Least 5 of {n_years} Years', fontsize=12, fontweight='bold', pad=10)
    for b in bars:
        plt.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, f'{int(b.get_width())} / {n_years} yrs',
                 va='center', fontweight='bold', color='#0d47a1')
    footnote(TIE_NOTE)
    return plt.gcf()


def finding_q_1_5(df, df_overall, params, pivot_rank, rank_diff):
    counts, n_years = _top10_years(df_overall)
    always = counts[counts == n_years].index
    rest = counts[counts < n_years]
    rest_txt = '; '.join(f'{fname(n)} ({int(v)})' for n, v in rest.items())
    return (f"{len(always)} institutions were in the Top 10 in all {n_years} editions: {_names(always)}. "
            f"Also frequent: {rest_txt}.")


# ================================================================================================
# 2. Geospatial and Regional Analysis
# ================================================================================================
def _top100(df_overall, year):
    return df_overall[(df_overall['Year'] == year) & (df_overall['Rank_Num'] <= 100)]


def plot_q_2_1(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    counts = top_counts(_top100(df_overall, LATEST)['State'].value_counts(), 7)
    plt.figure(figsize=(9, 3.8))
    bars = _barh(counts.index, counts.values, 'Blues_r', len(counts))
    plt.title('States with Highest Representation in Top 100 (2025)', fontsize=12, fontweight='bold', pad=10)
    for b in bars:
        plt.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f'{int(b.get_width())} institutions',
                 va='center', fontweight='bold', color='#0d47a1')
    footnote(TIE_NOTE_STATE)
    return plt.gcf()


def finding_q_2_1(df, df_overall, params, pivot_rank, rank_diff):
    counts = top_counts(_top100(df_overall, LATEST)['State'].value_counts())
    top3 = ', '.join(f'{s} ({int(v)})' for s, v in counts.head(4).iloc[1:].items())
    return (f"{counts.index[0]} leads with {int(counts.iloc[0])} of the 2025 Top-100 institutions, followed by {top3}. "
            f"In total, {len(counts)} states and UTs have at least one Top-100 institution.")


def _state_trend(df_overall):
    t100 = df_overall[df_overall['Rank_Num'] <= 100]
    top4 = top_counts(t100['State'].value_counts(), 4).index.tolist()
    trend = t100[t100['State'].isin(top4)].groupby(['Year', 'State']).size().unstack(fill_value=0)[top4]
    return trend, top4, t100.groupby('Year').size()


def plot_q_2_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    trend, top4, totals = _state_trend(df_overall)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    plt.figure(figsize=(8.5, 4))
    last = trend.iloc[-1].sort_values()
    label_y, prev = {}, -99
    for s in last.index:                       # spread end labels so they never overlap
        y = max(last[s], prev + 1.1)
        label_y[s], prev = y, y
    for state, c in zip(top4, colors):
        plt.plot(trend.index, trend[state], marker='o', label=state, color=c, linewidth=2)
        plt.text(LATEST + 0.15, label_y[state], f' {state} ({int(trend[state].iloc[-1])})', va='center', fontweight='bold', color=c)
    plt.title('Top 4 States: Institutions in the Top 100 (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Year')
    plt.ylabel('Institutions in Top 100')
    plt.xticks(range(FIRST, LATEST + 1))
    plt.xlim(FIRST - 0.2, LATEST + 2.4)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    footnote('Top 4 = states with the most Top-100 appearances across all nine editions.')
    return plt.gcf()


def finding_q_2_2(df, df_overall, params, pivot_rank, rank_diff):
    trend, top4, totals = _state_trend(df_overall)
    share = trend.sum(axis=1) / totals * 100
    lead = top4[0]
    peak_year = int(trend[lead].idxmax())
    others = ', '.join(f'{s} {int(trend[s].iloc[0])}{ARROW}{int(trend[s].iloc[-1])}' for s in top4[1:])
    return (f"The four leading states ({', '.join(top4)}) held {share.iloc[0]:.0f}% of the Top 100 in {FIRST} and "
            f"{share.iloc[-1]:.0f}% in {LATEST}. {lead} remains first but slipped from a peak of {int(trend[lead].max())} "
            f"({peak_year}) to {int(trend[lead].iloc[-1])}; the others moved {others} ({FIRST}{ARROW}{LATEST}).")


def _state_change(df_overall, y_old=FIRST, y_new=LATEST):
    c_old = _top100(df_overall, y_old)['State'].value_counts()
    c_new = _top100(df_overall, y_new)['State'].value_counts()
    change = c_new.sub(c_old, fill_value=0)
    change = change[change != 0]
    order = change.rename('_v').rename_axis('_k').reset_index()
    order['_a'] = -order['_v']
    change = order.sort_values(['_a', '_k']).set_index('_k')['_v']
    return change, c_old, c_new


def plot_q_2_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    change, c_old, c_new = _state_change(df_overall)
    plt.figure(figsize=(9, max(3.8, 0.42 * len(change) + 1)))
    colors = ['#2e7d32' if v > 0 else '#d32f2f' for v in change.values]
    bars = plt.barh(change.index, change.values, color=colors, height=0.55)
    plt.gca().invert_yaxis()
    plt.axvline(0, color='#888888', linewidth=1)
    plt.title(f'Change in Top-100 Institutions by State ({FIRST} {ARROW} {LATEST})', fontsize=12, fontweight='bold', pad=10)
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    lim = change.abs().max() + 1.6
    plt.xlim(-lim, lim)
    for b, s in zip(bars, change.index):
        w = b.get_width()
        txt = f"{signed(w, '.0f')}  ({int(c_old.get(s, 0))}→{int(c_new.get(s, 0))})"
        plt.text(w + (0.08 if w > 0 else -0.08), b.get_y() + b.get_height() / 2, txt, va='center',
                 ha='left' if w > 0 else 'right', fontweight='bold', color='#1b5e20' if w > 0 else '#b71c1c', fontsize=9)
    footnote('States whose Top-100 count did not change are omitted. ' + TIE_NOTE_STATE)
    return plt.gcf()


def finding_q_2_3(df, df_overall, params, pivot_rank, rank_diff):
    change, c_old, c_new = _state_change(df_overall)
    gains, losses = change[change > 0], change[change < 0]
    g = ', '.join(f'{s} ({signed(v, ".0f")})' for s, v in gains.items()) or 'none'
    l = ', '.join(f'{s} ({signed(v, ".0f")})' for s, v in losses.iloc[::-1].items()) or 'none'
    return f"From {FIRST} to {LATEST}, Top-100 representation rose in {g}, and fell in {l}."


def _ka_tn(df_overall):
    return df_overall[(df_overall['Year'] == LATEST) & (df_overall['State'].isin(['Karnataka', 'Tamil Nadu']))].dropna(subset=['Rank_Num'])


def _ka_tn_banded(df_overall):
    """2025 institutions per state listed only in the 101–150 / 151–200 bands (no exact rank)."""
    d = df_overall[(df_overall['Year'] == LATEST) & (df_overall['State'].isin(['Karnataka', 'Tamil Nadu']))]
    return d[d['Rank_Num'].isna()].groupby('State').size().reindex(['Karnataka', 'Tamil Nadu'], fill_value=0)


def plot_q_2_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d = _ka_tn(df_overall)
    order = ['Karnataka', 'Tamil Nadu']
    plt.figure(figsize=(7.5, 4.2))
    sns.boxplot(data=d, x='State', y='Rank_Num', order=order, hue='State', hue_order=order, legend=False,
                palette=['#9ecae1', '#fdd0a2'], width=0.4, showfliers=False,
                medianprops=dict(color='#1d1d1f', linewidth=2))
    sns.stripplot(data=d, x='State', y='Rank_Num', order=order, hue='State', hue_order=order, legend=False,
                  palette=['#1f77b4', '#e6550d'], size=6, jitter=0.08, edgecolor='white', linewidth=0.6)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_ylim(103, -2)
    ax.set_yticks([1, 20, 40, 60, 80, 100])
    plt.title('Rank Comparison: Karnataka vs Tamil Nadu (2025 Overall)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('')
    plt.ylabel('Rank (1 = Top)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    for i, state in enumerate(order):
        ranks = d[d['State'] == state]['Rank_Num']
        med = ranks.median()
        plt.annotate(f'Median #{med:g}', xy=(i + 0.2, med), xytext=(i + 0.27, med), va='center', ha='left',
                     fontweight='bold', fontsize=9, arrowprops=dict(arrowstyle='-', color='#555555', lw=0.8),
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#cccccc'))
    ax.set_xticklabels([f"{s}\n(n = {(d['State'] == s).sum()})" for s in order])
    plt.xlim(-0.6, 1.9)
    banded = _ka_tn_banded(df_overall)
    footnote('Each dot is one institution with an exact 2025 Top-100 rank. The line inside each box is the median.\n'
             f'Not shown (listed only in the 101–200 bands, no exact rank): Karnataka {banded["Karnataka"]}, '
             f'Tamil Nadu {banded["Tamil Nadu"]}.')
    return plt.gcf()


def finding_q_2_4(df, df_overall, params, pivot_rank, rank_diff):
    d = _ka_tn(df_overall)
    out, stats = [], {}
    for s in ['Tamil Nadu', 'Karnataka']:
        x = d[d['State'] == s]
        best = x.loc[x['Rank_Num'].idxmin()]
        stats[s] = (len(x), x['Rank_Num'].median(), int(best['Rank_Num']))
        out.append(f"{s} has {len(x)} institutions in the Top 100 (median rank #{x['Rank_Num'].median():g}, best "
                   f"{fname(best['Name'])} #{int(best['Rank_Num'])}, {int((x['Rank_Num'] <= 50).sum())} in the Top 50)")
    (n_tn, med_tn, best_tn), (n_ka, med_ka, best_ka) = stats['Tamil Nadu'], stats['Karnataka']
    more = 'Tamil Nadu' if n_tn > n_ka else 'Karnataka' if n_ka > n_tn else None
    gap = abs(med_tn - med_ka)
    if gap == 0:
        med_txt = 'both states have the same median rank'
    else:
        better = 'Karnataka' if med_ka < med_tn else 'Tamil Nadu'
        med_txt = f"{better} has the better median rank by {gap:g} places"
    top_txt = (f"Tamil Nadu has the higher-ranked best institution (#{best_tn} vs #{best_ka})" if best_tn < best_ka
               else f"Karnataka has the higher-ranked best institution (#{best_ka} vs #{best_tn})")
    count_txt = (f"{more} has more Top-100 institutions ({max(n_tn, n_ka)} vs {min(n_tn, n_ka)})" if more
                 else f"Both states have {n_tn} Top-100 institutions")
    return f"In {LATEST}, " + '; '.join(out) + f". {count_txt}, {med_txt}, and {top_txt}."


def _new_entrants(df_overall):
    seen_first = set(df_overall[df_overall['Year'] == FIRST]['Name'])
    new = df_overall[(df_overall['Year'] == LATEST) & (~df_overall['Name'].isin(seen_first))]
    return new, top_counts(new['State'].value_counts(), 6)


def plot_q_2_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    new, counts = _new_entrants(df_overall)
    plt.figure(figsize=(9, 3.8))
    bars = _barh(counts.index, counts.values, 'Blues_r', len(counts))
    plt.title('States with the Most New Entrants in 2025 (Not Ranked in 2017)', fontsize=12, fontweight='bold', pad=10)
    for b in bars:
        plt.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2, f'{int(b.get_width())} new', va='center',
                 fontweight='bold', color='#0d47a1')
    footnote("'New' = in the 2025 Overall list (Top 200) but not in the 2017 list. " + TIE_NOTE_STATE)
    return plt.gcf()


def finding_q_2_5(df, df_overall, params, pivot_rank, rank_diff):
    new, counts = _new_entrants(df_overall)
    top = ', '.join(f'{s} ({int(v)})' for s, v in counts.head(4).items())
    return (f"{len(new)} institutions in the 2025 Overall list did not appear in 2017 "
            f"({int(new['Rank_Num'].notna().sum())} of them hold an exact Top-100 rank). Most came from {top}.")


# ================================================================================================
# 3. Univariate Analysis
# ================================================================================================
def _scores(df_overall, year=LATEST):
    return df_overall[df_overall['Year'] == year].dropna(subset=['Score'])


def plot_q_3_1(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    s = _scores(df_overall)['Score']
    plt.figure(figsize=(8, 3.8))
    sns.histplot(s, kde=True, bins=15, color='#3470a3', alpha=0.6)
    plt.axvline(s.mean(), color='#d32f2f', linestyle='--', linewidth=1.8, label=f'Mean ({s.mean():.1f})')
    plt.axvline(s.median(), color='#2e7d32', linestyle=':', linewidth=2, label=f'Median ({s.median():.1f})')
    plt.title('Distribution of NIRF Overall Scores: Top-100 Institutions (2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Score (pts)')
    plt.ylabel('Number of institutions')
    plt.legend(frameon=False)
    sns.despine()
    footnote('NIRF publishes scores only for the Top-100 ranked institutions, so ranks 101–200 are not included.')
    return plt.gcf()


def finding_q_3_1(df, df_overall, params, pivot_rank, rank_diff):
    s = _scores(df_overall)['Score']
    q1, q3 = s.quantile([0.25, 0.75])
    return (f"Across the {len(s)} Top-100 institutions in 2025, scores are right-skewed (skewness {s.skew():.2f}). Half "
            f"fall between {q1:.1f} and {q3:.1f}, the mean ({s.mean():.1f}) sits above the median ({s.median():.1f}), and "
            f"only {int((s >= 75).sum())} institutions score 75 or more (max {s.max():.1f}).")


def _best_tier(df_overall):
    d = df_overall.assign(Tier=df_overall['Rank_Band'].map(tier_of)).dropna(subset=['Tier'])
    d['TierIdx'] = d['Tier'].map(TIERS.index)
    best = d.groupby('Name')['TierIdx'].min().map(lambda i: TIERS[i])
    return best.value_counts().reindex(TIERS, fill_value=0), d['Name'].nunique()


def plot_q_3_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    counts, total = _best_tier(df_overall)
    plt.figure(figsize=(8.5, 3.9))
    bars = plt.bar(counts.index, counts.values, color=sns.color_palette('Blues_r', n_colors=len(counts)), width=0.6)
    plt.title(f'Best Overall Rank Tier Ever Reached: {total} Institutions (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Best rank tier reached in any year')
    plt.gca().yaxis.set_visible(False)
    sns.despine(left=True)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + total * 0.006, f'{int(h)}\n({h / total * 100:.1f}%)', ha='center',
                 fontweight='bold', fontsize=9)
    plt.ylim(0, counts.max() * 1.25)
    footnote('Each institution is counted once, in the best tier it reached in any of the nine editions.')
    return plt.gcf()


def finding_q_3_2(df, df_overall, params, pivot_rank, rank_diff):
    counts, total = _best_tier(df_overall)
    top100 = counts.iloc[:4].sum()
    return (f"{total} different institutions appeared in the Overall list between 2017 and 2025. Only {int(counts.iloc[0])} "
            f"({counts.iloc[0] / total * 100:.1f}%) ever reached the Top 10 and {int(top100)} ({top100 / total * 100:.0f}%) "
            f"ever held an exact Top-100 rank. The other {int(total - top100)} never got past the 101–200 bands.")


def _stats(df_overall):
    d = _scores(df_overall)
    hi, lo = d.loc[d['Score'].idxmax()], d.loc[d['Score'].idxmin()]
    return {'Maximum': d['Score'].max(), 'Mean': d['Score'].mean(), 'Median': d['Score'].median(),
            'Minimum': d['Score'].min()}, hi, lo, len(d)


def plot_q_3_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    stats, hi, lo, n = _stats(df_overall)
    plt.figure(figsize=(7.5, 3.2))
    bars = plt.barh(list(stats.keys()), list(stats.values()), color=sns.color_palette('Blues_r', n_colors=4), height=0.45)
    plt.title('Key Score Statistics for 2025 (Top-100 Institutions)', fontsize=12, fontweight='bold', pad=10)
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    for b, k in zip(bars, stats):
        extra = f'  ({short_name(hi["Name"], 30)})' if k == 'Maximum' else f'  ({short_name(lo["Name"], 30)})' if k == 'Minimum' else ''
        plt.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f'{b.get_width():.2f} pts{extra}', va='center',
                 fontweight='bold', color='#0d47a1', fontsize=9)
    plt.xlim(0, stats['Maximum'] * 1.55)
    return plt.gcf()


def finding_q_3_3(df, df_overall, params, pivot_rank, rank_diff):
    stats, hi, lo, n = _stats(df_overall)
    return (f"In 2025 ({n} scored institutions): maximum {stats['Maximum']:.2f} ({fname(hi['Name'])}), mean "
            f"{stats['Mean']:.2f}, median {stats['Median']:.2f}, minimum {stats['Minimum']:.2f} ({fname(lo['Name'])}, "
            f"rank #{int(lo['Rank_Num'])}). The mean is {stats['Mean'] - stats['Median']:.2f} pts above the median, "
            f"which points to a small group of very high scorers.")


def _quartile_bands(df_overall):
    s = _scores(df_overall)['Score']
    edges = np.linspace(s.min(), s.max(), 5)
    bands = pd.cut(s, edges, include_lowest=True)
    counts = bands.value_counts(sort=False)
    labels = [f'Q{i + 1}: {edges[i]:.1f}–{edges[i + 1]:.1f}' for i in range(4)]
    return pd.Series(counts.values, index=labels), len(s)


def plot_q_3_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    counts, n = _quartile_bands(df_overall)
    pct = counts / n * 100
    plt.figure(figsize=(8.5, 3.6))
    bars = plt.barh(counts.index, pct.values, color=['#c6dbef', '#9ecae1', '#6baed6', '#2171b5'], height=0.55)
    plt.title('Share of Ranked Institutions in Each Score Quartile (2025)', fontsize=12, fontweight='bold', pad=10)
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    for b, c in zip(bars, counts.values):
        plt.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f'{b.get_width():.0f}%  ({int(c)} institutions)',
                 va='center', fontweight='bold', color='#0d47a1')
    plt.xlim(0, pct.max() * 1.4)
    footnote('Quartiles split the 2025 score range (lowest to highest score) into four equal-width bands.')
    return plt.gcf()


def finding_q_3_4(df, df_overall, params, pivot_rank, rank_diff):
    counts, n = _quartile_bands(df_overall)
    pct = (counts / n * 100).round(0).astype(int)
    return (f"Dividing the 2025 score range into four equal quartiles: {pct.iloc[0]}% of ranked institutions sit in the "
            f"lowest quartile ({counts.index[0].split(': ')[1]} pts), {pct.iloc[1]}% in Q2, {pct.iloc[2]}% in Q3 and just "
            f"{pct.iloc[3]}% ({int(counts.iloc[3])} institutions) in the top quartile ({counts.index[3].split(': ')[1]} pts).")


MAIN_CATS = ['Overall', 'Engineering', 'Management', 'University', 'Pharmacy', 'Medical']


def plot_q_3_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d = df[(df['Year'] == LATEST) & (df['Category'].isin(MAIN_CATS))].dropna(subset=['Score'])
    plt.figure(figsize=(8.5, 4))
    sns.boxplot(data=d, x='Category', y='Score', order=MAIN_CATS, palette='Set2', width=0.5)
    plt.title('Score Distribution Spread Across NIRF Categories (2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('')
    plt.ylabel('Score (pts)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    return plt.gcf()


def finding_q_3_5(df, df_overall, params, pivot_rank, rank_diff):
    d = df[(df['Year'] == LATEST) & (df['Category'].isin(MAIN_CATS))].dropna(subset=['Score'])
    g = d.groupby('Category')['Score']
    iqr = g.quantile(0.75) - g.quantile(0.25)
    rng = g.max() - g.min()
    med, top = g.median(), g.max()
    if med.idxmax() == top.idxmax():
        lead = (f"{med.idxmax()} has both the highest median ({med.max():.1f}) and the highest single score "
                f"({top.max():.1f}).")
    else:
        lead = (f"{med.idxmax()} has the highest median ({med.max():.1f}) and {top.idxmax()} the highest single score "
                f"({top.max():.1f}).")
    return (f"In 2025, {iqr.idxmax()} has the widest spread (IQR {iqr.max():.1f} pts, range {rng[iqr.idxmax()]:.1f}) and "
            f"{iqr.idxmin()} the tightest (IQR {iqr.min():.1f}). " + lead)


# ================================================================================================
# 4. Bivariate Analysis and Correlation
# ================================================================================================
def _corr_box(r, label=''):
    plt.text(0.05, 0.90, f'Pearson r = {signed(r, ".2f")} ({r_strength(r)}){label}', transform=plt.gca().transAxes,
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc'))


def _scored_all(df_overall, cols):
    return df_overall.dropna(subset=cols)


def plot_q_4_1(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d = _scored_all(df_overall, ['RPC', 'Score'])
    plt.figure(figsize=(7.5, 4))
    sns.regplot(data=d, x='RPC', y='Score', color='#1f77b4', scatter_kws={'alpha': 0.5, 's': 30}, line_kws={'color': '#d32f2f', 'linewidth': 2})
    plt.title('Association: Research (RPC) vs Overall Score (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Research and Professional Practice (RPC)')
    plt.ylabel('Overall Score')
    sns.despine()
    _corr_box(d['Score'].corr(d['RPC']))
    footnote(f'{len(d)} institution-years with published scores (Top-100 of each edition).')
    return plt.gcf()


def finding_q_4_1(df, df_overall, params, pivot_rank, rank_diff):
    d = _scored_all(df_overall, ['RPC', 'Score'])
    r = d['Score'].corr(d['RPC'])
    return (f"RPC has a {r_strength(r)} positive correlation with the Overall score (r = {r:.2f}, R² = {r * r:.2f}) "
            f"across {len(d)} scored institution-years. Research alone explains about {r * r * 100:.0f}% of the variation in overall score.")


def plot_q_4_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d = _scored_all(df_overall, ['Perception', 'Score'])
    plt.figure(figsize=(7.5, 4))
    sns.regplot(data=d, x='Perception', y='Score', color='#ff7f0e', scatter_kws={'alpha': 0.5, 's': 30}, line_kws={'color': '#1f77b4', 'linewidth': 2})
    plt.title('Association: Perception (PR) vs Overall Score (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Perception (PR) Score')
    plt.ylabel('Overall Score')
    sns.despine()
    _corr_box(d['Score'].corr(d['Perception']))
    return plt.gcf()


def finding_q_4_2(df, df_overall, params, pivot_rank, rank_diff):
    d = _scored_all(df_overall, ['Perception', 'Score'])
    r = d['Score'].corr(d['Perception'])
    low = (d['Perception'] < 20).mean() * 100
    return (f"Perception has a {r_strength(r)} positive correlation with the Overall score (r = {r:.2f}). The relationship "
            f"is uneven: {low:.0f}% of scored institution-years have Perception below 20, and the top scorers have "
            f"Perception near 100.")


def _param_corr(df_overall):
    return pd.Series({p: df_overall['Score'].corr(df_overall[p]) for p in PARAMS}).sort_values(ascending=False)


def plot_q_4_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    corr = _param_corr(df_overall)
    plt.figure(figsize=(8, 3.5))
    bars = plt.barh(corr.index, corr.values, color=sns.color_palette('Blues_r', n_colors=len(corr)), height=0.45)
    plt.gca().invert_yaxis()
    plt.title('NIRF Parameter Correlation with Overall Score (2017 to 2025)', fontsize=12, fontweight='bold', pad=10)
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    for b in bars:
        w = b.get_width()
        plt.text(w + 0.02, b.get_y() + b.get_height() / 2, f'r = {signed(w, ".2f")} ({r_strength(w)})', va='center',
                 fontweight='bold', color='#0d47a1')
    plt.xlim(0, 1.3)
    return plt.gcf()


def finding_q_4_3(df, df_overall, params, pivot_rank, rank_diff):
    corr = _param_corr(df_overall)
    chain = ' > '.join(f'{p} ({r:.2f})' for p, r in corr.items())
    return (f"Correlation with the Overall score, strongest first: {chain}. {corr.index[0]} is the parameter that best "
            f"separates institutions. {corr.index[-1]} barely tracks the overall score.")


def plot_q_4_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d = _scored_all(df_overall, ['TLR', 'RPC'])
    plt.figure(figsize=(7.5, 4))
    sns.regplot(data=d, x='TLR', y='RPC', color='#2ca02c', scatter_kws={'alpha': 0.5, 's': 30}, line_kws={'color': '#d32f2f', 'linewidth': 2})
    plt.title('Teaching (TLR) vs Research (RPC) Relationship (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Teaching, Learning & Resources (TLR)')
    plt.ylabel('Research and Professional Practice (RPC)')
    sns.despine()
    _corr_box(d['TLR'].corr(d['RPC']))
    return plt.gcf()


def finding_q_4_4(df, df_overall, params, pivot_rank, rank_diff):
    d = _scored_all(df_overall, ['TLR', 'RPC'])
    r = d['TLR'].corr(d['RPC'])
    return (f"TLR and RPC show only a {r_strength(r)} positive correlation (r = {r:.2f}). Strong teaching resources do "
            f"not reliably mean strong research: many institutions score 60–75 on TLR with RPC anywhere from about "
            f"10 to 60.")


def _yearly_rpc_corr(df_overall):
    return df_overall.groupby('Year').apply(lambda g: g['Score'].corr(g['RPC']))


def plot_q_4_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    yc = _yearly_rpc_corr(df_overall)
    plt.figure(figsize=(8, 3.8))
    plt.plot(yc.index, yc.values, marker='o', color='#1f77b4', linewidth=2.5, markersize=7)
    plt.title('Research Correlation with Overall Score Over Time (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Year')
    plt.ylabel('Pearson r')
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    for y, r in zip(yc.index, yc.values):
        plt.text(y, r + 0.015, f'{r:.2f}', ha='center', fontweight='bold', fontsize=9, color='#1f77b4')
    return plt.gcf()


def finding_q_4_5(df, df_overall, params, pivot_rank, rank_diff):
    yc = _yearly_rpc_corr(df_overall)
    return (f"The RPC–Score correlation has stayed {r_strength(yc.min())} every year, between {yc.min():.2f} "
            f"({int(yc.idxmin())}) and {yc.max():.2f} ({int(yc.idxmax())}). It went from {yc.iloc[0]:.2f} in {FIRST} to "
            f"{yc.iloc[-1]:.2f} in {LATEST}, peaking in {int(yc.idxmax())} and easing slightly since.")


# ================================================================================================
# 5. Multivariate Analysis
# ================================================================================================
def _top5(df_overall):
    return top_n(df_overall[(df_overall['Year'] == LATEST) & (df_overall['Rank_Num'] <= 5)], 'Rank_Num', ascending=True,
                 df_overall=df_overall)


def plot_q_5_1(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    t5 = _top5(df_overall).assign(Label=lambda x: [f'#{int(r)} {short_name(n)}' for r, n in zip(x['Rank_Num'], x['Name'])])
    m = t5.melt(id_vars='Label', value_vars=PARAMS, var_name='Parameter', value_name='Score_Val')
    plt.figure(figsize=(9, 4.2))
    sns.barplot(data=m, x='Parameter', y='Score_Val', hue='Label', palette='tab10')
    plt.title('Top 5 Institutions Across All 5 Parameters (2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('')
    plt.ylabel('Parameter Score')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    sns.despine()
    return plt.gcf()


def finding_q_5_1(df, df_overall, params, pivot_rank, rank_diff):
    t5 = _top5(df_overall).set_index('Name')[PARAMS]
    spread = (t5.max() - t5.min()).sort_values()
    lead = {p: fname(t5[p].idxmax()) for p in PARAMS}
    return (f"2025 Top 5: {_names(t5.index)}. They are closest on {spread.index[0]} (gap {spread.iloc[0]:.1f} pts) and "
            f"furthest apart on {spread.index[-1]} (gap {spread.iloc[-1]:.1f} pts). Leaders by parameter: "
            + ', '.join(f'{p} – {n}' for p, n in lead.items()) + '.')


def _pct_gap(df_overall, hi, lo, filter_hi_median=False):
    d = _scores(df_overall).dropna(subset=[hi, lo]).copy()
    d[f'{hi}_pct'] = d[hi].rank(pct=True) * 100
    d[f'{lo}_pct'] = d[lo].rank(pct=True) * 100
    d['Gap'] = d[f'{hi}_pct'] - d[f'{lo}_pct']
    if filter_hi_median:
        d = d[d[hi] >= d[hi].median()]
    return top_n(d[d['Gap'] > 0], 'Gap', 6, df_overall=df_overall), d


def _pct_gap_plot(top, hi, lo, title, palette, color, note):
    plt.figure(figsize=(9.5, 3.9))
    bars = _barh(top['Name'], top['Gap'].values, palette, 6)
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    for b, row in zip(bars, top.itertuples()):
        lo_lbl = 'PR' if lo == 'Perception' else lo
        txt = (f"{signed(b.get_width(), '.0f')} pctl   ({hi} {getattr(row, hi):.1f} (P{getattr(row, hi + '_pct'):.0f}) | "
               f"{lo_lbl} {getattr(row, lo):.1f} (P{getattr(row, lo + '_pct'):.0f}))")
        plt.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, txt, va='center', fontweight='bold', color=color, fontsize=9)
    plt.xlim(0, top['Gap'].max() * 2.1)
    footnote(note + ' P = percentile within the 2025 Top-100. ' + TIE_NOTE)
    return plt.gcf()


def plot_q_5_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    top, _ = _pct_gap(df_overall, 'RPC', 'TLR')
    return _pct_gap_plot(top, 'RPC', 'TLR', 'High Research, Relatively Low Teaching (RPC vs TLR percentile, 2025)',
                         'Blues_r', '#0d47a1',
                         'Gap = RPC percentile − TLR percentile among the 2025 Top-100 (raw points shown in brackets).')


def finding_q_5_2(df, df_overall, params, pivot_rank, rank_diff):
    top, d = _pct_gap(df_overall, 'RPC', 'TLR')
    parts = [f"{fname(r.Name)} (RPC {r.RPC:.1f} vs TLR {r.TLR:.1f}, {signed(r.Gap, '.0f')} pctl)" for r in top.head(3).itertuples()]
    raw = int((d['RPC'] > d['TLR']).sum())
    return (f"Comparing each institution's standing on research with its standing on teaching, the largest gaps are "
            f"{'; '.join(parts)}. In raw points only {raw} institutions score higher on RPC than TLR, "
            f"because RPC is scored much lower on average, so the relative (percentile) comparison is the fair one.")


def plot_q_5_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    top, _ = _pct_gap(df_overall, 'TLR', 'Perception', filter_hi_median=True)
    return _pct_gap_plot(top, 'TLR', 'Perception', 'High Teaching, Relatively Low Perception (2025)',
                         'Oranges_r', '#e65100',
                         'Only institutions with TLR at or above the 2025 median. Gap = TLR percentile − Perception percentile.')


def finding_q_5_3(df, df_overall, params, pivot_rank, rank_diff):
    top, d = _pct_gap(df_overall, 'TLR', 'Perception', filter_hi_median=True)
    parts = [f"{fname(r.Name)} (TLR {r.TLR:.1f}, PR {r.Perception:.1f})" for r in top.head(3).itertuples()]
    return (f"Among institutions with above-median teaching scores (TLR ≥ {d['TLR'].min():.1f}), those whose perception "
            f"lags their teaching the most are {'; '.join(parts)}. They rank in the upper half for TLR but much lower "
            f"for perception.")


def _paradox(df_overall):
    cols = ['Score'] + PARAMS
    p24 = df_overall[df_overall['Year'] == LATEST - 1].set_index('Name')[cols]
    p25 = df_overall[df_overall['Year'] == LATEST].set_index('Name')[cols]
    diff = (p25 - p24).dropna()
    cond = (diff['Score'] > 0) & (diff[PARAMS] < 0).any(axis=1)
    sel = top_series(diff.loc[cond, 'Score'], 6, df_overall=df_overall)
    return diff.loc[sel.index], int(cond.sum())


def plot_q_5_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d, n = _paradox(df_overall)
    d = d.rename(columns={'Score': 'Overall\nScore', 'Perception': 'PR'})
    d.index = [short_name(x, 48) for x in d.index]
    annot = d.map(lambda v: signed(v, '.2f'))
    plt.figure(figsize=(9.5, 4))
    lim = np.nanmax(np.abs(d.values))
    ax = sns.heatmap(d, annot=annot, fmt='', cmap='RdYlGn', center=0, vmin=-lim, vmax=lim, linewidths=1, linecolor='white',
                     cbar_kws={'label': 'Change 2024 → 2025 (pts)', 'shrink': 0.8}, annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    ax.axvline(1, color='#1d1d1f', linewidth=2)
    ax.xaxis.tick_top()
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.ylabel('')
    plt.title('Score Gain Despite a Decline in ≥1 Parameter (2024 → 2025)', fontsize=12, fontweight='bold', pad=12)
    footnote(f'Top 6 of {n} such institutions by overall gain. Red cells = parameters that declined, green = gained. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_5_4(df, df_overall, params, pivot_rank, rank_diff):
    d, n = _paradox(df_overall)
    parts = []
    for name, row in d.head(3).iterrows():
        down = [f'{p} {signed(row[p], ".1f")}' for p in PARAMS if row[p] < 0]
        up = [f'{p} {signed(row[p], ".1f")}' for p in PARAMS if row[p] > 0]
        parts.append(f"{fname(name)} ({signed(row['Score'], '.2f')} overall; fell: {', '.join(down)}; rose: {', '.join(up)})")
    return (f"{n} institutions raised their overall score from 2024 to 2025 while declining in at least one parameter. "
            f"The largest gains: " + '; '.join(parts) + '.')


def _spread(df_overall):
    d = _scores(df_overall).dropna(subset=PARAMS).copy()
    d['Spread'] = d[PARAMS].max(axis=1) - d[PARAMS].min(axis=1)
    d['MaxP'] = d[PARAMS].idxmax(axis=1)
    d['MinP'] = d[PARAMS].idxmin(axis=1)
    return top_n(d, 'Spread', 6, df_overall=df_overall)


def plot_q_5_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    top = _spread(df_overall)
    plt.figure(figsize=(9.5, 3.8))
    bars = _barh(top['Name'], top['Spread'].values, 'Blues_r', 6)
    plt.title('Largest Gap Between Strongest and Weakest Parameter (2025)', fontsize=12, fontweight='bold', pad=10)
    for b, r in zip(bars, top.itertuples()):
        txt = f'{b.get_width():.1f} pts   ({r.MaxP} {getattr(r, r.MaxP):.1f} → {r.MinP} {getattr(r, r.MinP):.1f})'
        plt.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, txt, va='center', fontweight='bold', color='#0d47a1', fontsize=9)
    plt.xlim(0, top['Spread'].max() * 1.7)
    footnote('Strongest parameter → weakest parameter. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_5_5(df, df_overall, params, pivot_rank, rank_diff):
    top = _spread(df_overall)
    r = top.iloc[0]
    pairs = top.groupby(['MaxP', 'MinP']).size().sort_values(ascending=False)
    (mx, mn), k = pairs.index[0], pairs.iloc[0]
    return (f"{fname(r['Name'])} has the widest gap: {r['MaxP']} {r[r['MaxP']]:.1f} vs {r['MinP']} {r[r['MinP']]:.1f} "
            f"({r['Spread']:.1f} pts). For {k} of the top 6, the strongest parameter is {mx} and the weakest is {mn}.")


# ================================================================================================
# 6. Distribution and Statistical Comparison
# ================================================================================================
def _tiers_2025(df_overall):
    d = _scores(df_overall).copy()
    d['Tier'] = np.where(d['Rank_Num'] <= 10, 'Top 10', 'Ranks 11-100')
    return d


def plot_q_6_1(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d = _tiers_2025(df_overall)
    order = ['Top 10', 'Ranks 11-100']
    plt.figure(figsize=(7, 3.8))
    sns.boxplot(data=d, x='Tier', y='Score', order=order, palette=['#1f77b4', '#8c96a0'], width=0.45)
    plt.title('Score Distribution: Top 10 vs Ranks 11-100 (2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('')
    plt.ylabel('Score (pts)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    for i, t in enumerate(order):
        med = d[d['Tier'] == t]['Score'].median()
        plt.text(i + 0.26, med, f'Median: {med:.1f}', va='center', ha='left', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc'))
    plt.xlim(-0.5, 1.95)
    return plt.gcf()


def finding_q_6_1(df, df_overall, params, pivot_rank, rank_diff):
    d = _tiers_2025(df_overall)
    t, r = d[d['Tier'] == 'Top 10']['Score'], d[d['Tier'] != 'Top 10']['Score']
    return (f"In 2025 the Top 10 have a median score of {t.median():.1f} (range {t.min():.1f}–{t.max():.1f}), against "
            f"{r.median():.1f} for ranks 11–100 (range {r.min():.1f}–{r.max():.1f}), a gap of {t.median() - r.median():.1f} pts. "
            f"The Top 10 also vary more (std {t.std():.1f} vs {r.std():.1f}).")


def plot_q_6_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    sd = df_overall.groupby('Year')['Score'].std()
    plt.figure(figsize=(8, 3.9))
    plt.plot(sd.index, sd.values, marker='D', color='#1f77b4', linewidth=2.5, markersize=7)
    plt.title('Score Dispersion Over Time (Standard Deviation of Top-100 Scores)', fontsize=12, fontweight='bold', pad=16)
    plt.xlabel('Year')
    plt.ylabel('Standard Deviation (pts)')
    plt.xticks(range(FIRST, LATEST + 1))
    plt.ylim(sd.min() - 0.6, sd.max() + 0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    for y, s in zip(sd.index, sd.values):
        plt.annotate(f'{s:.2f}', (y, s), textcoords='offset points', xytext=(0, 9), ha='center', fontweight='bold',
                     color='#1f77b4', fontsize=9)
    return plt.gcf()


def finding_q_6_2(df, df_overall, params, pivot_rank, rank_diff):
    sd = df_overall.groupby('Year')['Score'].std()
    return (f"Yes, a little. The standard deviation of Top-100 scores rose from {sd.iloc[0]:.2f} ({FIRST}) to a peak of "
            f"{sd.max():.2f} ({int(sd.idxmax())}), then fell every year to {sd.iloc[-1]:.2f} ({LATEST}), the lowest of all "
            f"nine editions. Scores are now more tightly bunched.")


def plot_q_6_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    ps = _scores(df_overall)[PARAMS].std().sort_values(ascending=False)
    plt.figure(figsize=(8, 3.5))
    bars = plt.barh(ps.index, ps.values, color=sns.color_palette('Blues_r', n_colors=len(ps)), height=0.45)
    plt.gca().invert_yaxis()
    plt.title('NIRF Parameter Variability Across Institutions (Std Dev, 2025)', fontsize=12, fontweight='bold', pad=10)
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    for b in bars:
        plt.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f'σ = {b.get_width():.2f} pts', va='center',
                 fontweight='bold', color='#0d47a1')
    return plt.gcf()


def finding_q_6_3(df, df_overall, params, pivot_rank, rank_diff):
    ps = _scores(df_overall)[PARAMS].std().sort_values(ascending=False)
    return (f"In 2025, {ps.index[0]} varies most across institutions (σ = {ps.iloc[0]:.1f}), followed by "
            f"{ps.index[1]} ({ps.iloc[1]:.1f}). {ps.index[-1]} ({ps.iloc[-1]:.1f}) and {ps.index[-2]} ({ps.iloc[-2]:.1f}) "
            f"vary least, so they do little to separate ranks.")


def _iqr(df_overall):
    g = df_overall.groupby('Year')['Score']
    return g.quantile(0.25), g.median(), g.quantile(0.75)


def plot_q_6_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    q25, q50, q75 = _iqr(df_overall)
    plt.figure(figsize=(8, 3.8))
    plt.plot(q50.index, q50.values, marker='o', color='#1f77b4', linewidth=2.5, label='Median Score')
    plt.fill_between(q50.index, q25.values, q75.values, color='#1f77b4', alpha=0.15, label='IQR (25th to 75th %tile)')
    plt.title('Median and Interquartile Range of Scores Over Time (2017 to 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Year')
    plt.ylabel('Score (pts)')
    plt.xticks(range(FIRST, LATEST + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=False)
    sns.despine()
    return plt.gcf()


def finding_q_6_4(df, df_overall, params, pivot_rank, rank_diff):
    q25, q50, q75 = _iqr(df_overall)
    w = q75 - q25
    return (f"The median Top-100 score rose from {q50.iloc[0]:.1f} ({FIRST}) to {q50.iloc[-1]:.1f} ({LATEST}). The IQR "
            f"moved up with it (from {q25.iloc[0]:.1f}–{q75.iloc[0]:.1f} to {q25.iloc[-1]:.1f}–{q75.iloc[-1]:.1f}), "
            f"and its width went from {w.iloc[0]:.1f} to {w.iloc[-1]:.1f} pts (widest in {int(w.idxmax())}: {w.max():.1f}).")


def _state_scores(df_overall):
    d = _scores(df_overall)
    states = top_counts(d['State'].value_counts(), 6).index.tolist()
    d = d[d['State'].isin(states)]
    order = d.groupby('State')['Score'].median().loc[states].sort_values(ascending=False).index.tolist()
    return d, order


def plot_q_6_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d, order = _state_scores(df_overall)
    plt.figure(figsize=(9, 4))
    sns.boxplot(data=d, x='State', y='Score', order=order, palette='Blues_r', width=0.5)
    ax = plt.gca()
    ax.set_xticklabels([f"{s}\n(n = {(d['State'] == s).sum()})" for s in order])
    plt.title('Score Distribution Across the 6 States with Most Top-100 Institutions (2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('')
    plt.ylabel('Score (pts)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    footnote('Only Top-100 institutions have published scores. States are ordered by median score. ' + TIE_NOTE_STATE)
    return plt.gcf()


def finding_q_6_5(df, df_overall, params, pivot_rank, rank_diff):
    d, order = _state_scores(df_overall)
    med = d.groupby('State')['Score'].median()
    n = d['State'].value_counts()
    return (f"Among the six states with the most Top-100 institutions in 2025, {order[0]} has the highest median score "
            f"({med[order[0]]:.1f}, n = {n[order[0]]}) and {order[-1]} the lowest ({med[order[-1]]:.1f}, n = {n[order[-1]]}). "
            f"Tamil Nadu has by far the most institutions (n = {n.get('Tamil Nadu', 0)}), but its median is "
            f"{med.get('Tamil Nadu', float('nan')):.1f}.")


# ================================================================================================
# 7. Ranking Transitions and Institutional Mobility
# ================================================================================================
MOBILITY_FROM = 2023


def _mobility(df_overall, y0=MOBILITY_FROM, y1=LATEST):
    piv = df_overall.pivot_table(index='Name', columns='Year', values='Rank_Num')
    m = piv[(piv[y0] > 50) & (piv[y0] <= 100) & (piv[y1] <= 50)][[y0, y1]]
    gained = top_series(m[y0] - m[y1], df_overall=df_overall)
    return gained, m


def _check_years(y0, y1):
    if not (FIRST <= y0 < y1 <= LATEST):
        raise ValueError(f'Choose two years with {FIRST} ≤ from < to ≤ {LATEST} (got {y0} → {y1}).')


def plot_q_7_1(df, df_overall, params, pivot_rank, rank_diff, y0=MOBILITY_FROM, y1=LATEST):
    plt.close('all')
    _check_years(y0, y1)
    gained, m = _mobility(df_overall, y0, y1)
    plt.figure(figsize=(9, max(3.4, 0.5 * len(gained) + 1)))
    title = f'Institutions Moving from Ranks 51–100 into the Top 50 ({len(gained)}) [{y0} {ARROW} {y1}]'
    if gained.empty:
        plt.title(title, fontsize=12, fontweight='bold', pad=10)
        plt.text(0.5, 0.5, f'No institution moved from ranks 51–100 in {y0}\ninto the Top 50 in {y1}.',
                 ha='center', va='center', fontsize=11, color='#6e6e73', transform=plt.gca().transAxes)
        plt.axis('off')
        return plt.gcf()
    bars = _barh(gained.index, gained.values, 'Greens_r', len(gained))
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    for b, name in zip(bars, gained.index):
        plt.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, move_label(m.at[name, y0], m.at[name, y1]),
                 va='center', fontweight='bold', color='#1b5e20')
    plt.xlim(0, gained.max() * 1.6)
    covid = (f'{MOBILITY_FROM}–{LATEST} window avoids the COVID-affected 2020–2022 editions. '
             if (y0, y1) == (MOBILITY_FROM, LATEST) else '')
    footnote(covid + 'Bar length = ranks gained. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_7_1(df, df_overall, params, pivot_rank, rank_diff, y0=MOBILITY_FROM, y1=LATEST):
    _check_years(y0, y1)
    gained, m = _mobility(df_overall, y0, y1)
    if gained.empty:
        return f"Between {y0} and {y1}, no institution moved from the 51–100 band into the Top 50."
    parts = [f"{fname(n)} ({move_label(m.at[n, y0], m.at[n, y1], False)})" for n in gained.index]
    noun = 'institution' if len(gained) == 1 else 'institutions'
    return f"Between {y0} and {y1}, {len(gained)} {noun} moved from the 51–100 band into the Top 50: " + '; '.join(parts) + '.'


def _turnover(df_overall):
    piv = df_overall.pivot_table(index='Name', columns='Year', values='Rank_Num')
    rows = []
    for y in range(FIRST, LATEST):
        a = set(piv.index[piv[y] <= 10])
        b = set(piv.index[piv[y + 1] <= 10])
        rows.append((y, sorted(b - a, key=lambda n: piv.at[n, y + 1]), sorted(a - b, key=lambda n: piv.at[n, y])))
    return rows, piv


def _fmt_rank(v):
    return f'#{int(v)}' if pd.notna(v) else '101+'


def plot_q_7_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    rows, piv = _turnover(df_overall)
    cells = []
    for y, ins, outs in rows:
        cells.append([f'{y} → {y + 1}',
                      '\n'.join(f'{short_name(n, 34)} ({_fmt_rank(piv.at[n, y])}→#{int(piv.at[n, y + 1])})' for n in ins) or '—',
                      '\n'.join(f'{short_name(n, 34)} (#{int(piv.at[n, y])}→{_fmt_rank(piv.at[n, y + 1])})' for n in outs) or '—'])
    n_lines = sum(max(1, len(i), len(o)) for _, i, o in rows)
    fig, ax = plt.subplots(figsize=(9.5, 1.2 + 0.32 * n_lines))
    ax.axis('off')
    tbl = ax.table(cellText=cells, colLabels=['Years', 'Entered Top 10', 'Exited Top 10'], colWidths=[0.16, 0.42, 0.42],
                   cellLoc='left', loc='upper center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#e0e0e0')
        lines = max(1, len(rows[r - 1][1]), len(rows[r - 1][2])) if r > 0 else 1
        cell.set_height(0.95 / (n_lines + 1) * lines)
        if r == 0:
            cell.set_facecolor('#f0f4fa')
            cell.set_text_props(fontweight='bold')
        elif c == 1 and cells[r - 1][1] != '—':
            cell.set_text_props(color='#1b5e20', fontweight='bold')
        elif c == 2 and cells[r - 1][2] != '—':
            cell.set_text_props(color='#b71c1c', fontweight='bold')
    ax.set_title('Top 10 Entries and Exits Between Consecutive Years', fontsize=12, fontweight='bold', pad=8)
    return fig


def finding_q_7_2(df, df_overall, params, pivot_rank, rank_diff):
    rows, piv = _turnover(df_overall)
    changes = [(y, i, o) for y, i, o in rows if i or o]
    quiet = len(rows) - len(changes)
    parts = [f"{y}→{y + 1}: {_names(i)} in, {_names(o)} out" for y, i, o in changes]
    return (f"The Top 10 is very stable: {quiet} of {len(rows)} year-to-year transitions had no change at all. The swaps were "
            + '; '.join(parts) + '.')


def _band_retention(df_overall):
    d = df_overall.assign(Tier=df_overall['Rank_Band'].map(tier_of)).dropna(subset=['Tier'])
    out = {}
    for y in range(FIRST, LATEST):
        m = d[d['Year'] == y][['Name', 'Tier']].merge(d[d['Year'] == y + 1][['Name', 'Tier']], on='Name')
        out[f'{y}–{str(y + 1)[2:]}'] = ((m['Tier_x'] == m['Tier_y']).mean() * 100, len(m))
    return out


def plot_q_7_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    ret = _band_retention(df_overall)
    years, rates = list(ret), [v[0] for v in ret.values()]
    plt.figure(figsize=(8.5, 3.9))
    plt.plot(years, rates, marker='o', color='#1f77b4', linewidth=2.5, markersize=7)
    plt.title('Institutions Staying in the Same Rank Band Year-on-Year (Overall)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Consecutive Years')
    plt.ylabel('Retention Rate (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    for y, (r, n) in ret.items():
        plt.annotate(f'{r:.0f}%\n(n={n})', (y, r), textcoords='offset points', xytext=(0, 8), ha='center',
                     fontweight='bold', fontsize=8.5, color='#1f77b4')
    footnote('Bands: 1–10, 11–25, 26–50, 51–100, 101–150, 151–200. n = institutions listed in both years.')
    return plt.gcf()


def finding_q_7_3(df, df_overall, params, pivot_rank, rank_diff):
    ret = _band_retention(df_overall)
    rates = pd.Series({k: v[0] for k, v in ret.items()})
    return (f"Of the institutions listed in both of two consecutive years, on average {rates.mean():.0f}% stay in the same "
            f"rank band (range {rates.min():.0f}% in {rates.idxmin()} to {rates.max():.0f}% in {rates.idxmax()}). "
            f"Latest: {rates.iloc[-1]:.0f}% for {rates.index[-1]}.")


def _extreme_moves(df_overall, rank_diff):
    up = top_series(rank_diff, 3, df_overall=df_overall)
    down = top_series(rank_diff, 3, ascending=True, df_overall=df_overall)
    piv = df_overall.pivot_table(index='Name', columns='Year', values='Rank_Num')
    return pd.concat([up, down]), piv


def plot_q_7_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    moves, piv = _extreme_moves(df_overall, rank_diff)
    plt.figure(figsize=(9.5, 4))
    bars = plt.barh([short_name(n, 44) for n in moves.index], moves.values,
                    color=['#2e7d32' if x > 0 else '#d32f2f' for x in moves.values], height=0.45)
    plt.gca().invert_yaxis()
    plt.axvline(0, color='#888888', linewidth=1)
    plt.title('Largest Upward (Green) & Downward (Red) Rank Moves (2017 → 2025)', fontsize=12, fontweight='bold', pad=12)
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    lim = moves.abs().max()
    plt.xlim(-lim * 2.0, lim * 2.0)
    for b, name in zip(bars, moves.index):
        w = b.get_width()
        plt.text(w + (1.5 if w > 0 else -1.5), b.get_y() + b.get_height() / 2, move_label(piv.at[name, FIRST], piv.at[name, LATEST]),
                 va='center', ha='left' if w > 0 else 'right', fontweight='bold', color='#1b5e20' if w > 0 else '#b71c1c')
    footnote('Institutions holding an exact Top-100 rank in both 2017 and 2025. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_7_4(df, df_overall, params, pivot_rank, rank_diff):
    moves, piv = _extreme_moves(df_overall, rank_diff)
    f = lambda n: f"{fname(n)} ({move_label(piv.at[n, FIRST], piv.at[n, LATEST], False)})"
    return (f"Biggest rise: {f(moves.index[0])}; next {f(moves.index[1])} and {f(moves.index[2])}. "
            f"Biggest fall: {f(moves.index[3])}; next {f(moves.index[4])} and {f(moves.index[5])}.")


def _directions(df_overall):
    piv = df_overall.pivot_table(index='Name', columns='Year', values='Rank_Num')
    out = {}
    for y in range(FIRST, LATEST):
        r = piv[[y, y + 1]].dropna()
        out[f'{y}–{str(y + 1)[2:]}'] = {'Moved up': int((r[y + 1] < r[y]).sum()),
                                             'Moved down': int((r[y + 1] > r[y]).sum()),
                                             'Same rank': int((r[y + 1] == r[y]).sum())}
    return pd.DataFrame(out).T


def plot_q_7_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d = _directions(df_overall)
    plt.figure(figsize=(9, 4))
    bottom = np.zeros(len(d))
    for col, c in zip(d.columns, ['#2e7d32', '#d32f2f', '#9e9e9e']):
        bars = plt.bar(d.index, d[col], bottom=bottom, color=c, width=0.6, label=col)
        for b, v, bt in zip(bars, d[col], bottom):
            if v >= 4:
                plt.text(b.get_x() + b.get_width() / 2, bt + v / 2, str(v), ha='center', va='center', color='white',
                         fontweight='bold', fontsize=9)
        bottom += d[col].values
    plt.title('Rank Movement Direction Each Year (Institutions Ranked 1–100 in Both Years)', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Institutions')
    plt.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    sns.despine()
    return plt.gcf()


def finding_q_7_5(df, df_overall, params, pivot_rank, rank_diff):
    d = _directions(df_overall)
    last = d.iloc[-1]
    total = d.sum(axis=1)
    return (f"{d.index[-1]}: {last['Moved up']} institutions moved up, {last['Moved down']} moved down and "
            f"{last['Same rank']} kept the same rank (of {int(total.iloc[-1])} ranked in both years). Across all years only "
            f"{d['Same rank'].sum() / total.sum() * 100:.0f}% keep their exact rank, so most institutions move every year.")


# ================================================================================================
# 8. Outlier Detection and Performance Gaps
# ================================================================================================
def _residuals(df_overall):
    d = _scores(df_overall).dropna(subset=['RPC']).copy()
    m, b = np.polyfit(d['Score'], d['RPC'], 1)
    d['Expected'] = m * d['Score'] + b
    d['Residual'] = d['RPC'] - d['Expected']
    return d, top_n(d, 'Residual', 5, df_overall=df_overall), (m, b)


def plot_q_8_1(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    d, out, (m, b) = _residuals(df_overall)
    plt.figure(figsize=(9, 4.6))
    sns.scatterplot(data=d, x='Score', y='RPC', alpha=0.35, color='#8c96a0', s=30)
    xs = np.linspace(d['Score'].min(), d['Score'].max(), 50)
    plt.plot(xs, m * xs + b, color='#444444', linestyle='--', linewidth=1.5, label='Expected RPC for that score')
    sns.scatterplot(data=out, x='Score', y='RPC', color='#d32f2f', s=70, label='Research outliers')
    # Stack the labels in the empty lower-right area, highest RPC first, with leader lines leaving the
    # label's left edge, so lines fan out without crossing.
    span = d['RPC'].max() - d['RPC'].min()
    y_top, y_step = out['RPC'].max() - 0.05 * span, span * 0.085
    x_text = d['Score'].min() + 0.5 * (d['Score'].max() - d['Score'].min())
    for i, (_, row) in enumerate(out.sort_values('RPC', ascending=False).iterrows()):
        plt.annotate(f"{short_name(row['Name'], 44)} (+{row['Residual']:.1f})", (row['Score'], row['RPC']),
                     xytext=(x_text, y_top - i * y_step), textcoords='data', fontsize=8.5, fontweight='bold', color='#b71c1c',
                     va='center', arrowprops=dict(arrowstyle='-', color='#b71c1c', lw=0.7, relpos=(0, 0.5)),
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))
    plt.title('Outliers: Research Score Far Above What the Overall Score Predicts (2025)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Overall Score')
    plt.ylabel('Research (RPC) Score')
    plt.legend(frameon=False, loc='upper left')
    sns.despine()
    footnote('Numbers in brackets = RPC points above the regression line. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_8_1(df, df_overall, params, pivot_rank, rank_diff):
    d, out, _ = _residuals(df_overall)
    parts = [f"{fname(r.Name)} (RPC {r.RPC:.1f} vs expected {r.Expected:.1f}, rank #{int(r.Rank_Num)})" for r in out.itertuples()]
    return ("Relative to the 2025 regression of RPC on overall score, the biggest positive research outliers are "
            + '; '.join(parts) + '. Their research output is stronger than their overall rank suggests.')


def _rpc_tlr_gap(df_overall):
    d = _scores(df_overall).dropna(subset=['RPC', 'TLR']).copy()
    d['Gap'] = d['RPC'] - d['TLR']
    return top_n(d[d['Gap'] > 0], 'Gap', 6, df_overall=df_overall), len(d)


def plot_q_8_2(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    top, n = _rpc_tlr_gap(df_overall)
    plt.figure(figsize=(9, max(2.6, 0.55 * len(top) + 1.3)))
    bars = _barh(top['Name'], top['Gap'].values, 'Blues_r', len(top))
    plt.title('Institutions Scoring Higher in Research than Teaching (2025)', fontsize=12, fontweight='bold', pad=10)
    for b, r in zip(bars, top.itertuples()):
        plt.text(b.get_width() + top['Gap'].max() * 0.03, b.get_y() + b.get_height() / 2,
                 f'{signed(b.get_width())} pts   (RPC {r.RPC:.1f} vs TLR {r.TLR:.1f}, rank #{int(r.Rank_Num)})',
                 va='center', fontweight='bold', color='#0d47a1', fontsize=9)
    plt.xlim(0, top['Gap'].max() * 2.6)
    footnote(f'Only institutions with RPC > TLR in raw points ({len(top)} of {n}). ' + TIE_NOTE)
    return plt.gcf()


def finding_q_8_2(df, df_overall, params, pivot_rank, rank_diff):
    top, n = _rpc_tlr_gap(df_overall)
    parts = [f"{fname(r.Name)} ({signed(r.Gap)} pts)" for r in top.itertuples()]
    return (f"Only {len(top)} of the {n} scored institutions in 2025 have a higher research score than teaching score: "
            + ', '.join(parts) + '. All are long-established, research-intensive national institutions.')


def _yoy_change(df_overall):
    sp = df_overall.pivot_table(index='Name', columns='Year', values='Score')
    ch = (sp[LATEST] - sp[LATEST - 1]).dropna()
    order = top_series(ch.abs(), 6, df_overall=df_overall).index
    return ch.loc[order], sp


def plot_q_8_3(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    ch, sp = _yoy_change(df_overall)
    plt.figure(figsize=(9.5, 3.9))
    bars = plt.barh([short_name(n, 44) for n in ch.index], ch.values,
                    color=['#2e7d32' if v > 0 else '#d32f2f' for v in ch.values], height=0.45)
    plt.gca().invert_yaxis()
    plt.axvline(0, color='#888888', linewidth=1)
    plt.title('Largest Year-to-Year Score Changes (2024 → 2025)', fontsize=12, fontweight='bold', pad=10)
    plt.gca().xaxis.set_visible(False)
    sns.despine(left=True, bottom=True)
    lim = ch.abs().max()
    plt.xlim(-lim * 2.3, lim * 2.3)
    for b, n in zip(bars, ch.index):
        w = b.get_width()
        plt.text(w + (0.2 if w > 0 else -0.2), b.get_y() + b.get_height() / 2,
                 f'{signed(w, ".2f")} pts  ({sp.at[n, LATEST - 1]:.1f}→{sp.at[n, LATEST]:.1f})',
                 va='center', ha='left' if w > 0 else 'right', fontweight='bold', color='#1b5e20' if w > 0 else '#b71c1c', fontsize=9)
    footnote('Ranked by size of change (up or down). ' + TIE_NOTE)
    return plt.gcf()


def finding_q_8_3(df, df_overall, params, pivot_rank, rank_diff):
    ch, sp = _yoy_change(df_overall)
    all_ch = (sp[LATEST] - sp[LATEST - 1]).dropna()
    parts = [f"{fname(n)} ({signed(v, '.2f')})" for n, v in ch.head(3).items()]
    return (f"The largest score swings from 2024 to 2025 were {', '.join(parts)}. Typical change was much smaller: the "
            f"median absolute change across {len(all_ch)} institutions was {all_ch.abs().median():.2f} pts.")


def _strong_rpc_low_rank(df_overall):
    d = _scores(df_overall).dropna(subset=['RPC']).copy()
    q = d['RPC'].quantile(0.75)
    sel = d[(d['RPC'] >= q) & (d['Rank_Num'] > 30)]
    return top_n(sel, 'Rank_Num', 6, df_overall=df_overall), q


def plot_q_8_4(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    top, q = _strong_rpc_low_rank(df_overall)
    plt.figure(figsize=(9, max(3, 0.55 * len(top) + 1.3)))
    bars = _barh(top['Name'], top['RPC'].values, 'Blues_r', len(top))
    plt.title(f'Top-Quartile Research (RPC ≥ {q:.1f}) but Overall Rank Below 30 (2025)', fontsize=12, fontweight='bold', pad=10)
    for b, r in zip(bars, top.itertuples()):
        plt.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f'RPC {r.RPC:.1f}  (Rank #{int(r.Rank_Num)})',
                 va='center', fontweight='bold', color='#0d47a1')
    plt.xlim(0, top['RPC'].max() * 1.45)
    footnote('Worst-ranked first. ' + TIE_NOTE)
    return plt.gcf()


def finding_q_8_4(df, df_overall, params, pivot_rank, rank_diff):
    top, q = _strong_rpc_low_rank(df_overall)
    parts = [f"{fname(r.Name)} (RPC {r.RPC:.1f}, #{int(r.Rank_Num)})" for r in top.itertuples()]
    weakest = top[PARAMS].sub(top[PARAMS].mean().mean()).mean().idxmin()
    return (f"{len(top)} institutions have top-quartile research (RPC ≥ {q:.1f}) yet rank below #30 in 2025: "
            + '; '.join(parts) + f". Their weakest parameter on average is {weakest}.")


def _volatility(df_overall):
    rs = df_overall.groupby('Name')['Rank_Num'].agg(['count', 'std', 'min', 'max']).dropna()
    rs = rs[rs['count'] >= 5]
    order = top_series(rs['std'], 6, df_overall=df_overall).index
    return rs.loc[order], rs


def plot_q_8_5(df, df_overall, params, pivot_rank, rank_diff):
    plt.close('all')
    top, rs = _volatility(df_overall)
    plt.figure(figsize=(9.5, 3.8))
    bars = _barh(top.index, top['std'].values, 'Blues_r', 6)
    plt.title('Highest Ranking Volatility Across Years (Std Dev of Rank)', fontsize=12, fontweight='bold', pad=10)
    for b, r in zip(bars, top.itertuples()):
        plt.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                 f'±{r.std:.1f} ranks  (best #{int(r.min)}, worst #{int(r.max)}, {int(r.count)} yrs)',
                 va='center', fontweight='bold', color='#0d47a1', fontsize=9)
    plt.xlim(0, top['std'].max() * 1.9)
    footnote(f'Institutions with an exact Top-100 rank in at least 5 editions (n = {len(rs)}). ' + TIE_NOTE)
    return plt.gcf()


def finding_q_8_5(df, df_overall, params, pivot_rank, rank_diff):
    top, rs = _volatility(df_overall)
    parts = [f"{fname(n)} (±{r['std']:.1f}, #{int(r['min'])}–#{int(r['max'])})" for n, r in top.head(3).iterrows()]
    return (f"Compared with the typical institution ranked in 5+ editions (median rank std ±{rs['std'].median():.1f}), "
            f"the most volatile are " + '; '.join(parts) + '.')


# ================================================================================================
# Catalog
# ================================================================================================
NOTES = {
    "1.4": ("NIRF publishes scores only for the Top-100 of each year, and that group changes from year to year. "
            "We therefore read the question as 'the mean score of each year's Top-100'. To check whether the change "
            "is real and not just a different set of institutions, we also plot a fixed cohort: the institutions that "
            "were in the Top-100 in all nine editions."),
    "2.3": ("'Over time' is measured across the full period: for each state we compare its number of Top-100 "
            "institutions in the first edition (2017) with the latest (2025). Bars show the net change, and the "
            "counts in brackets are 2017 → 2025. The year-by-year trend for the leading states is in Q2.2."),
    "3.2": ("Each year has exactly one institution per rank 1–100 plus 50 in each of the 101–150 and 151–200 "
            "bands, so a single year's rank distribution is the same every time. Instead we take every institution "
            "that appeared between 2017 and 2025 and count it in the best tier it ever reached. This shows how many "
            "institutions reach each level."),
    "3.4": ("Standard quartiles always hold 25% of institutions each, so a chart of them says nothing. We split the "
            "2025 score range (lowest to highest score) into four equal-width bands and show the share of "
            "institutions in each, which shows how scores are bunched."),
    "5.2": ("RPC averages about 36 pts and TLR about 64 pts, so in raw points almost every institution looks "
            "'better at teaching'. 'Relatively low teaching' is measured as the gap between an institution's "
            "percentile rank in RPC and its percentile rank in TLR (2025 Top-100). Raw-point gaps are covered in Q8.2."),
    "5.3": ("Perception is scored far lower than TLR (2025 Top-100 median about 18 vs about 70, and a quarter of "
            "institutions score under 10), so raw TLR − Perception just lists the lowest-perception institutions. We keep institutions with high teaching scores (TLR at or above the 2025 median) and rank them by "
            "TLR percentile − Perception percentile."),
    "7.3": ("Rank bands are 1–10, 11–25, 26–50, 51–100 (exact ranks grouped) and NIRF's own 101–150 and "
            "151–200 bands. For each pair of consecutive years we take institutions listed in both years and count "
            "how many stayed in the same band."),
}

_TITLES = [
    ("1.1", 1, "1. Time-Series and Trend Analysis", "How has the overall NIRF ranking of a selected institution changed from 2017 to 2025?"),
    ("1.2", 1, "1. Time-Series and Trend Analysis", "Which institutions have shown the greatest improvement in rank over the nine years?"),
    ("1.3", 1, "1. Time-Series and Trend Analysis", "Which institutions have experienced the largest decline in ranking?"),
    ("1.4", 1, "1. Time-Series and Trend Analysis", "How has the average score of institutions changed over time?"),
    ("1.5", 1, "1. Time-Series and Trend Analysis", "Which institutions have remained consistently in the top 10 across multiple years?"),
    ("2.1", 2, "2. Geospatial and Regional Analysis", "Which states have the highest number of institutions in the top 100?"),
    ("2.2", 2, "2. Geospatial and Regional Analysis", "How is the geographical distribution of top-ranked institutions changing from 2017 to 2025?"),
    ("2.3", 2, "2. Geospatial and Regional Analysis", "Which states have improved their representation in the top 100 over time?"),
    ("2.4", 2, "2. Geospatial and Regional Analysis", "How do the rankings of institutions in Karnataka compare with those in Tamil Nadu?"),
    ("2.5", 2, "2. Geospatial and Regional Analysis", "Which states have produced the largest number of newly appearing institutions in the rankings?"),
    ("3.1", 3, "3. Univariate Analysis", "What is the distribution of NIRF scores for a selected year?"),
    ("3.2", 3, "3. Univariate Analysis", "What is the distribution of overall ranks across all institutions?"),
    ("3.3", 3, "3. Univariate Analysis", "What is the average, median, minimum, and maximum score in a given year?"),
    ("3.4", 3, "3. Univariate Analysis", "What percentage of ranked institutions fall into each score quartile?"),
    ("3.5", 3, "3. Univariate Analysis", "How does the spread of scores differ across NIRF categories?"),
    ("4.1", 4, "4. Bivariate Analysis and Correlation", "How strongly are Research and Professional Practice (RPC) scores associated with overall scores?"),
    ("4.2", 4, "4. Bivariate Analysis and Correlation", "How does Perception (PR) relate to overall scores?"),
    ("4.3", 4, "4. Bivariate Analysis and Correlation", "Which NIRF parameter has the strongest correlation with the overall score?"),
    ("4.4", 4, "4. Bivariate Analysis and Correlation", "How are TLR and RPC scores related across institutions?"),
    ("4.5", 4, "4. Bivariate Analysis and Correlation", "Has the correlation between research scores and overall scores changed from 2017 to 2025?"),
    ("5.1", 5, "5. Multivariate Analysis", "How do the top five institutions compare across all five NIRF parameters?"),
    ("5.2", 5, "5. Multivariate Analysis", "Which institutions have high research scores but relatively low teaching scores?"),
    ("5.3", 5, "5. Multivariate Analysis", "Which institutions have high teaching scores but relatively low perception scores?"),
    ("5.4", 5, "5. Multivariate Analysis", "Which institutions have improved their overall scores despite declining scores in one or more parameters?"),
    ("5.5", 5, "5. Multivariate Analysis", "Which institutions have the largest difference between their strongest and weakest parameters?"),
    ("6.1", 6, "6. Distribution and Statistical Comparison", "How does the score distribution differ between the top 10 and the remaining institutions?"),
    ("6.2", 6, "6. Distribution and Statistical Comparison", "Are NIRF scores more dispersed in one year than another?"),
    ("6.3", 6, "6. Distribution and Statistical Comparison", "Which NIRF parameter exhibits the greatest variability across institutions?"),
    ("6.4", 6, "6. Distribution and Statistical Comparison", "How do the median and interquartile range of scores change over time?"),
    ("6.5", 6, "6. Distribution and Statistical Comparison", "How does the score distribution differ across states?"),
    ("7.1", 7, "7. Ranking Transitions and Institutional Mobility", "How many institutions moved from the 51–100 band into the top 50 between two selected years?"),
    ("7.2", 7, "7. Ranking Transitions and Institutional Mobility", "Which institutions entered or exited the top 10 between consecutive years?"),
    ("7.3", 7, "7. Ranking Transitions and Institutional Mobility", "What percentage of institutions remained in the same rank band from one year to the next?"),
    ("7.4", 7, "7. Ranking Transitions and Institutional Mobility", "Which institutions experienced the largest upward and downward rank movements?"),
    ("7.5", 7, "7. Ranking Transitions and Institutional Mobility", "How many institutions moved up, moved down, or retained their positions each year?"),
    ("8.1", 8, "8. Outlier Detection and Performance Gaps", "Which institutions have unusually high research scores relative to their overall scores?"),
    ("8.2", 8, "8. Outlier Detection and Performance Gaps", "Which institutions perform substantially better in research than in teaching?"),
    ("8.3", 8, "8. Outlier Detection and Performance Gaps", "Which institutions experienced an unusually large year-to-year score change?"),
    ("8.4", 8, "8. Outlier Detection and Performance Gaps", "Which institutions have a strong parameter score but a relatively poor overall rank?"),
    ("8.5", 8, "8. Outlier Detection and Performance Gaps", "Which institutions have unusually high ranking volatility compared with similar institutions?"),
]

QUESTIONS_CATALOG = {
    qid: {
        "id": qid,
        "category_id": cid,
        "category_name": cname,
        "title": title,
        "note": NOTES.get(qid),
        "plot_fn": globals()[f"plot_q_{qid.replace('.', '_')}"],
        "finding_fn": globals()[f"finding_q_{qid.replace('.', '_')}"],
    }
    for qid, cid, cname, title in _TITLES
}
