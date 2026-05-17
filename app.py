"""
app.py — Nifty Screener Streamlit App
Reads live data from Google Sheet, applies multi-indicator filters with AND/OR logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
import gspread
from google.oauth2.service_account import Credentials
import gspread_dataframe as gd
from datetime import datetime
import threading

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Live Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SHEET_ID = "1JWHOhfTFhS0345GC4KMGHYCa1F8YEdDk2Skb85R2p5U"
SCOPES   = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Flat indicator lookup: label → (col, vmin, vmax, vdefault)
# vmin=None means text filter (BUY/SELL)
ALL_INDICATORS = {
    "Open":               ("Open",               0,    10000, 100),
    "High":               ("High",               0,    10000, 100),
    "Low":                ("Low",                0,    10000, 100),
    "Close":              ("Close",              0,    10000, 100),
    "Prev Close":         ("Prev_Close",         0,    10000, 100),
    "Volume":             ("Volume",             0,    1e8,   500000),
    "Supertrend Signal":  ("Supertrend_Signal",  None, None,  None),
    "Supertrend":         ("Supertrend",         0,    10000, 100),
    "Parabolic SAR":      ("Parabolic_SAR",      0,    10000, 100),
    "SMA 20":             ("SMA_20",             0,    10000, 100),
    "SMA 50":             ("SMA_50",             0,    10000, 100),
    "SMA 100":            ("SMA_100",            0,    10000, 100),
    "SMA 200":            ("SMA_200",            0,    10000, 100),
    "EMA 10":             ("EMA_10",             0,    10000, 100),
    "EMA 20":             ("EMA_20",             0,    10000, 100),
    "EMA 50":             ("EMA_50",             0,    10000, 100),
    "EMA 200":            ("EMA_200",            0,    10000, 100),
    "HMA 20":             ("HMA_20",             0,    10000, 100),
    "KAMA 20":            ("KAMA_20",            0,    10000, 100),
    "FRAMA":              ("FRAMA",              0,    10000, 100),
    "ADX 14":             ("ADX_14",             0,    100,   25),
    "Ichimoku Tenkan":    ("Ichimoku_Tenkan",    0,    10000, 100),
    "Ichimoku Kijun":     ("Ichimoku_Kijun",     0,    10000, 100),
    "Donchian High":      ("Donchian_High",      0,    10000, 100),
    "Donchian Low":       ("Donchian_Low",        0,    10000, 100),
    "RSI 14":             ("RSI_14",             0,    100,   50),
    "MACD Line":          ("MACD_line",          -100, 100,   0),
    "MACD Signal":        ("MACD_signal",        -100, 100,   0),
    "MACD Histogram":     ("MACD_hist",          -100, 100,   0),
    "Stoch K":            ("Stoch_K",            0,    100,   50),
    "Stoch D":            ("Stoch_D",            0,    100,   50),
    "Stoch RSI":          ("Stoch_RSI",          0,    1,     0.5),
    "CCI 20":             ("CCI_20",             -300, 300,   0),
    "Williams R":         ("Williams_R",         -100, 0,     -50),
    "ROC 12":             ("ROC_12",             -30,  30,    0),
    "Ultimate Oscillator":("Ultimate_Oscillator",0,    100,   50),
    "CMO":                ("CMO",                -100, 100,   0),
    "TRIX":               ("TRIX",               -2,   2,     0),
    "Schaff Trend Cycle": ("Schaff_Trend_Cycle", 0,    100,   75),
    "Fisher Transform":   ("Fisher_Transform",   -5,   5,     0),
    "Coppock Curve":      ("Coppock_Curve",      -10,  10,    0),
    "Vortex VI+":         ("Vortex_Pos",         0,    3,     1),
    "Vortex VI-":         ("Vortex_Neg",         0,    3,     1),
    "Elder Bull Power":   ("Elder_Bull_Power",   -50,  50,    0),
    "Elder Bear Power":   ("Elder_Bear_Power",   -50,  50,    0),
    "RVI":                ("RVI",                -1,   1,     0),
    "Mass Index":         ("Mass_Index",         20,   30,    26.5),
    "ATR 14":             ("ATR_14",             0,    300,   50),
    "Volatility %":       ("Volatility",         0,    10,    3),
    "BB Upper":           ("BB_Upper",           0,    10000, 100),
    "BB Middle":          ("BB_Middle",          0,    10000, 100),
    "BB Lower":           ("BB_Lower",           0,    10000, 100),
    "Keltner Upper":      ("Keltner_Upper",      0,    10000, 100),
    "Keltner Lower":      ("Keltner_Lower",      0,    10000, 100),
    "Spread (H-L)":       ("Spread",             0,    500,   50),
    "MFI 14":             ("MFI_14",             0,    100,   50),
    "OBV":                ("OBV",                -1e8, 1e8,   0),
    "VWAP":               ("VWAP",               0,    10000, 100),
    "Gap":                ("Gap",                -50,  50,    0),
    "Pivot Point":        ("Pivot_Point",        0,    10000, 100),
    "52W High":           ("52W_High",           0,    10000, 100),
    "52W Low":            ("52W_Low",            0,    10000, 100),
    "Returns %":          ("Returns",            -20,  20,    0),
    "Log Returns":        ("Log_Returns",        -0.2, 0.2,   0),
}

PRESETS_FILE = "presets.json"

def load_presets():
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE) as f:
            return json.load(f)
    return {}

def save_presets(presets):
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f)

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;500;600;700&display=swap');

:root {
    --bg: #f4f3ee;
    --bg2: #ffffff;
    --bg3: #eeede8;
    --bg4: #e4e3de;

    --border: rgba(0,0,0,0.08);
    --border2: rgba(0,0,0,0.14);

    --text: #1a1a14;
    --text2: #5a5950;
    --text3: #9a9990;

    --accent: #5a8a00;
    --accent2: #008a58;
    --red: #c24141;

    --radius: 10px;
    --radius-lg: 18px;

    --font-head: 'Syne', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;

    color-scheme: light !important;
}

/* ─────────────────────────────────────────────────────────────
   GLOBAL
───────────────────────────────────────────────────────────── */

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono);
}

.block-container {
    max-width: 100%;
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Typography */
h1, h2, h3 {
    font-family: var(--font-head) !important;
    color: var(--text) !important;
    letter-spacing: -0.03em;
}

h1 {
    font-size: 2.3rem !important;
    font-weight: 700 !important;
}

p,
label,
span,
div {
    color: var(--text);
}

/* ─────────────────────────────────────────────────────────────
   SIDEBAR
───────────────────────────────────────────────────────────── */

section[data-testid="stSidebar"] {
    background: #f8f7f3 !important;
    border-right: 1px solid rgba(0,0,0,0.06);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.8rem;
    padding-left: 1rem;
    padding-right: 1rem;
    padding-bottom: 2rem;
}

/* Sidebar title */
.sidebar-title {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.sidebar-icon {
    width: 42px;
    height: 42px;

    border-radius: 12px;

    background: linear-gradient(135deg, #f2f7ea, #edf6e6);
    border: 1px solid rgba(90,138,0,0.10);

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 20px;
}

.sidebar-heading {
    font-family: var(--font-head);
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text);
}

.sidebar-sub {
    color: var(--text3);
    font-size: 12px;
    margin-top: -2px;
}

/* Sidebar labels */
.sidebar-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;

    color: #5e5d55;

    margin-bottom: 12px;
}

/* Sidebar cards */
.sidebar-card {
    background: #ffffff;

    border: 1px solid rgba(0,0,0,0.06);
    border-radius: var(--radius-lg);

    padding: 16px;

    margin-bottom: 20px;

    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

/* Dividers */
hr {
    border-color: rgba(0,0,0,0.06) !important;
    margin-top: 24px !important;
    margin-bottom: 24px !important;
}

/* ─────────────────────────────────────────────────────────────
   INPUTS
───────────────────────────────────────────────────────────── */

.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: #fcfcfa !important;

    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 12px !important;

    min-height: 48px !important;

    display: flex !important;
    align-items: center !important;

    font-size: 14px !important;

    transition: all 0.15s ease;
}

.stSelectbox div[data-baseweb="select"] > div:hover,
.stNumberInput input:hover,
.stTextInput input:hover {
    border-color: rgba(90,138,0,0.25) !important;
}

/* Fix select alignment */
.stSelectbox span {
    display: flex !important;
    align-items: center !important;
}

/* Radio styling */
div[data-testid="stRadio"] {
    margin-top: -6px !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"] {
    padding: 2px 0 !important;
}

div[data-testid="stRadio"] > div {
    gap: 0.6rem;
}

/* ─────────────────────────────────────────────────────────────
   BUTTONS
───────────────────────────────────────────────────────────── */

.stButton button,
.stDownloadButton button,
.stLinkButton a {
    border-radius: 12px !important;

    border: 1px solid rgba(0,0,0,0.08) !important;

    background: white !important;

    min-height: 44px;

    font-size: 13px !important;
    font-weight: 600 !important;

    transition: all 0.15s ease;
}

/* Primary buttons */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6fa81f, #5a8a00) !important;

    color: white !important;

    border: none !important;

    font-weight: 700 !important;

    box-shadow: 0 6px 18px rgba(90,138,0,0.16);
}

.stButton button[kind="primary"]:hover {
    transform: translateY(-1px);

    box-shadow: 0 8px 22px rgba(90,138,0,0.22);
}

/* Secondary button hover */
.stButton button:hover,
.stDownloadButton button:hover,
.stLinkButton a:hover {
    border-color: rgba(90,138,0,0.22) !important;

    color: #5a8a00 !important;

    background: #f7fbef !important;
}

/* ─────────────────────────────────────────────────────────────
   METRICS
───────────────────────────────────────────────────────────── */

[data-testid="metric-container"] {
    background: var(--bg2);

    border: 1px solid var(--border);
    border-radius: var(--radius-lg);

    padding: 18px;

    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

[data-testid="stMetricLabel"] {
    color: var(--text3) !important;

    text-transform: uppercase;
    letter-spacing: 0.06em;

    font-size: 10px !important;
    font-family: var(--font-mono);
}

[data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;

    color: var(--text);

    font-size: 30px !important;
    font-weight: 700 !important;
}

/* ─────────────────────────────────────────────────────────────
   TABLE
───────────────────────────────────────────────────────────── */

.tbl-wrap {
    overflow-x: auto;
    overflow-y: auto;

    max-height: 72vh;

    background: var(--bg2);

    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
}

.screener-table {
    width: 100%;

    border-collapse: collapse;

    min-width: 1200px;

    font-size: 12px;
}

/* Table headers */
.screener-table th {
    position: sticky;
    top: 0;
    z-index: 5;

    background: var(--bg2);

    color: var(--text3);

    text-transform: uppercase;
    letter-spacing: 0.05em;

    font-size: 10px;
    font-weight: 600;

    padding: 12px;

    text-align: left;

    border-bottom: 1px solid var(--border);

    white-space: nowrap;
}

/* Table cells */
.screener-table td {
    padding: 12px;

    border-bottom: 1px solid var(--border);

    color: var(--text);

    white-space: nowrap;
}

.screener-table tr:last-child td {
    border-bottom: none;
}

.screener-table tr:hover td {
    background: var(--bg3);
}

/* ─────────────────────────────────────────────────────────────
   BADGES
───────────────────────────────────────────────────────────── */

.badge-buy {
    display: inline-flex !important;
    align-items: center;
    justify-content: center;

    padding: 4px 10px !important;

    border-radius: 999px !important;

    border: 1px solid rgba(0,138,88,0.18) !important;

    background: rgba(0,138,88,0.10) !important;

    color: #008a58 !important;

    font-size: 10px !important;
    font-weight: 700 !important;

    line-height: 1 !important;

    text-transform: uppercase;
}

.badge-sell {
    display: inline-flex !important;
    align-items: center;
    justify-content: center;

    padding: 4px 10px !important;

    border-radius: 999px !important;

    border: 1px solid rgba(194,65,65,0.18) !important;

    background: rgba(194,65,65,0.10) !important;

    color: #c24141 !important;

    font-size: 10px !important;
    font-weight: 700 !important;

    line-height: 1 !important;

    text-transform: uppercase;
}

/* Returns */
.up {
    color: var(--accent2);
    font-weight: 600;
}

.dn {
    color: var(--red);
    font-weight: 600;
}

.neu {
    color: var(--text2);
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #8c8b83 !important;
}

/* Remove excess spacing */
.element-container {
    margin-bottom: 0.6rem;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--bg4);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ── AUTH ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_gspread_client():
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        # Try streamlit secrets
        try:
            creds_json = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
        except Exception:
            pass
    if creds_json:
        info = json.loads(creds_json) if isinstance(creds_json, str) else dict(creds_json)
    else:
        with open("service_account.json") as f:
            info = json.load(f)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


# ── DATA FETCH ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_sheet_data():
    gc        = get_gspread_client()
    sh        = gc.open_by_key(SHEET_ID)
    worksheet = sh.get_worksheet(0)
    df        = gd.get_as_dataframe(worksheet, evaluate_formulas=True, dtype=str)
    df        = df.dropna(how='all').dropna(axis=1, how='all')
    numeric_cols = [c for c in df.columns if c not in ('Date','Stock','Universe','Supertrend_Signal')]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ── FILTER ENGINE ─────────────────────────────────────────────────────────────
def apply_filters(df, filters, logic):
    if not filters:
        return df
    masks = []
    for f in filters:
        col      = f['col']
        op       = f['op']
        val      = f['val']
        val_type = f.get('val_type', 'number')   # 'number' or 'column'

        if col not in df.columns:
            continue

        left = pd.to_numeric(df[col], errors='coerce')

        # Text filter (BUY / SELL)
        if op == '==' and val_type == 'number' and isinstance(val, str):
            masks.append(df[col].astype(str).str.upper() == val.upper())
            continue

        # Right-hand side: another column or a fixed number
        if val_type == 'column':
            if val not in df.columns:
                continue
            right = pd.to_numeric(df[val], errors='coerce')
        else:
            right = float(val)

        if op == '>':
            masks.append(left > right)
        elif op == '<':
            masks.append(left < right)
        elif op == '>=':
            masks.append(left >= right)
        elif op == '<=':
            masks.append(left <= right)
        elif op == '==':
            masks.append(left == right)

    if not masks:
        return df
    combined = masks[0]
    for m in masks[1:]:
        combined = (combined & m) if logic == "AND" else (combined | m)
    return df[combined]


# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'filters' not in st.session_state:
    st.session_state.filters = []
if 'logic' not in st.session_state:
    st.session_state.logic = "AND"
if 'universe' not in st.session_state:
    st.session_state.universe = "ALL"
if 'presets' not in st.session_state:
    st.session_state.presets = load_presets()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">
        <div class="sidebar-icon">📈</div>
        <div>
            <div class="sidebar-heading">Live Screener</div>
            <div class="sidebar-sub">Multi-indicator filter · AND / OR logic</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Universe toggle
    st.markdown('<div class="sidebar-label">Universe</div>', unsafe_allow_html=True)
    univ = st.radio("Universe", ["NIFTY 100", "LargeMidCap 250", "Both"], index=2, label_visibility="collapsed")
    st.session_state.universe = {
        "NIFTY 100":       "NIFTY100",
        "LargeMidCap 250": "NIFTY_LARGEMIDCAP250",
        "Both":            "ALL",
    }[univ]

    st.divider()

    # ── ADD FILTER (flat single dropdown) ─────────────────────────────────────
    st.markdown("**Add Filter**")
    sel_indicator = st.selectbox("Indicator", list(ALL_INDICATORS.keys()),
                                 label_visibility="collapsed")
    col_name, vmin, vmax, vdefault = ALL_INDICATORS[sel_indicator]

    is_text = vmin is None
    if is_text:
        op       = "=="
        val      = st.selectbox("Value", ["BUY", "SELL"])
        val_type = "number"
        val_display = val
    else:
        op = st.selectbox("Operator", [">", "<", ">=", "<="])

        # ── Toggle: fixed number vs another column ──────────────────────────
        threshold_mode = st.radio(
            "Threshold type",
            ["Fixed Value", "Another Column"],
            horizontal=True,
            key="threshold_mode",
            label_visibility="collapsed",
        )

        if threshold_mode == "Fixed Value":
            val      = st.number_input("Threshold", value=float(vdefault), step=0.1, format="%.2f")
            val_type = "number"
            val_display = f"{val:.2f}"
        else:
            # All numeric indicator labels except the currently selected one
            numeric_labels = [
                lbl for lbl, (c, mn, mx, _) in ALL_INDICATORS.items()
                if mn is not None and lbl != sel_indicator
            ]
            val_col_label = st.selectbox(
                "Compare to column",
                numeric_labels,
                key="val_col",
                label_visibility="visible",
            )
            val         = ALL_INDICATORS[val_col_label][0]   # actual column name
            val_type    = "column"
            val_display = val_col_label

    if st.button("＋ Add Filter", use_container_width=True, type="primary"):
        st.session_state.filters.append({
            "label":   sel_indicator,
            "col":     col_name,
            "op":      op,
            "val":     val,
            "val_type": val_type,
            "display": f"{sel_indicator} {op} {val_display}",
        })
        st.rerun()

    st.divider()

    # ── CUSTOM PRESETS ────────────────────────────────────────────────────────
    st.markdown("**My Presets**")

    # Save current filters as preset
    if st.session_state.filters:
        preset_name = st.text_input("Save current filters as:", placeholder="e.g. Breakout Setup",
                                    label_visibility="collapsed")
        if st.button("💾 Save Preset", use_container_width=True):
            if preset_name.strip():
                st.session_state.presets[preset_name.strip()] = {
                    "filters": st.session_state.filters.copy(),
                    "logic":   st.session_state.logic,
                }
                save_presets(st.session_state.presets)
                st.success(f'Saved "{preset_name.strip()}"')
                st.rerun()
            else:
                st.warning("Enter a name first.")

    # Load / delete saved presets
    if st.session_state.presets:
        for pname, pdata in list(st.session_state.presets.items()):
            pc1, pc2 = st.columns([3, 1])
            with pc1:
                if st.button(f"▶ {pname}", use_container_width=True, key=f"load_{pname}"):
                    st.session_state.filters = pdata["filters"].copy()
                    st.session_state.logic   = pdata.get("logic", "AND")
                    st.rerun()
            with pc2:
                if st.button("✕", key=f"del_{pname}"):
                    del st.session_state.presets[pname]
                    save_presets(st.session_state.presets)
                    st.rerun()
    else:
        st.caption("No presets yet. Add filters and save.")

    st.divider()

    # Logic toggle
    st.markdown("**Filter Logic**")
    st.session_state.logic = st.radio("", ["AND", "OR"], horizontal=True,
                                      index=0 if st.session_state.logic=="AND" else 1,
                                      label_visibility="collapsed")

    # Active filters list
    if st.session_state.filters:
        st.markdown("**Active Filters**")
        to_remove = []
        for i, f in enumerate(st.session_state.filters):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                display = f.get('display') or (
                    f"`{f['label']}` {f['op']} " +
                    (f['val'] if isinstance(f['val'], str) else f"{f['val']:.2f}")
                )
                st.caption(display)
            with col_b:
                if st.button("✕", key=f"rm_{i}"):
                    to_remove.append(i)
        if to_remove:
            st.session_state.filters = [f for i, f in enumerate(st.session_state.filters) if i not in to_remove]
            st.rerun()

        if st.button("✕ Clear All", use_container_width=True):
            st.session_state.filters = []
            st.rerun()

    st.divider()

    # Manual run trigger
    st.markdown("**Data Refresh**")
    if st.button("☁ Run Screener Now", use_container_width=True, type="primary"):
        with st.spinner("Fetching & calculating... (takes 3–5 min)"):
            try:
                import screener
                log_lines = []
                screener.run(log=lambda s: log_lines.append(s))
                st.success("Done! Refresh page to see new data.")
                for line in log_lines:
                    st.caption(line)
                st.cache_data.clear()
            except Exception as e:
                import traceback
                st.error(traceback.format_exc())

    st.link_button("↗ Open Source Sheet",
                   f"https://docs.google.com/spreadsheets/d/{SHEET_ID}",
                   use_container_width=True)


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("# Screener")
st.caption("Multi-indicator filter · AND / OR logic · live results")

# Load data
with st.spinner("Fetching from Google Sheet..."):
    try:
        df = load_sheet_data()
        data_ok = True
    except Exception as e:
        st.error(f"Failed to load sheet: {e}")
        data_ok = False
        df = pd.DataFrame()

if data_ok and not df.empty:
    # Universe filter
    if st.session_state.universe != "ALL":
        view_df = df[df['Universe'] == st.session_state.universe].copy()
    else:
        view_df = df.copy()

    # Apply filters
    filtered = apply_filters(view_df, st.session_state.filters, st.session_state.logic)

    # Sort controls
    sort_col, sort_dir = st.columns([5, 1.5], vertical_alignment="bottom")
    with sort_col:
        sort_by = st.selectbox("Sort by", ["Returns", "Close", "Volume", "RSI_14", "MFI_14", "Open", "High", "Low"],
                               label_visibility="visible", key="sort_by")
    with sort_dir:
        sort_asc = st.selectbox("", ["High→Low", "Low→High"], label_visibility="collapsed", key="sort_dir")

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=(sort_asc=="Low→High"))

    # KPIs
    last_date = df['Date'].max() if 'Date' in df.columns else "—"
    buy_count = (filtered['Supertrend_Signal'].astype(str).str.upper()=="BUY").sum() if 'Supertrend_Signal' in filtered.columns else 0

    k1, k2, k3, k4 = st.columns([1,1,1,1], gap="medium")
    k1.metric("Total Stocks", len(view_df))
    k2.metric("Matching", len(filtered))
    k3.metric("Active Filters", len(st.session_state.filters))
    k4.metric("Last Update", str(last_date))

    st.divider()

    # ── STOCK DETAIL POPUP ────────────────────────────────────────────────────
    @st.dialog("Stock Details", width="large")
    def show_stock_detail(row):
        stock_name = str(row.get('Stock', '')).replace('.NS', '')
        univ_full  = "NIFTY 100" if str(row.get('Universe','')) == "NIFTY100" else "LargeMidCap 250"
        signal     = str(row.get('Supertrend_Signal', '')).upper()
        ret_val    = row.get('Returns', None)

        # Header
        badge_color = "#008a58" if signal == "BUY" else "#c24141"
        badge_bg    = "rgba(0,138,88,0.10)" if signal == "BUY" else "rgba(194,65,65,0.10)"
        try:
            ret_str   = f"{float(ret_val):+.2f}%"
            ret_color = "#008a58" if float(ret_val) > 0 else ("#c24141" if float(ret_val) < 0 else "#5a5950")
        except:
            ret_str   = "—"
            ret_color = "#5a5950"

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
            <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;letter-spacing:-0.03em;">
                {stock_name}
            </div>
            <div style="font-size:11px;color:#9a9990;background:#eeede8;padding:4px 10px;
                        border-radius:999px;font-weight:600;">{univ_full}</div>
            <div style="font-size:11px;font-weight:700;padding:4px 10px;border-radius:999px;
                        color:{badge_color};background:{badge_bg};border:1px solid {badge_color}33;">
                {signal}
            </div>
            <div style="font-size:1.1rem;font-weight:700;color:{ret_color};margin-left:auto;">
                {ret_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

        def fmt(val, decimals=2):
            if pd.isna(val): return "—"
            try:    return f"{float(val):,.{decimals}f}"
            except: return str(val)

        # Group columns into sections for clean block layout
        SECTIONS = {
            "📊 Price": [
                ("Date",       "Date",       0),
                ("Open",       "Open",       2),
                ("High",       "High",       2),
                ("Low",        "Low",        2),
                ("Close",      "Close",      2),
                ("Prev Close", "Prev_Close", 2),
                ("Volume",     "Volume",     0),
            ],
            "📈 Moving Averages": [
                ("SMA 20",   "SMA_20",   2),
                ("SMA 50",   "SMA_50",   2),
                ("SMA 100",  "SMA_100",  2),
                ("SMA 200",  "SMA_200",  2),
                ("EMA 10",   "EMA_10",   2),
                ("EMA 13",   "EMA_13",   2),
                ("EMA 20",   "EMA_20",   2),
                ("EMA 50",   "EMA_50",   2),
                ("EMA 200",  "EMA_200",  2),
            ],
            "🔬 Oscillators": [
                ("RSI 14",       "RSI_14",      2),
                ("MACD Line",    "MACD_line",   4),
                ("MACD Signal",  "MACD_signal", 4),
                ("MACD Hist",    "MACD_hist",   4),
                ("CCI 20",       "CCI_20",      2),
                ("MFI 14",       "MFI_14",      2),
            ],
            "🚦 Trend": [
                ("Supertrend",        "Supertrend",        2),
                ("Supertrend Signal", "Supertrend_Signal", 0),
            ],
        }

        for section_title, fields in SECTIONS.items():
            # Only render section if at least one field exists in the row
            available = [(lbl, col, dec) for lbl, col, dec in fields if col in row.index]
            if not available:
                continue

            st.markdown(f"**{section_title}**")
            cols_per_row = 4
            for chunk_start in range(0, len(available), cols_per_row):
                chunk = available[chunk_start:chunk_start + cols_per_row]
                st_cols = st.columns(cols_per_row)
                for i, (lbl, col, dec) in enumerate(chunk):
                    raw = row.get(col, None)
                    # Special rendering for signal and returns
                    if col == "Supertrend_Signal":
                        display_val = str(raw).upper() if pd.notna(raw) else "—"
                        color = "#008a58" if display_val == "BUY" else "#c24141"
                        st_cols[i].markdown(
                            f"<div style='font-size:10px;color:#9a9990;text-transform:uppercase;"
                            f"letter-spacing:0.05em;margin-bottom:2px'>{lbl}</div>"
                            f"<div style='font-size:15px;font-weight:700;color:{color}'>{display_val}</div>",
                            unsafe_allow_html=True
                        )
                    elif col == "Returns":
                        try:
                            v = float(raw)
                            color = "#008a58" if v > 0 else ("#c24141" if v < 0 else "#5a5950")
                            display_val = f"{v:+.2f}%"
                        except:
                            color = "#5a5950"
                            display_val = "—"
                        st_cols[i].markdown(
                            f"<div style='font-size:10px;color:#9a9990;text-transform:uppercase;"
                            f"letter-spacing:0.05em;margin-bottom:2px'>{lbl}</div>"
                            f"<div style='font-size:15px;font-weight:700;color:{color}'>{display_val}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        display_val = fmt(raw, dec)
                        st_cols[i].markdown(
                            f"<div style='font-size:10px;color:#9a9990;text-transform:uppercase;"
                            f"letter-spacing:0.05em;margin-bottom:2px'>{lbl}</div>"
                            f"<div style='font-size:15px;font-weight:600;color:#1a1a14'>{display_val}</div>",
                            unsafe_allow_html=True
                        )
            st.divider()

        # Any remaining columns not covered by sections
        covered = {col for fields in SECTIONS.values() for _, col, _ in fields}
        extra   = [(c, c) for c in row.index if c not in covered and c not in ('Stock','Universe')]
        if extra:
            st.markdown("**📋 Other**")
            st_cols = st.columns(4)
            for i, (lbl, col) in enumerate(extra):
                raw = row.get(col, None)
                st_cols[i % 4].markdown(
                    f"<div style='font-size:10px;color:#9a9990;text-transform:uppercase;"
                    f"letter-spacing:0.05em;margin-bottom:2px'>{lbl}</div>"
                    f"<div style='font-size:14px;font-weight:600;color:#1a1a14'>"
                    f"{fmt(raw)}</div>",
                    unsafe_allow_html=True
                )

    # ── RESULTS TABLE ─────────────────────────────────────────────────────────
    if filtered.empty:
        st.info("∅ No stocks match active filters. Relax a threshold or switch to OR logic.")
    else:
        def fmt(val, decimals=2):
            if pd.isna(val): return "—"
            try: return f"{float(val):.{decimals}f}"
            except: return str(val)

        def ret_class(val):
            try:
                v = float(val)
                cls = "up" if v > 0 else ("dn" if v < 0 else "neu")
                return f'<span class="{cls}">{v:+.2f}%</span>'
            except:
                return "—"

        def signal_badge(val):
            v = str(val).upper()
            if v == "BUY":  return '<span class="badge-buy">BUY</span>'
            if v == "SELL": return '<span class="badge-sell">SELL</span>'
            return v

        # ── Ghost-button CSS: stock name button looks like plain bold text ──
        st.markdown("""
        <style>
        .stock-btn-row button {
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 2px !important;
            margin: 0 !important;
            min-height: unset !important;
            height: auto !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            font-family: 'IBM Plex Mono', monospace !important;
            color: #1a1a14 !important;
            text-decoration: underline !important;
            text-decoration-color: rgba(90,138,0,0.4) !important;
            text-underline-offset: 3px !important;
            letter-spacing: 0 !important;
            justify-content: flex-start !important;
            width: auto !important;
        }
        .stock-btn-row button:hover {
            color: #5a8a00 !important;
            background: none !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stock-btn-row button:focus {
            box-shadow: none !important;
            outline: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

        COL_W  = [1.8, 0.7, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.6, 0.9, 0.9, 1.0]
        BORDER = "border-bottom:1px solid rgba(0,0,0,0.06);"
        CELL   = f"padding:10px 4px;white-space:nowrap;font-size:12px;{BORDER}"

        # ── Header row ────────────────────────────────────────────────────
        hcols = st.columns(COL_W)
        for hc, lbl in zip(hcols, [
            "STOCK","UNIV","OPEN","HIGH","LOW","CLOSE","PREV CLOSE","RETURN%","VOLUME","RSI","MFI","SIGNAL"
        ]):
            hc.markdown(
                f"<div style='font-size:10px;font-weight:600;color:#9a9990;"
                f"text-transform:uppercase;letter-spacing:0.05em;"
                f"padding:8px 4px 6px 4px;border-bottom:2px solid rgba(0,0,0,0.10);'>"
                f"{lbl}</div>", unsafe_allow_html=True)

        # ── Data rows ─────────────────────────────────────────────────────
        clicked_row = None
        for idx, (_, row) in enumerate(filtered.iterrows()):
            stock  = str(row.get('Stock','')).replace('.NS','')
            univ   = "N100" if str(row.get('Universe',''))=="NIFTY100" else "LMC"
            rcols  = st.columns(COL_W)

            with rcols[0]:
                st.markdown('<div class="stock-btn-row">', unsafe_allow_html=True)
                if st.button(stock, key=f"sb_{idx}_{stock}_{univ}"):
                    clicked_row = row
                st.markdown('</div>', unsafe_allow_html=True)

            rcols[1].markdown(f"<div style='{CELL}color:#9a9990;font-size:11px'>{univ}</div>", unsafe_allow_html=True)
            for rc, val, dec in [
                (rcols[2], row.get('Open'),      2),
                (rcols[3], row.get('High'),       2),
                (rcols[4], row.get('Low'),        2),
                (rcols[5], row.get('Close'),      2),
                (rcols[6], row.get('Prev_Close'), 2),
            ]:
                rc.markdown(f"<div style='{CELL}'>{fmt(val,dec)}</div>", unsafe_allow_html=True)

            rcols[7].markdown(f"<div style='{CELL}'>{ret_class(row.get('Returns'))}</div>", unsafe_allow_html=True)
            rcols[8].markdown(f"<div style='{CELL}'>{fmt(row.get('Volume'),0)}</div>", unsafe_allow_html=True)
            rcols[9].markdown(f"<div style='{CELL}'>{fmt(row.get('RSI_14'))}</div>", unsafe_allow_html=True)
            rcols[10].markdown(f"<div style='{CELL}'>{fmt(row.get('MFI_14'))}</div>", unsafe_allow_html=True)
            rcols[11].markdown(f"<div style='{CELL}'>{signal_badge(row.get('Supertrend_Signal',''))}</div>", unsafe_allow_html=True)

        # ── Open dialog for clicked stock ─────────────────────────────────
        if clicked_row is not None:
            show_stock_detail(clicked_row)

        st.caption(f"{len(filtered)} stocks · {st.session_state.logic} logic · sorted by {sort_by}")

        # CSV download
        st.download_button(
            "⬇ Download CSV",
            filtered.to_csv(index=False),
            file_name=f"screener_{last_date}.csv",
            mime="text/csv",
        )
else:
    if data_ok:
        st.warning("Sheet is empty. Run the screener first.")
