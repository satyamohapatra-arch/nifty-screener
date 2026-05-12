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

  --radius: 8px;
  --radius-lg: 14px;

  --font-head: 'Syne', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-mono);
    background: var(--bg);
    color: var(--text);
}

body {
    background: var(--bg);
}

/* BUY badge */
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

/* SELL badge */
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

/* Main app */
.stApp {
    background: var(--bg);
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
    color: var(--text);
    letter-spacing: -0.03em;
}

h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
}

p, label, span, div {
    color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg2);
    border-right: 1px solid var(--border);
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
        col, op, val = f['col'], f['op'], f['val']
        if col not in df.columns:
            continue
        if op == '==' and isinstance(val, str):
            masks.append(df[col].astype(str).str.upper() == val.upper())
        elif op == '>':
            masks.append(pd.to_numeric(df[col], errors='coerce') > float(val))
        elif op == '<':
            masks.append(pd.to_numeric(df[col], errors='coerce') < float(val))
        elif op == '>=':
            masks.append(pd.to_numeric(df[col], errors='coerce') >= float(val))
        elif op == '<=':
            masks.append(pd.to_numeric(df[col], errors='coerce') <= float(val))
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
    st.markdown("### 📈 Live Screener")
    st.caption("Multi-indicator filter · AND / OR logic")
    st.divider()

    # Universe toggle
    st.markdown("**Universe**")
    univ = st.radio("", ["NIFTY 100", "LargeMidCap 250", "Both"], index=2, label_visibility="collapsed")
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
        op  = "=="
        val = st.selectbox("Value", ["BUY", "SELL"])
    else:
        op  = st.selectbox("Operator", [">", "<", ">=", "<="])
        val = st.number_input("Threshold", value=float(vdefault), step=0.1, format="%.2f")

    if st.button("＋ Add Filter", use_container_width=True):
        st.session_state.filters.append({
            "label": sel_indicator,
            "col":   col_name,
            "op":    op,
            "val":   val,
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
                val_str = f['val'] if isinstance(f['val'], str) else f"{f['val']:.2f}"
                st.caption(f"`{f['label']}` {f['op']} {val_str}")
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
                st.error(f"Error: {e}")

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

    # Results table
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

        rows = []
        display_cols = ['Stock','Universe','Open','High','Low','Close','Prev_Close','Returns','Volume','RSI_14','MFI_14','Supertrend_Signal']
        for _, row in filtered.iterrows():
            stock = str(row.get('Stock','')).replace('.NS','')
            univ  = "N100" if str(row.get('Universe',''))=="NIFTY100" else "LMC"
            rows.append(f"""<tr>
                <td><strong>{stock}</strong></td>
                <td><span style="color:#9b9a94;font-size:10px">{univ}</span></td>
                <td>{fmt(row.get('Open'))}</td>
                <td>{fmt(row.get('High'))}</td>
                <td>{fmt(row.get('Low'))}</td>
                <td>{fmt(row.get('Close'))}</td>
                <td>{fmt(row.get('Prev_Close'))}</td>
                <td>{ret_class(row.get('Returns'))}</td>
                <td>{fmt(row.get('Volume'),0)}</td>
                <td>{fmt(row.get('RSI_14'))}</td>
                <td>{fmt(row.get('MFI_14'))}</td>
                <td>{signal_badge(row.get('Supertrend_Signal',''))}</td>
            </tr>""")

        table_html = f"""
        <div class="tbl-wrap">
        <table class="screener-table">
          <thead><tr>
            <th>Stock</th><th>Univ</th>
            <th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Prev Close</th>
            <th>Return%</th><th>Volume</th><th>RSI</th><th>MFI</th><th>Signal</th>
          </tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)
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
