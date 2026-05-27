# screener.py — Full indicator suite, matches Google Colab exactly.
# Pushes 68-column snapshot to Google Sheet.

import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import gspread_dataframe as gd
import zoneinfo
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials

# ── CONFIG ────────────────────────────────────────────────────────────────────

SHEET_ID        = "1JWHOhfTFhS0345GC4KMGHYCa1F8YEdDk2Skb85R2p5U"
NIFTY100_URL    = "https://drive.google.com/uc?id=1SbcUYzWZPEd2zhK1kkNndYVmkDskp9fp"
LARGEMIDCAP_URL = "https://drive.google.com/uc?id=1BzI5KjtkkQ2H-LvUNnFXJDAki5IslJUP"

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# ── COLUMNS (68 total) ────────────────────────────────────────────────────────

COLS = [
    'Date', 'Stock', 'Universe',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_10', 'EMA_13', 'EMA_20', 'EMA_50', 'EMA_200',
    'HMA_20', 'KAMA_20',
    'Ichimoku_Tenkan', 'Ichimoku_Kijun',
    'Supertrend', 'Supertrend_Signal',
    'Parabolic_SAR', 'ADX_14',
    'Donchian_High', 'Donchian_Low',
    'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_hist',
    'Stoch_K', 'Stoch_D', 'Stoch_RSI',
    'CCI_20', 'Williams_R', 'ROC_12',
    'Ultimate_Oscillator',
    'ATR_14',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'Keltner_Upper', 'Keltner_Lower',
    'OBV', 'VWAP', 'MFI_14',
    'Pivot_Point',
    '52W_High', '52W_Low',
    'Fisher_Transform', 'Schaff_Trend_Cycle', 'FRAMA',
    'Coppock_Curve', 'Mass_Index',
    'Vortex_Pos', 'Vortex_Neg',
    'CMO', 'TRIX',
    'Elder_Bull_Power', 'Elder_Bear_Power',
    'RVI',
    'Prev_Close', 'Gap', 'Returns', 'Log_Returns',
    'Spread', 'Volatility',
]

# ── GOOGLE AUTH ───────────────────────────────────────────────────────────────

def get_gspread_client():
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        try:
            import streamlit as st
            val = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
            creds_json = json.dumps(dict(val)) if hasattr(val, 'keys') else val
        except Exception:
            pass
    if creds_json:
        info = json.loads(creds_json) if isinstance(creds_json, str) else dict(creds_json)
    else:
        with open("service_account.json") as f:
            info = json.load(f)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

# ── LAST TRADING DAY ──────────────────────────────────────────────────────────

def last_trading_day():
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    day = now if now >= market_close else now - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.strftime('%Y-%m-%d')

# ── BULK DOWNLOAD ─────────────────────────────────────────────────────────────

def download_universe(symbols_url, universe_name, log=print):
    master_path = f"master_data_{universe_name}.csv"
    stocks      = [s + ".NS" for s in pd.read_csv(symbols_url)["Symbol"].tolist()]
    END_DATE    = last_trading_day()
    FETCH_END   = (datetime.strptime(END_DATE, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    log(f"[{universe_name}] {len(stocks)} stocks | end={END_DATE}")

    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        existing['Date'] = pd.to_datetime(existing['Date'])
        last_date = existing['Date'].max()
        if pd.isna(last_date):
            start_date = "2021-01-01"
            log(f"[{universe_name}] No valid dates in master — full download")
        else:
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            log(f"[{universe_name}] Last date in master: {last_date.date()} — fetching from {start_date}")
    else:
        start_date = "2021-01-01"
        existing   = None
        log(f"[{universe_name}] No master CSV found — full download from {start_date}")

    if start_date > END_DATE:
        log(f"[{universe_name}] Already up to date. Skipping download.")
        return existing if existing is not None else pd.DataFrame()

    # ── Batch download: 10 tickers at a time for reliable full history ────────
    BATCH_SIZE = 10
    batches    = [stocks[i:i+BATCH_SIZE] for i in range(0, len(stocks), BATCH_SIZE)]
    log(f"[{universe_name}] Downloading {len(stocks)} tickers in {len(batches)} batches of {BATCH_SIZE}...")

    all_data = []
    success  = 0
    skipped  = 0

    for b_idx, batch in enumerate(batches):
        try:
            raw = yf.download(
                batch,
                start=start_date,
                end=FETCH_END,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            log(f"  Batch {b_idx+1} download exception: {e}")
            skipped += len(batch)
            continue

        if raw is None or raw.empty:
            log(f"  Batch {b_idx+1} returned empty — skipping")
            skipped += len(batch)
            continue

        for stock in batch:
            try:
                # Extract per-ticker slice
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl0 = raw.columns.get_level_values(0).unique().tolist()
                    lvl1 = raw.columns.get_level_values(1).unique().tolist()

                    # group_by='ticker' → level 0 = ticker, level 1 = field
                    if stock in lvl0:
                        df = raw[stock].copy()
                    # single ticker in batch → level 0 = field, no ticker level
                    elif len(batch) == 1:
                        df = raw.droplevel(1, axis=1).copy()
                        df = df.loc[:, ~df.columns.duplicated()]
                    else:
                        log(f"  SKIP {stock}: not found in batch result")
                        skipped += 1
                        continue
                else:
                    df = raw.copy()

                # Drop rows where ALL price cols are NaN
                df = df.dropna(subset=['Close'])
                if df.empty:
                    log(f"  SKIP {stock}: no Close data")
                    skipped += 1
                    continue

                df = df.reset_index()

                # Normalise column names — 'index' happens when yfinance index has no name
                rename_map = {}
                for c in df.columns:
                    cl = str(c).lower().strip()
                    if cl in ('date', 'datetime', 'index'):  rename_map[c] = 'Date'
                    elif cl == 'open':                       rename_map[c] = 'Open'
                    elif cl == 'high':                       rename_map[c] = 'High'
                    elif cl == 'low':                        rename_map[c] = 'Low'
                    elif cl == 'close':                      rename_map[c] = 'Close'
                    elif cl == 'volume':                     rename_map[c] = 'Volume'
                df = df.rename(columns=rename_map)

                required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing  = [c for c in required if c not in df.columns]
                if missing:
                    log(f"  SKIP {stock}: missing {missing} — got {df.columns.tolist()}")
                    skipped += 1
                    continue

                df = df[required].copy()
                df['Date']     = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df['Stock']    = stock
                df['Universe'] = universe_name
                all_data.append(df)
                success += 1

            except Exception as e:
                log(f"  ERROR {stock}: {e}")
                skipped += 1

        log(f"  Batch {b_idx+1}/{len(batches)} done | running total: {success} OK")

    log(f"[{universe_name}] Parsed: {success} OK | {skipped} skipped")

    if not all_data:
        log(f"[{universe_name}] No usable data after parsing — aborting save.")
        return existing if existing is not None else pd.DataFrame()

    new_data = pd.concat(all_data, ignore_index=True)
    log(f"[{universe_name}] New rows: {len(new_data):,}")

    if existing is not None and not new_data.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'Stock'])
        combined.to_csv(master_path, index=False)
        log(f"[{universe_name}] Master updated: {len(combined):,} rows → {master_path}")
        return combined

    if not new_data.empty:
        new_data.to_csv(master_path, index=False)
        log(f"[{universe_name}] Master saved: {len(new_data):,} rows → {master_path}")
    return new_data

# ── INDICATORS ────────────────────────────────────────────────────────────────

def calculate_indicators(data):
    data  = data.sort_values('Date').copy()
    n     = len(data)
    if n < 30:
        return data
    close = data['Close']
    high  = data['High']
    low   = data['Low']
    vol   = data['Volume']
    open_ = data['Open']

    # SMA
    for w in [20, 50, 100, 200]:
        data[f'SMA_{w}'] = close.rolling(w).mean()

    # EMA
    for w in [10, 13, 20, 50, 200]:
        data[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    # HMA_20
    half_wma = wma(close, 10)
    full_wma = wma(close, 20)
    hull_raw = 2 * half_wma - full_wma
    data['HMA_20'] = wma(hull_raw, int(np.sqrt(20)))

    # KAMA_20
    fast_sc = 2 / (2  + 1)
    slow_sc = 2 / (30 + 1)
    kama = close.copy().astype(float)
    for i in range(20, n):
        direction  = abs(close.iloc[i] - close.iloc[i - 20])
        volatility = close.diff().abs().iloc[i - 19:i + 1].sum()
        er   = direction / (volatility + 1e-10)
        sc   = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama.iloc[i] = kama.iloc[i - 1] + sc * (close.iloc[i] - kama.iloc[i - 1])
    kama.iloc[:20] = np.nan
    data['KAMA_20'] = kama

    # Ichimoku
    data['Ichimoku_Tenkan'] = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    data['Ichimoku_Kijun']  = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # Donchian
    data['Donchian_High'] = high.rolling(20).max()
    data['Donchian_Low']  = low.rolling(20).min()

    # ATR_14
    hl   = high - low
    hcp  = (high - close.shift()).abs()
    lcp  = (low  - close.shift()).abs()
    tr   = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    data['ATR_14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ADX_14
    up   = high.diff()
    down = low.shift() - low
    pdm  = pd.Series(np.where((up > down) & (up > 0),   up,   0), index=data.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0), index=data.index)
    atr_s = data['ATR_14'].replace(0, 1e-10)
    pdi   = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    mdi   = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    data['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Parabolic SAR
    af_start, af_step, af_max = 0.02, 0.02, 0.2
    high_arr  = high.values
    low_arr   = low.values
    sar       = np.zeros(n)
    trend_arr = np.ones(n)
    ep        = np.zeros(n)
    af_arr    = np.zeros(n)
    sar[0]    = low_arr[0]
    ep[0]     = high_arr[0]
    af_arr[0] = af_start
    for i in range(1, n):
        ps, pe, pa, pt = sar[i-1], ep[i-1], af_arr[i-1], trend_arr[i-1]
        if pt == 1:
            sar[i] = ps + pa * (pe - ps)
            sar[i] = min(sar[i], low_arr[i-1], low_arr[i-2] if i > 1 else low_arr[i-1])
            if low_arr[i] < sar[i]:
                trend_arr[i] = -1; sar[i] = pe; ep[i] = low_arr[i]; af_arr[i] = af_start
            else:
                trend_arr[i] = 1; ep[i] = max(pe, high_arr[i])
                af_arr[i] = min(af_max, pa + af_step) if ep[i] > pe else pa
        else:
            sar[i] = ps + pa * (pe - ps)
            sar[i] = max(sar[i], high_arr[i-1], high_arr[i-2] if i > 1 else high_arr[i-1])
            if high_arr[i] > sar[i]:
                trend_arr[i] = 1; sar[i] = pe; ep[i] = high_arr[i]; af_arr[i] = af_start
            else:
                trend_arr[i] = -1; ep[i] = min(pe, low_arr[i])
                af_arr[i] = min(af_max, pa + af_step) if ep[i] < pe else pa
    data['Parabolic_SAR'] = np.round(sar, 2)

    # Supertrend (period=7, multiplier=3)
    atr7       = tr.ewm(span=7, adjust=False).mean()
    hl2        = (high + low) / 2
    upper_band = (hl2 + 3 * atr7).values
    lower_band = (hl2 - 3 * atr7).values
    close_arr  = close.values
    supertrend = np.zeros(n)
    signal     = [''] * n
    supertrend[0] = upper_band[0]
    signal[0]     = 'SELL'
    for i in range(1, n):
        if close_arr[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]; signal[i] = 'BUY'
        else:
            supertrend[i] = upper_band[i]; signal[i] = 'SELL'
    data['Supertrend']        = np.round(supertrend, 2)
    data['Supertrend_Signal'] = signal

    # RSI_14
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    data['RSI_14'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD_line']   = ema12 - ema26
    data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
    data['MACD_hist']   = data['MACD_line'] - data['MACD_signal']

    # Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    data['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()

    # Stoch RSI
    rsi        = data['RSI_14']
    rsi_low14  = rsi.rolling(14).min()
    rsi_high14 = rsi.rolling(14).max()
    data['Stoch_RSI'] = (rsi - rsi_low14) / (rsi_high14 - rsi_low14 + 1e-10)

    # CCI_20
    tp2     = (high + low + close) / 3
    tp_mean = tp2.rolling(20).mean()
    tp_std  = tp2.rolling(20).std()
    data['CCI_20'] = (tp2 - tp_mean) / (0.015 * tp_std + 1e-10)

    # Williams %R
    low14w  = low.rolling(14).min()
    high14w = high.rolling(14).max()
    data['Williams_R'] = -100 * (high14w - close) / (high14w - low14w + 1e-10)

    # ROC_12
    data['ROC_12'] = close.pct_change(12) * 100

    # Ultimate Oscillator
    prev_close = close.shift(1)
    bp  = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr2 = pd.concat([high, prev_close], axis=1).max(axis=1) - \
          pd.concat([low,  prev_close], axis=1).min(axis=1)
    avg7  = bp.rolling(7).sum()  / (tr2.rolling(7).sum()  + 1e-10)
    avg14 = bp.rolling(14).sum() / (tr2.rolling(14).sum() + 1e-10)
    avg28 = bp.rolling(28).sum() / (tr2.rolling(28).sum() + 1e-10)
    data['Ultimate_Oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

    # CMO
    up_sum   = gain.rolling(14).sum()
    down_sum = loss.rolling(14).sum()
    data['CMO'] = 100 * (up_sum - down_sum) / (up_sum + down_sum + 1e-10)

    # TRIX
    ema1 = close.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    data['TRIX'] = ema3.pct_change() * 100

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data['BB_Upper']  = bb_mid + 2 * bb_std
    data['BB_Middle'] = bb_mid
    data['BB_Lower']  = bb_mid - 2 * bb_std

    # Keltner Channel
    kc_mid = close.ewm(span=20, adjust=False).mean()
    data['Keltner_Upper'] = kc_mid + 1.5 * data['ATR_14']
    data['Keltner_Lower'] = kc_mid - 1.5 * data['ATR_14']

    # OBV
    obv = np.zeros(n)
    for i in range(1, n):
        if close_arr[i] > close.values[i - 1]:
            obv[i] = obv[i - 1] + vol.values[i]
        elif close_arr[i] < close.values[i - 1]:
            obv[i] = obv[i - 1] - vol.values[i]
        else:
            obv[i] = obv[i - 1]
    data['OBV'] = obv

    # VWAP (rolling 14-day)
    tp_vwap = (high + low + close) / 3
    data['VWAP'] = (tp_vwap * vol).rolling(14).sum() / (vol.rolling(14).sum() + 1e-10)

    # MFI_14
    tp3  = (high + low + close) / 3
    rmf  = tp3 * vol
    pmf  = rmf.where(tp3 > tp3.shift(), 0.0)
    nmf  = rmf.where(tp3 < tp3.shift(), 0.0)
    mfr  = pmf.rolling(14).sum() / (nmf.rolling(14).sum() + 1e-10)
    data['MFI_14'] = 100 - (100 / (1 + mfr))

    # Pivot Point
    data['Pivot_Point'] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3

    # 52W High/Low
    data['52W_High'] = high.rolling(252).max()
    data['52W_Low']  = low.rolling(252).min()

    # Fisher Transform
    median_price = (high + low) / 2
    highest_high = median_price.rolling(10).max()
    lowest_low   = median_price.rolling(10).min()
    value = 2 * ((median_price - lowest_low) / (highest_high - lowest_low + 1e-10)) - 1
    value = value.clip(-0.999, 0.999)
    data['Fisher_Transform'] = 0.5 * np.log((1 + value) / (1 - value))

    # Schaff Trend Cycle
    stc_macd  = data['MACD_line']
    stc_low1  = stc_macd.rolling(10).min()
    stc_high1 = stc_macd.rolling(10).max()
    stc_k1    = 100 * (stc_macd - stc_low1) / (stc_high1 - stc_low1 + 1e-10)
    stc_d1    = stc_k1.ewm(span=3, adjust=False).mean()
    stc_low2  = stc_d1.rolling(10).min()
    stc_high2 = stc_d1.rolling(10).max()
    stc_k2    = 100 * (stc_d1 - stc_low2) / (stc_high2 - stc_low2 + 1e-10)
    data['Schaff_Trend_Cycle'] = stc_k2.ewm(span=3, adjust=False).mean()

    # FRAMA
    frama = close.copy().astype(float)
    w     = 16
    for i in range(w, n):
        seg  = close.iloc[i - w:i]
        half = w // 2
        h1 = seg.iloc[:half].max(); l1 = seg.iloc[:half].min()
        h2 = seg.iloc[half:].max(); l2 = seg.iloc[half:].min()
        h3 = seg.max();             l3 = seg.min()
        n1 = (h1 - l1) / half
        n2 = (h2 - l2) / half
        n3 = (h3 - l3) / w
        if n1 + n2 > 0 and n3 > 0:
            dim = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        else:
            dim = 1.5
        alpha = np.clip(np.exp(-4.6 * (dim - 1)), 0.01, 1.0)
        frama.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * frama.iloc[i - 1]
    frama.iloc[:w] = np.nan
    data['FRAMA'] = frama

    # Coppock Curve
    roc11 = close.pct_change(11) * 100
    roc14 = close.pct_change(14) * 100
    data['Coppock_Curve'] = wma(roc11 + roc14, 10)

    # Mass Index
    ema9_hl  = (high - low).ewm(span=9, adjust=False).mean()
    ema9_ema = ema9_hl.ewm(span=9, adjust=False).mean()
    data['Mass_Index'] = (ema9_hl / (ema9_ema + 1e-10)).rolling(25).sum()

    # Vortex
    vm_pos = (high - low.shift(1)).abs()
    vm_neg = (low  - high.shift(1)).abs()
    data['Vortex_Pos'] = vm_pos.rolling(14).sum() / (tr.rolling(14).sum() + 1e-10)
    data['Vortex_Neg'] = vm_neg.rolling(14).sum() / (tr.rolling(14).sum() + 1e-10)

    # Elder
    data['Elder_Bull_Power'] = high - data['EMA_13']
    data['Elder_Bear_Power'] = low  - data['EMA_13']

    # RVI
    co   = close - open_
    hl_r = high - low
    rvi_num = (co + 2 * co.shift(1) + 2 * co.shift(2) + co.shift(3)) / 6
    rvi_den = (hl_r + 2 * hl_r.shift(1) + 2 * hl_r.shift(2) + hl_r.shift(3)) / 6
    data['RVI'] = rvi_num.rolling(10).sum() / (rvi_den.rolling(10).sum() + 1e-10)

    # Derived
    data['Prev_Close']  = close.shift(1)
    data['Gap']         = open_ - close.shift(1)
    data['Returns']     = close.pct_change() * 100
    data['Log_Returns'] = np.log(close / (close.shift(1) + 1e-10))
    data['Spread']      = high - low
    data['Volatility']  = data['Returns'].rolling(20).std()

    return data

# ── RUN ───────────────────────────────────────────────────────────────────────

def run(log=print):
    if os.path.exists("master_data.csv"):
        os.remove("master_data.csv")
        log("Removed legacy master_data.csv")

    universes = [
        (NIFTY100_URL,    "NIFTY100"),
        (LARGEMIDCAP_URL, "NIFTY_LARGEMIDCAP250"),
    ]

    for url, u_name in universes:
        log(f"\n{'='*50}")
        log(f"Downloading {u_name}...")
        log(f"{'='*50}")
        download_universe(url, u_name, log=log)

    log(f"\n{'='*50}")
    log("Loading master CSVs...")
    log(f"{'='*50}")

    all_frames = []
    for _, u_name in universes:
        path = f"master_data_{u_name}.csv"
        if os.path.exists(path):
            u_df = pd.read_csv(path)
            u_df['Date'] = pd.to_datetime(u_df['Date'])
            log(f"  Loaded {path}: {len(u_df):,} rows")
            all_frames.append(u_df)
        else:
            log(f"  WARNING: {path} not found — universe will be missing")

    if not all_frames:
        log("FATAL: No master CSVs loaded. Aborting.")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)

    for col in ['Stock', 'Universe']:
        if col not in df.columns:
            log(f"FATAL: column '{col}' missing. Aborting.")
            return pd.DataFrame()

    df = df.dropna(subset=['Stock', 'Universe'])
    log(f"Master total: {len(df):,} rows | {df['Stock'].nunique()} stocks | latest: {df['Date'].max().date()}")

    log(f"\n{'='*50}")
    log("Calculating indicators...")
    log(f"{'='*50}")

    all_results = []
    for _, u_name in universes:
        log(f"  [{u_name}] processing...")
        u_df = df[df['Universe'] == u_name].copy()
        ok, skip = 0, 0
        for stock, grp in u_df.groupby('Stock', group_keys=False):
            try:
                all_results.append(calculate_indicators(grp.copy()))
                ok += 1
            except Exception as e:
                log(f"    Skipping {stock}: {e}")
                skip += 1
        log(f"  [{u_name}] done: {ok} OK | {skip} skipped")

    if not all_results:
        log("FATAL: No indicator results. Aborting.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    available_cols = [c for c in COLS if c in combined.columns]
    latest = (
        combined[available_cols]
        .sort_values('Date')
        .drop_duplicates(subset=['Stock', 'Universe'], keep='last')
        .reset_index(drop=True)
    )
    latest['Date'] = pd.to_datetime(latest['Date']).dt.strftime('%Y-%m-%d')

    log(f"\nSnapshot: {len(latest)} rows | {len(available_cols)} columns")
    log(str(latest.groupby('Universe')['Stock'].nunique()))

    log(f"\n{'='*50}")
    log("Pushing to Google Sheet...")
    log(f"{'='*50}")
    gc        = get_gspread_client()
    sh        = gc.open_by_key(SHEET_ID)
    worksheet = sh.get_worksheet(0)
    worksheet.clear()
    gd.set_with_dataframe(worksheet, latest)
    log(f"Done. {len(latest)} rows → '{sh.title}'")
    log(f"URL: https://docs.google.com/spreadsheets/d/{SHEET_ID}")

    return latest


if __name__ == "__main__":
    run()
