"""
screener.py
Fetches OHLCV data, calculates all indicators, pushes to Google Sheet.
Run directly: python screener.py
"""

import os, warnings, json
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import gspread_dataframe as gd
import zoneinfo
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SHEET_ID    = "1JWHOhfTFhS0345GC4KMGHYCa1F8YEdDk2Skb85R2p5U"
MASTER_PATH = "master_data.csv"

NIFTY100_URL    = "https://drive.google.com/uc?id=1SbcUYzWZPEd2zhK1kkNndYVmkDskp9fp"
LARGEMIDCAP_URL = "https://drive.google.com/uc?id=1BzI5KjtkkQ2H-LvUNnFXJDAki5IslJUP"

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

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
    'CCI_20', 'Williams_R', 'ROC_12', 'Ultimate_Oscillator',
    'ATR_14',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'Keltner_Upper', 'Keltner_Lower',
    'OBV', 'VWAP', 'MFI_14', 'Pivot_Point',
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


# ── AUTH ──────────────────────────────────────────────────────────────────────
def get_gspread_client():
    """Auth via service account JSON stored in env var or file."""
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if creds_json:
        info = json.loads(creds_json)
    else:
        # fallback: local file (for local dev)
        with open("service_account.json") as f:
            info = json.load(f)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


# ── DATE HELPERS ──────────────────────────────────────────────────────────────
def last_trading_day():
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    day = now if now >= market_close else now - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.strftime('%Y-%m-%d')


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
def download_universe(symbols_url: str, universe_name: str, end_date: str, fetch_end: str):
    stocks = [s + ".NS" for s in pd.read_csv(symbols_url)["Symbol"].tolist()]

    existing = None
    if os.path.exists(MASTER_PATH):
        existing = pd.read_csv(MASTER_PATH)
        existing['Date'] = pd.to_datetime(existing['Date'])
        last_date = existing[existing['Universe'] == universe_name]['Date'].max()
        if pd.isna(last_date):
            start_date = "2021-01-01"
        else:
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = "2021-01-01"

    print(f"{universe_name} — fetching from {start_date}")

    if start_date > end_date:
        print(f"{universe_name} already up to date.")
        return existing[existing['Universe'] == universe_name] if existing is not None else pd.DataFrame()

    all_data = []
    for stock in stocks:
        try:
            df = yf.download(stock, start=start_date, end=fetch_end,
                             interval="1d", auto_adjust=False, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df["Stock"] = stock
            df["Universe"] = universe_name
            all_data.append(df)
        except Exception as e:
            print(f"  Error {stock}: {e}")

    new_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    if existing is not None and not new_data.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'Stock', 'Universe'])
        combined.to_csv(MASTER_PATH, index=False)
        return combined[combined['Universe'] == universe_name]

    if not new_data.empty:
        new_data.to_csv(MASTER_PATH, index=False)
    return new_data


# ── INDICATORS ────────────────────────────────────────────────────────────────
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values('Date').copy()
    n = len(data)
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

    # WMA helper
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # HMA_20
    half_wma = wma(close, 10)
    full_wma = wma(close, 20)
    hull_raw = 2 * half_wma - full_wma
    data['HMA_20'] = wma(hull_raw, int(np.sqrt(20)))

    # KAMA_20
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    kama = close.copy().astype(float)
    for i in range(20, n):
        direction  = abs(close.iloc[i] - close.iloc[i - 20])
        volatility = (close.diff().abs()).iloc[i - 19:i + 1].sum()
        er  = direction / (volatility + 1e-10)
        sc  = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama.iloc[i] = kama.iloc[i - 1] + sc * (close.iloc[i] - kama.iloc[i - 1])
    kama.iloc[:20] = np.nan
    data['KAMA_20'] = kama

    # Ichimoku
    data['Ichimoku_Tenkan'] = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    data['Ichimoku_Kijun']  = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # Donchian
    data['Donchian_High'] = high.rolling(20).max()
    data['Donchian_Low']  = low.rolling(20).min()

    # ATR + ADX
    up   = high.diff()
    down = low.shift() - low
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0), index=data.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0), index=data.index)
    hl   = high - low
    hcp  = (high - close.shift()).abs()
    lcp  = (low  - close.shift()).abs()
    tr   = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    data['ATR_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    atr_s = data['ATR_14'].replace(0, 1e-10)
    pdi   = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    mdi   = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    data['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Parabolic SAR
    af_start, af_step, af_max = 0.02, 0.02, 0.2
    high_arr = high.values; low_arr = low.values
    sar = np.zeros(n); trend_arr = np.ones(n)
    ep  = np.zeros(n); af_arr    = np.zeros(n)
    sar[0]=low_arr[0]; ep[0]=high_arr[0]; af_arr[0]=af_start
    for i in range(1, n):
        ps, pe, pa, pt = sar[i-1], ep[i-1], af_arr[i-1], trend_arr[i-1]
        if pt == 1:
            sar[i] = ps + pa * (pe - ps)
            sar[i] = min(sar[i], low_arr[i-1], low_arr[i-2] if i > 1 else low_arr[i-1])
            if low_arr[i] < sar[i]:
                trend_arr[i]=-1; sar[i]=pe; ep[i]=low_arr[i]; af_arr[i]=af_start
            else:
                trend_arr[i]=1; ep[i]=max(pe, high_arr[i])
                af_arr[i]=min(af_max, pa+af_step) if ep[i]>pe else pa
        else:
            sar[i] = ps + pa * (pe - ps)
            sar[i] = max(sar[i], high_arr[i-1], high_arr[i-2] if i > 1 else high_arr[i-1])
            if high_arr[i] > sar[i]:
                trend_arr[i]=1; sar[i]=pe; ep[i]=high_arr[i]; af_arr[i]=af_start
            else:
                trend_arr[i]=-1; ep[i]=min(pe, low_arr[i])
                af_arr[i]=min(af_max, pa+af_step) if ep[i]<pe else pa
    data['Parabolic_SAR'] = np.round(sar, 2)

    # Supertrend
    atr7       = tr.ewm(span=7, adjust=False).mean()
    hl2        = (high + low) / 2
    upper_band = (hl2 + 3 * atr7).values
    lower_band = (hl2 - 3 * atr7).values
    close_arr  = close.values
    supertrend = np.zeros(n); signal = [''] * n
    supertrend[0]=upper_band[0]; signal[0]='SELL'
    for i in range(1, n):
        if close_arr[i] > supertrend[i-1]:
            supertrend[i]=lower_band[i]; signal[i]='BUY'
        else:
            supertrend[i]=upper_band[i]; signal[i]='SELL'
    data['Supertrend']        = np.round(supertrend, 2)
    data['Supertrend_Signal'] = signal

    # RSI
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

    # CCI
    tp2     = (high + low + close) / 3
    tp_mean = tp2.rolling(20).mean()
    tp_std  = tp2.rolling(20).std()
    data['CCI_20'] = (tp2 - tp_mean) / (0.015 * tp_std + 1e-10)

    # Williams %R
    data['Williams_R'] = -100 * (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min() + 1e-10)

    # ROC
    data['ROC_12'] = close.pct_change(12) * 100

    # Ultimate Oscillator
    prev_close = close.shift(1)
    bp  = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr2 = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
    avg7  = bp.rolling(7).sum()  / (tr2.rolling(7).sum()  + 1e-10)
    avg14 = bp.rolling(14).sum() / (tr2.rolling(14).sum() + 1e-10)
    avg28 = bp.rolling(28).sum() / (tr2.rolling(28).sum() + 1e-10)
    data['Ultimate_Oscillator'] = 100 * (4*avg7 + 2*avg14 + avg28) / 7

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

    # Keltner
    kc_mid = close.ewm(span=20, adjust=False).mean()
    data['Keltner_Upper'] = kc_mid + 1.5 * data['ATR_14']
    data['Keltner_Lower'] = kc_mid - 1.5 * data['ATR_14']

    # OBV
    obv = np.zeros(n)
    for i in range(1, n):
        if close_arr[i] > close.values[i-1]:
            obv[i] = obv[i-1] + vol.values[i]
        elif close_arr[i] < close.values[i-1]:
            obv[i] = obv[i-1] - vol.values[i]
        else:
            obv[i] = obv[i-1]
    data['OBV'] = obv

    # VWAP
    tp_vwap = (high + low + close) / 3
    data['VWAP'] = (tp_vwap * vol).rolling(14).sum() / (vol.rolling(14).sum() + 1e-10)

    # MFI
    tp3  = (high + low + close) / 3
    rmf  = tp3 * vol
    pmf  = rmf.where(tp3 > tp3.shift(), 0.0)
    nmf  = rmf.where(tp3 < tp3.shift(), 0.0)
    mfr  = pmf.rolling(14).sum() / (nmf.rolling(14).sum() + 1e-10)
    data['MFI_14'] = 100 - (100 / (1 + mfr))

    # Pivot
    data['Pivot_Point'] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3

    # 52W
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
    w = 16
    for i in range(w, n):
        seg  = close.iloc[i-w:i]
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
        frama.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * frama.iloc[i-1]
    frama.iloc[:w] = np.nan
    data['FRAMA'] = frama

    # Coppock
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
    co    = close - open_
    hl_r  = high - low
    rvi_num = (co + 2*co.shift(1) + 2*co.shift(2) + co.shift(3)) / 6
    rvi_den = (hl_r + 2*hl_r.shift(1) + 2*hl_r.shift(2) + hl_r.shift(3)) / 6
    data['RVI'] = rvi_num.rolling(10).sum() / (rvi_den.rolling(10).sum() + 1e-10)

    # Derived
    data['Prev_Close']  = close.shift(1)
    data['Gap']         = open_ - close.shift(1)
    data['Returns']     = close.pct_change() * 100
    data['Log_Returns'] = np.log(close / (close.shift(1) + 1e-10))
    data['Spread']      = high - low
    data['Volatility']  = data['Returns'].rolling(20).std()

    return data


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(log=print):
    end_date  = last_trading_day()
    fetch_end = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    log(f"▶ Target date: {end_date}")

    log("⬇  Downloading NIFTY100...")
    download_universe(NIFTY100_URL,    "NIFTY100",             end_date, fetch_end)
    log("⬇  Downloading LARGEMIDCAP250...")
    download_universe(LARGEMIDCAP_URL, "NIFTY_LARGEMIDCAP250", end_date, fetch_end)

    log("📊 Calculating indicators...")
    df = pd.read_csv(MASTER_PATH)
    df = df.dropna(subset=['Stock', 'Universe'])
    df['Date'] = pd.to_datetime(df['Date'])

    output_data = {}
    for u in df['Universe'].unique():
        log(f"   {u}...")
        u_df = df[df['Universe'] == u].copy()
        output_data[u] = (
            u_df.groupby(['Stock', 'Universe'], group_keys=False)
                 .apply(calculate_indicators, include_groups=True)
        )

    combined = pd.concat(output_data.values(), ignore_index=True)
    available_cols = [c for c in COLS if c in combined.columns]
    latest = (
        combined[available_cols]
        .sort_values('Date')
        .groupby(['Stock', 'Universe'], group_keys=False)
        .apply(lambda x: x.iloc[-1])
        .reset_index(drop=True)
    )
    latest['Date'] = pd.to_datetime(latest['Date']).dt.strftime('%Y-%m-%d')
    log(f"✅ Snapshot: {len(latest)} rows | {len(available_cols)} columns")

    log("☁  Pushing to Google Sheet...")
    gc        = get_gspread_client()
    sh        = gc.open_by_key(SHEET_ID)
    worksheet = sh.get_worksheet(0)
    worksheet.clear()
    gd.set_with_dataframe(worksheet, latest)
    log(f"✅ Done — {len(latest)} rows pushed")
    return latest


if __name__ == "__main__":
    run()
