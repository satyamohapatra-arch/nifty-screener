# screener.py
# ── SETUP ─────────────────────────────────────────────────────────────────────

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

SHEET_ID   = "1JWHOhfTFhS0345GC4KMGHYCa1F8YEdDk2Skb85R2p5U"
MASTER_PATH = "master_data.csv"
NIFTY100_URL     = "https://drive.google.com/uc?id=1SbcUYzWZPEd2zhK1kkNndYVmkDskp9fp"
LARGEMIDCAP_URL  = "https://drive.google.com/uc?id=1BzI5KjtkkQ2H-LvUNnFXJDAki5IslJUP"

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# ── COLUMNS ───────────────────────────────────────────────────────────────────

COLS = [
    'Date', 'Stock', 'Universe',
    'Open', 'High', 'Low', 'Close', 'Volume',
    # Trend / MAs
    'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_10', 'EMA_13', 'EMA_20', 'EMA_50', 'EMA_200',
    'HMA_20', 'KAMA_20', 'FRAMA',
    # Trend systems
    'Supertrend', 'Supertrend_Signal',
    'Parabolic_SAR',
    'Ichimoku_Tenkan', 'Ichimoku_Kijun',
    'Donchian_High', 'Donchian_Low',
    'ADX_14',
    # Momentum
    'RSI_14',
    'MACD_line', 'MACD_signal', 'MACD_hist',
    'Stoch_K', 'Stoch_D', 'Stoch_RSI',
    'CCI_20',
    'Williams_R',
    'ROC_12',
    'Ultimate_Oscillator',
    'CMO',
    'TRIX',
    'Schaff_Trend_Cycle',
    'Fisher_Transform',
    'Coppock_Curve',
    'Vortex_Pos', 'Vortex_Neg',
    'Elder_Bull_Power', 'Elder_Bear_Power',
    'RVI',
    'Mass_Index',
    # Volatility
    'ATR_14',
    'Volatility',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'Keltner_Upper', 'Keltner_Lower',
    'Spread',
    # Volume
    'MFI_14',
    'OBV',
    'VWAP',
    # Price derived
    'Gap',
    'Pivot_Point',
    '52W_High', '52W_Low',
    'Prev_Close',
    'Returns',
    'Log_Returns',
]

# ── GOOGLE AUTH ───────────────────────────────────────────────────────────────

def get_gspread_client():
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if creds_json:
        info = json.loads(creds_json)
    else:
        with open("service_account.json") as f:
            info = json.load(f)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

# ── DATE ──────────────────────────────────────────────────────────────────────

def last_trading_day():
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    day = now if now >= market_close else now - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.strftime('%Y-%m-%d')

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def download_universe(symbols_url, universe_name):
    stocks = [
        s + ".NS"
        for s in pd.read_csv(symbols_url)["Symbol"].tolist()
    ]

    END_DATE   = last_trading_day()
    # Use today+1 (in IST) as fetch end — NOT END_DATE+1.
    # yfinance for .NS stocks needs end to be strictly after the last desired date
    # on the exchange calendar. Using END_DATE+1 sometimes still misses the last bar
    # because yfinance pages by calendar day, not trading day.
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    FETCH_END = (datetime.now(ist) + timedelta(days=1)).strftime('%Y-%m-%d')

    if os.path.exists(MASTER_PATH):
        existing   = pd.read_csv(MASTER_PATH)
        existing['Date'] = pd.to_datetime(existing['Date']).dt.tz_localize(None).dt.normalize()
        last_date  = existing[existing['Universe'] == universe_name]['Date'].max()
        if pd.isna(last_date):
            start_date = "2021-01-01"
        else:
            # Always re-fetch the last stored date — it may have been saved with a
            # wrong date due to timezone issues, or may be a partial day's data.
            start_date = last_date.strftime('%Y-%m-%d')
            # Drop the last date's rows so we cleanly replace them
            existing = existing[existing['Date'] < last_date]
    else:
        start_date = "2021-01-01"
        existing   = None

    if start_date > END_DATE:
        print(f"  [{universe_name}] Already up-to-date (last={END_DATE}). Skipping download.")
        return (
            existing[existing['Universe'] == universe_name]
            if existing is not None
            else pd.DataFrame()
        )

    print(f"  [{universe_name}] Fetching {start_date} → {END_DATE} for {len(stocks)} stocks…")
    all_data = []
    for stock in stocks:
        try:
            df = yf.download(
                stock,
                start=start_date,
                end=FETCH_END,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            # Strip timezone (NSE returns Asia/Kolkata tz-aware timestamps).
            # Without this, CSV roundtrip converts to UTC, shifting dates back by ~5.5hrs
            # and turning e.g. 2026-06-03 00:00+05:30 into 2026-06-02.
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
            df["Stock"]    = stock
            df["Universe"] = universe_name
            all_data.append(df)
        except Exception as e:
            print(f"    Error {stock}: {e}")

    new_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    if existing is not None and not new_data.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'Stock', 'Universe'])
        combined.to_csv(MASTER_PATH, index=False)
        return combined[combined['Universe'] == universe_name]

    if existing is not None and new_data.empty:
        return existing[existing['Universe'] == universe_name]

    if not new_data.empty:
        new_data.to_csv(MASTER_PATH, index=False)
        return new_data

    return pd.DataFrame()

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def _sma(series, window):
    return series.rolling(window).mean()

def _atr(high, low, close, period=14):
    hl  = high - low
    hc  = (high - close.shift()).abs()
    lc  = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ── INDICATORS ────────────────────────────────────────────────────────────────

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data  = data.sort_values('Date').reset_index(drop=True).copy()

    # Force numeric dtypes — object dtype causes all rolling/ewm to silently return NaN
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows where Close is NaN (corrupt / missing bars)
    data = data.dropna(subset=['Close', 'High', 'Low', 'Open']).reset_index(drop=True)

    if len(data) < 2:
        return data

    close = data['Close'].astype(float)
    high  = data['High'].astype(float)
    low   = data['Low'].astype(float)
    vol   = data['Volume'].astype(float)
    n     = len(data)

    # ── Simple / Exponential MAs ──────────────────────────────────────────────
    for w in [20, 50, 100, 200]:
        data[f'SMA_{w}'] = _sma(close, w)

    for w in [10, 13, 20, 50, 200]:
        data[f'EMA_{w}'] = _ema(close, w)

    # ── HMA (Hull MA) — window 20 ─────────────────────────────────────────────
    wma_half = _sma(close, 10)          # WMA(n/2) approximated with SMA for simplicity
    wma_full = _sma(close, 20)
    raw_hma  = 2 * wma_half - wma_full
    data['HMA_20'] = _sma(raw_hma, int(np.sqrt(20)))

    # ── KAMA (Kaufman Adaptive MA) — window 20 ────────────────────────────────
    fast_sc = 2 / (2  + 1)
    slow_sc = 2 / (30 + 1)
    kama    = [close.iloc[0]] * n
    for i in range(1, n):
        direction = abs(close.iloc[i] - close.iloc[max(0, i - 20)])
        volatility = (close.diff().abs()).iloc[max(0, i - 20):i].sum()
        er   = direction / volatility if volatility != 0 else 0
        sc   = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close.iloc[i] - kama[i - 1])
    data['KAMA_20'] = kama

    # ── FRAMA (Fractal Adaptive MA) ───────────────────────────────────────────
    period = 16
    frama  = [close.iloc[0]] * n
    for i in range(period, n):
        h1 = high.iloc[i - period    : i - period // 2].max()
        l1 = low.iloc[i  - period    : i - period // 2].min()
        h2 = high.iloc[i - period // 2 : i].max()
        l2 = low.iloc[i  - period // 2 : i].min()
        h3 = high.iloc[i - period    : i].max()
        l3 = low.iloc[i  - period    : i].min()
        n1 = (h1 - l1) / (period / 2) if (h1 - l1) > 0 else 0
        n2 = (h2 - l2) / (period / 2) if (h2 - l2) > 0 else 0
        n3 = (h3 - l3) / period       if (h3 - l3) > 0 else 0
        if n1 > 0 and n2 > 0 and n3 > 0:
            dim = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        else:
            dim = 1.0
        alpha = np.exp(-4.6 * (dim - 1))
        alpha = max(0.01, min(1.0, alpha))
        frama[i] = alpha * close.iloc[i] + (1 - alpha) * frama[i - 1]
    data['FRAMA'] = frama

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12             = _ema(close, 12)
    ema26             = _ema(close, 26)
    data['MACD_line']   = ema12 - ema26
    data['MACD_signal'] = _ema(data['MACD_line'], 9)
    data['MACD_hist']   = data['MACD_line'] - data['MACD_signal']

    # ── Stochastic K / D ──────────────────────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    data['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()

    # ── Stochastic RSI ────────────────────────────────────────────────────────
    rsi        = data['RSI_14']
    rsi_min    = rsi.rolling(14).min()
    rsi_max    = rsi.rolling(14).max()
    data['Stoch_RSI'] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    # ── CCI ───────────────────────────────────────────────────────────────────
    tp      = (high + low + close) / 3
    tp_mean = tp.rolling(20).mean()
    tp_std  = tp.rolling(20).std()
    data['CCI_20'] = (tp - tp_mean) / (0.015 * tp_std + 1e-10)

    # ── MFI ───────────────────────────────────────────────────────────────────
    typical = (high + low + close) / 3
    rmf     = typical * vol
    pmf     = rmf.where(typical > typical.shift(), 0.0)
    nmf     = rmf.where(typical < typical.shift(), 0.0)
    mfr     = pmf.rolling(14).sum() / (nmf.rolling(14).sum() + 1e-10)
    data['MFI_14'] = 100 - (100 / (1 + mfr))

    # ── Williams %R ───────────────────────────────────────────────────────────
    data['Williams_R'] = -100 * (high.rolling(14).max() - close) / (
        high.rolling(14).max() - low.rolling(14).min() + 1e-10
    )

    # ── ROC 12 ───────────────────────────────────────────────────────────────
    data['ROC_12'] = close.pct_change(12) * 100

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr14 = _atr(high, low, close, 14)
    data['ATR_14'] = atr14

    # ── Volatility % ──────────────────────────────────────────────────────────
    data['Volatility'] = (close.pct_change().rolling(20).std() * np.sqrt(252) * 100)

    # ── Bollinger Bands (20, 2) ───────────────────────────────────────────────
    bb_mid             = _sma(close, 20)
    bb_std             = close.rolling(20).std()
    data['BB_Upper']   = bb_mid + 2 * bb_std
    data['BB_Middle']  = bb_mid
    data['BB_Lower']   = bb_mid - 2 * bb_std

    # ── Keltner Channels (20, 2×ATR10) ───────────────────────────────────────
    kc_mid                = _ema(close, 20)
    kc_atr                = _atr(high, low, close, 10)
    data['Keltner_Upper'] = kc_mid + 2 * kc_atr
    data['Keltner_Lower'] = kc_mid - 2 * kc_atr

    # ── Spread ────────────────────────────────────────────────────────────────
    data['Spread'] = high - low

    # ── Supertrend (fixed stateful implementation) ────────────────────────────
    atr7    = _atr(high, low, close, 7)
    hl2     = (high + low) / 2
    upper_b = (hl2 + 3 * atr7).values
    lower_b = (hl2 - 3 * atr7).values
    close_v = close.values

    final_upper = upper_b.copy()
    final_lower = lower_b.copy()
    st_signal   = [True] * n    # True = BUY

    for i in range(1, n):
        # Tighten bands: they can only move in one direction
        final_upper[i] = (
            min(upper_b[i], final_upper[i - 1])
            if close_v[i - 1] <= final_upper[i - 1]
            else upper_b[i]
        )
        final_lower[i] = (
            max(lower_b[i], final_lower[i - 1])
            if close_v[i - 1] >= final_lower[i - 1]
            else lower_b[i]
        )
        # Determine direction
        if close_v[i] > final_upper[i - 1]:
            st_signal[i] = True
        elif close_v[i] < final_lower[i - 1]:
            st_signal[i] = False
        else:
            st_signal[i] = st_signal[i - 1]

    data['Supertrend']        = [final_lower[i] if st_signal[i] else final_upper[i] for i in range(n)]
    data['Supertrend_Signal'] = ['BUY' if s else 'SELL' for s in st_signal]

    # ── Parabolic SAR ─────────────────────────────────────────────────────────
    af_start, af_step, af_max = 0.02, 0.02, 0.2
    psar   = close.iloc[0]
    ep     = high.iloc[0]
    af     = af_start
    uptrend = True
    psar_vals = [psar]

    for i in range(1, n):
        if uptrend:
            psar = psar + af * (ep - psar)
            psar = min(psar, low.iloc[i - 1], low.iloc[max(0, i - 2)])
            if low.iloc[i] < psar:
                uptrend = False
                psar    = ep
                ep      = low.iloc[i]
                af      = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            psar = psar + af * (ep - psar)
            psar = max(psar, high.iloc[i - 1], high.iloc[max(0, i - 2)])
            if high.iloc[i] > psar:
                uptrend = True
                psar    = ep
                ep      = high.iloc[i]
                af      = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
        psar_vals.append(psar)

    data['Parabolic_SAR'] = psar_vals

    # ── Ichimoku (Tenkan / Kijun) ─────────────────────────────────────────────
    data['Ichimoku_Tenkan'] = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    data['Ichimoku_Kijun']  = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # ── Donchian Channel (20) ─────────────────────────────────────────────────
    data['Donchian_High'] = high.rolling(20).max()
    data['Donchian_Low']  = low.rolling(20).min()

    # ── ADX 14 ───────────────────────────────────────────────────────────────
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_adx   = _atr(high, low, close, 14)
    plus_di   = 100 * _ema(plus_dm,  14) / (atr_adx + 1e-10)
    minus_di  = 100 * _ema(minus_dm, 14) / (atr_adx + 1e-10)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    data['ADX_14'] = dx.ewm(span=14, adjust=False).mean()

    # ── OBV ───────────────────────────────────────────────────────────────────
    direction     = np.sign(close.diff().fillna(0))
    data['OBV']   = (direction * vol).cumsum()

    # ── VWAP (rolling 20-bar) ─────────────────────────────────────────────────
    tp2          = (high + low + close) / 3
    data['VWAP'] = (tp2 * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)

    # ── Gap ───────────────────────────────────────────────────────────────────
    data['Gap'] = ((data['Open'] - close.shift()) / (close.shift() + 1e-10)) * 100

    # ── Pivot Point ───────────────────────────────────────────────────────────
    data['Pivot_Point'] = (high.shift() + low.shift() + close.shift()) / 3

    # ── 52-Week High / Low ────────────────────────────────────────────────────
    data['52W_High'] = high.rolling(252).max()
    data['52W_Low']  = low.rolling(252).min()

    # ── Returns ───────────────────────────────────────────────────────────────
    data['Prev_Close']  = close.shift(1)
    data['Returns']     = close.pct_change() * 100
    data['Log_Returns'] = np.log(close / (close.shift() + 1e-10))

    # ── Ultimate Oscillator ───────────────────────────────────────────────────
    bp    = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    tr_uo = pd.concat([high, close.shift()], axis=1).max(axis=1) - \
            pd.concat([low,  close.shift()], axis=1).min(axis=1)
    avg7  = bp.rolling(7).sum()  / (tr_uo.rolling(7).sum()  + 1e-10)
    avg14 = bp.rolling(14).sum() / (tr_uo.rolling(14).sum() + 1e-10)
    avg28 = bp.rolling(28).sum() / (tr_uo.rolling(28).sum() + 1e-10)
    data['Ultimate_Oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

    # ── CMO (Chande Momentum Oscillator) ─────────────────────────────────────
    diff       = close.diff()
    cmo_up     = diff.clip(lower=0).rolling(14).sum()
    cmo_down   = (-diff).clip(lower=0).rolling(14).sum()
    data['CMO'] = 100 * (cmo_up - cmo_down) / (cmo_up + cmo_down + 1e-10)

    # ── TRIX ─────────────────────────────────────────────────────────────────
    t1          = _ema(close, 15)
    t2          = _ema(t1, 15)
    t3          = _ema(t2, 15)
    data['TRIX'] = t3.pct_change() * 100

    # ── Schaff Trend Cycle ────────────────────────────────────────────────────
    macd_stc      = _ema(close, 23) - _ema(close, 50)
    stc_k         = 100 * (macd_stc - macd_stc.rolling(10).min()) / \
                    (macd_stc.rolling(10).max() - macd_stc.rolling(10).min() + 1e-10)
    stc_d         = _ema(stc_k, 3)
    data['Schaff_Trend_Cycle'] = stc_d.clip(0, 100)

    # ── Fisher Transform ──────────────────────────────────────────────────────
    median_price = (high + low) / 2
    highest      = median_price.rolling(9).max()
    lowest       = median_price.rolling(9).min()
    value        = 2 * ((median_price - lowest) / (highest - lowest + 1e-10)) - 1
    value        = value.clip(-0.999, 0.999)
    data['Fisher_Transform'] = 0.5 * np.log((1 + value) / (1 - value))

    # ── Coppock Curve ─────────────────────────────────────────────────────────
    roc11 = close.pct_change(11) * 100
    roc14 = close.pct_change(14) * 100
    data['Coppock_Curve'] = _ema(roc11 + roc14, 10)

    # ── Vortex Indicator ─────────────────────────────────────────────────────
    vm_plus  = (high - low.shift()).abs()
    vm_minus = (low  - high.shift()).abs()
    tr_v     = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    data['Vortex_Pos'] = vm_plus.rolling(14).sum()  / (tr_v.rolling(14).sum() + 1e-10)
    data['Vortex_Neg'] = vm_minus.rolling(14).sum() / (tr_v.rolling(14).sum() + 1e-10)

    # ── Elder Ray ─────────────────────────────────────────────────────────────
    ema13_elder             = _ema(close, 13)
    data['Elder_Bull_Power'] = high - ema13_elder
    data['Elder_Bear_Power'] = low  - ema13_elder

    # ── RVI (Relative Vigor Index) ────────────────────────────────────────────
    num = (close - data['Open']) + 2 * (close.shift(1) - data['Open'].shift(1)) + \
          2 * (close.shift(2) - data['Open'].shift(2)) + (close.shift(3) - data['Open'].shift(3))
    den = (high - low) + 2 * (high.shift(1) - low.shift(1)) + \
          2 * (high.shift(2) - low.shift(2)) + (high.shift(3) - low.shift(3))
    data['RVI'] = num.rolling(10).mean() / (den.rolling(10).mean() + 1e-10)

    # ── Mass Index ────────────────────────────────────────────────────────────
    ema_hl   = _ema(high - low, 9)
    ema2_hl  = _ema(ema_hl, 9)
    data['Mass_Index'] = (ema_hl / (ema2_hl + 1e-10)).rolling(25).sum()

    return data

# ── RUN ───────────────────────────────────────────────────────────────────────

def run(log=print):
    log("Downloading NIFTY100")
    download_universe(NIFTY100_URL, "NIFTY100")

    log("Downloading LARGEMIDCAP250")
    download_universe(LARGEMIDCAP_URL, "NIFTY_LARGEMIDCAP250")

    df = pd.read_csv(MASTER_PATH)
    df = df.dropna(subset=['Stock', 'Universe'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    log(f"Master rows: {len(df):,}")

    all_latest = []
    for u in df['Universe'].unique():
        log(f"Calculating indicators: {u}")
        u_df = df[df['Universe'] == u].copy()
        for stock, stock_df in u_df.groupby('Stock'):
            try:
                result = calculate_indicators(stock_df.copy())
                if not result.empty:
                    all_latest.append(result.iloc[-1])
            except Exception as e:
                log(f"  Warning: skipped {stock} — {e}")

    combined = pd.DataFrame(all_latest).reset_index(drop=True)

    available_cols = [c for c in COLS if c in combined.columns]
    latest = combined[available_cols].reset_index(drop=True)
    latest['Date'] = pd.to_datetime(latest['Date']).dt.strftime('%Y-%m-%d')

    # Stamp every row with the IST time this screener run produced the data
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    run_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M IST')
    latest['Last_Run_IST'] = run_time_ist

    log(f"Snapshot: {len(latest)} rows, {len(available_cols)} columns")
    log(f"Data date: {latest['Date'].max()}  |  Run time: {run_time_ist}")

    # ── Push to Google Sheets ──────────────────────────────────────────────────
    gc        = get_gspread_client()
    sh        = gc.open_by_key(SHEET_ID)
    worksheet = sh.get_worksheet(0)
    worksheet.clear()
    gd.set_with_dataframe(worksheet, latest)

    log(f"Done. {len(latest)} rows pushed to Google Sheets.")
    return latest


if __name__ == "__main__":
    run()
