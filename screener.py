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

SHEET_ID = "1JWHOhfTFhS0345GC4KMGHYCa1F8YEdDk2Skb85R2p5U"

MASTER_PATH = "master_data.csv"

NIFTY100_URL = "https://drive.google.com/uc?id=1SbcUYzWZPEd2zhK1kkNndYVmkDskp9fp"
LARGEMIDCAP_URL = "https://drive.google.com/uc?id=1BzI5KjtkkQ2H-LvUNnFXJDAki5IslJUP"

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


# ── COLUMNS ───────────────────────────────────────────────────────────────────

COLS = [
    'Date', 'Stock', 'Universe',
    'Open', 'High', 'Low', 'Close', 'Volume',

    'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_10', 'EMA_13', 'EMA_20', 'EMA_50', 'EMA_200',

    'RSI_14',
    'MACD_line', 'MACD_signal', 'MACD_hist',

    'CCI_20',
    'MFI_14',

    'Prev_Close',
    'Returns',

    'Supertrend',
    'Supertrend_Signal',
]


# ── GOOGLE AUTH ───────────────────────────────────────────────────────────────

def get_gspread_client():

    creds_json = os.environ.get(
        "GOOGLE_SERVICE_ACCOUNT_JSON"
    )

    if creds_json:

        info = json.loads(creds_json)

    else:

        with open("service_account.json") as f:
            info = json.load(f)

    creds = Credentials.from_service_account_info(
        info,
        scopes=SCOPES
    )

    return gspread.authorize(creds)


# ── DATE ──────────────────────────────────────────────────────────────────────

def last_trading_day():

    ist = zoneinfo.ZoneInfo("Asia/Kolkata")

    now = datetime.now(ist)

    market_close = now.replace(
        hour=15,
        minute=30,
        second=0,
        microsecond=0
    )

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

    END_DATE = last_trading_day()

    FETCH_END = (
        datetime.strptime(END_DATE, '%Y-%m-%d')
        + timedelta(days=1)
    ).strftime('%Y-%m-%d')

    if os.path.exists(MASTER_PATH):

        existing = pd.read_csv(MASTER_PATH)

        existing['Date'] = pd.to_datetime(
            existing['Date']
        )

        last_date = existing[
            existing['Universe'] == universe_name
        ]['Date'].max()

        if pd.isna(last_date):

            start_date = "2021-01-01"

        else:

            start_date = (
                last_date + timedelta(days=1)
            ).strftime('%Y-%m-%d')

    else:

        start_date = "2021-01-01"

        existing = None

    if start_date > END_DATE:

        return (
            existing[
                existing['Universe'] == universe_name
            ]
            if existing is not None
            else pd.DataFrame()
        )

    all_data = []

    for stock in stocks:

        try:

            df = yf.download(
                stock,
                start=start_date,
                end=FETCH_END,
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):

                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()[
                [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]
            ]

            df["Stock"] = stock
            df["Universe"] = universe_name

            all_data.append(df)

        except Exception as e:

            print(f"Error {stock}: {e}")

    new_data = (
        pd.concat(all_data, ignore_index=True)
        if all_data
        else pd.DataFrame()
    )

    if existing is not None and not new_data.empty:

        combined = pd.concat(
            [existing, new_data],
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset=['Date', 'Stock', 'Universe']
        )

        combined.to_csv(
            MASTER_PATH,
            index=False
        )

        return combined[
            combined['Universe'] == universe_name
        ]

    if not new_data.empty:

        new_data.to_csv(
            MASTER_PATH,
            index=False
        )

    return new_data


# ── INDICATORS ────────────────────────────────────────────────────────────────

def calculate_indicators(data):

    data = data.sort_values('Date').copy()

    close = data['Close']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    # SMA
    for w in [20, 50, 100, 200]:

        data[f'SMA_{w}'] = (
            close.rolling(w).mean()
        )

    # EMA
    for w in [10, 13, 20, 50, 200]:

        data[f'EMA_{w}'] = close.ewm(
            span=w,
            adjust=False
        ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    rs = avg_gain / (avg_loss + 1e-10)

    data['RSI_14'] = (
        100 - (100 / (1 + rs))
    )

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data['MACD_line'] = ema12 - ema26

    data['MACD_signal'] = (
        data['MACD_line']
        .ewm(span=9, adjust=False)
        .mean()
    )

    data['MACD_hist'] = (
        data['MACD_line']
        - data['MACD_signal']
    )

    # CCI
    tp = (high + low + close) / 3

    tp_mean = tp.rolling(20).mean()

    tp_std = tp.rolling(20).std()

    data['CCI_20'] = (
        (tp - tp_mean)
        / (0.015 * tp_std + 1e-10)
    )

    # MFI
    typical = (high + low + close) / 3

    rmf = typical * vol

    pmf = rmf.where(
        typical > typical.shift(),
        0.0
    )

    nmf = rmf.where(
        typical < typical.shift(),
        0.0
    )

    mfr = (
        pmf.rolling(14).sum()
        / (nmf.rolling(14).sum() + 1e-10)
    )

    data['MFI_14'] = (
        100 - (100 / (1 + mfr))
    )

    # ATR
    hl = high - low

    hc = (high - close.shift()).abs()

    lc = (low - close.shift()).abs()

    tr = pd.concat(
        [hl, hc, lc],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        span=7,
        adjust=False
    ).mean()

    # Supertrend
    hl2 = (high + low) / 2

    upper = hl2 + 3 * atr

    lower = hl2 - 3 * atr

    supertrend = np.where(
        close > upper.shift(1),
        lower,
        upper
    )

    data['Supertrend'] = supertrend

    data['Supertrend_Signal'] = np.where(
        close > supertrend,
        'BUY',
        'SELL'
    )

    # Returns
    data['Prev_Close'] = close.shift(1)

    data['Returns'] = (
        close.pct_change() * 100
    )

    return data


# ── RUN ───────────────────────────────────────────────────────────────────────

def run(log=print):

    log("Downloading NIFTY100")

    download_universe(
        NIFTY100_URL,
        "NIFTY100"
    )

    log("Downloading LARGEMIDCAP250")

    download_universe(
        LARGEMIDCAP_URL,
        "NIFTY_LARGEMIDCAP250"
    )

    df = pd.read_csv(MASTER_PATH)

    df = df.dropna(
        subset=['Stock', 'Universe']
    )

    df['Date'] = pd.to_datetime(
        df['Date']
    )

    log(
        f"Master rows: {len(df):,}"
    )

    output_data = {}

    for u in df['Universe'].unique():

        log(f"Calculating {u}")

        u_df = df[
            df['Universe'] == u
        ].copy()

        output_data[u] = (
            u_df.groupby(
                ['Stock', 'Universe'],
                group_keys=False
            )
            .apply(
                calculate_indicators,
                include_groups=True
            )
        )

    combined = pd.concat(
        output_data.values(),
        ignore_index=True
    )

    available_cols = [
        c for c in COLS
        if c in combined.columns
    ]

    latest = (
        combined[available_cols]
        .sort_values('Date')
        .groupby(
            ['Stock', 'Universe'],
            group_keys=False
        )
        .apply(lambda x: x.iloc[-1])
        .reset_index(drop=True)
    )

    latest['Date'] = pd.to_datetime(
        latest['Date']
    ).dt.strftime('%Y-%m-%d')

    log(
        f"Snapshot: {len(latest)} rows"
    )

    # ── GOOGLE SHEETS ─────────────────────────────────

    gc = get_gspread_client()

    sh = gc.open_by_key(SHEET_ID)

    worksheet = sh.get_worksheet(0)

    worksheet.clear()

    gd.set_with_dataframe(
        worksheet,
        latest
    )

    log(
        f"Done. {len(latest)} rows pushed."
    )

    return latest


if __name__ == "__main__":

    run()
