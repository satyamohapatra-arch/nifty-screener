"""
screener.py
Fetches OHLCV data, calculates all indicators, pushes to Google Sheet.
Run directly: python screener.py
"""

import os
import warnings
import json

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

SHEET_ID = "1JWHOhfTFhS0345GC4KMGHYCa1F8YEdDk2Skb85R2p5U"
MASTER_PATH = "master_data.csv"

NIFTY100_URL = "https://drive.google.com/uc?id=1SbcUYzWZPEd2zhK1kkNndYVmkDskp9fp"
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
        with open("service_account.json") as f:
            info = json.load(f)

    creds = Credentials.from_service_account_info(
        info,
        scopes=SCOPES
    )

    return gspread.authorize(creds)


# ── DATE HELPERS ──────────────────────────────────────────────────────────────

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

def download_universe(
    symbols_url: str,
    universe_name: str,
    end_date: str,
    fetch_end: str
):

    stocks = [
        s + ".NS"
        for s in pd.read_csv(symbols_url)["Symbol"].tolist()
    ]

    existing = None

    if os.path.exists(MASTER_PATH):

        existing = pd.read_csv(MASTER_PATH)

        existing['Date'] = pd.to_datetime(existing['Date'])

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

    print(f"{universe_name} — fetching from {start_date}")

    if start_date > end_date:

        print(f"{universe_name} already up to date.")

        return (
            existing[existing['Universe'] == universe_name]
            if existing is not None
            else pd.DataFrame()
        )

    all_data = []

    for stock in stocks:

        try:

            df = yf.download(
                stock,
                start=start_date,
                end=fetch_end,
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()[
                ["Date", "Open", "High", "Low", "Close", "Volume"]
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

        combined.to_csv(MASTER_PATH, index=False)

        return combined[
            combined['Universe'] == universe_name
        ]

    if not new_data.empty:
        new_data.to_csv(MASTER_PATH, index=False)

    return new_data


# ── INDICATORS ────────────────────────────────────────────────────────────────

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:

    data = data.sort_values('Date').copy()

    close = data['Close']

    for w in [20, 50, 100, 200]:
        data[f'SMA_{w}'] = close.rolling(w).mean()

    for w in [10, 13, 20, 50, 200]:
        data[f'EMA_{w}'] = close.ewm(
            span=w,
            adjust=False
        ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    data['RSI_14'] = 100 - (
        100 / (1 + avg_gain / (avg_loss + 1e-10))
    )

    ema12 = close.ewm(span=12, adjust=False).mean()

    ema26 = close.ewm(span=26, adjust=False).mean()

    data['MACD_line'] = ema12 - ema26

    data['MACD_signal'] = data['MACD_line'].ewm(
        span=9,
        adjust=False
    ).mean()

    data['MACD_hist'] = (
        data['MACD_line'] - data['MACD_signal']
    )

    data['Prev_Close'] = close.shift(1)

    data['Returns'] = close.pct_change() * 100

    return data


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(log=print):

    end_date = last_trading_day()

    fetch_end = (
        datetime.strptime(end_date, '%Y-%m-%d')
        + timedelta(days=1)
    ).strftime('%Y-%m-%d')

    log(f"▶ Target date: {end_date}")

    log("⬇ Downloading NIFTY100...")

    download_universe(
        NIFTY100_URL,
        "NIFTY100",
        end_date,
        fetch_end
    )

    log("⬇ Downloading LARGEMIDCAP250...")

    download_universe(
        LARGEMIDCAP_URL,
        "NIFTY_LARGEMIDCAP250",
        end_date,
        fetch_end
    )

    log("📊 Calculating indicators...")

    # ── SAFE CACHE LOADING ──────────────────────────────────────

    if os.path.exists(MASTER_PATH):

        try:

            df = pd.read_csv(MASTER_PATH)

            required_cols = [
                'Stock',
                'Universe',
                'Date'
            ]

            if not all(
                col in df.columns
                for col in required_cols
            ):

                raise ValueError(
                    f"Corrupted cache. Missing columns: "
                    f"{[c for c in required_cols if c not in df.columns]}"
                )

            df = df.dropna(
                subset=['Stock', 'Universe']
            )

            if df.empty:
                raise ValueError(
                    "Cache became empty after cleanup."
                )

            df['Date'] = pd.to_datetime(df['Date'])

        except Exception as e:

            log(f"⚠ Corrupted cache detected: {e}")

            log(
                "♻ Removing corrupted master_data.csv "
                "and rebuilding..."
            )

            os.remove(MASTER_PATH)

            log("⬇ Re-downloading NIFTY100...")

            download_universe(
                NIFTY100_URL,
                "NIFTY100",
                end_date,
                fetch_end
            )

            log("⬇ Re-downloading LARGEMIDCAP250...")

            download_universe(
                LARGEMIDCAP_URL,
                "NIFTY_LARGEMIDCAP250",
                end_date,
                fetch_end
            )

            df = pd.read_csv(MASTER_PATH)

            df = df.dropna(
                subset=['Stock', 'Universe']
            )

            df['Date'] = pd.to_datetime(df['Date'])

    else:

        raise FileNotFoundError(
            "master_data.csv was not created correctly."
        )

    output_data = {}

    for u in df['Universe'].unique():

        log(f"   {u}...")

        u_df = df[
            df['Universe'] == u
        ].copy()

        output_data[u] = (
            u_df.groupby(
                ['Stock', 'Universe'],
                group_keys=False
            )
            .apply(calculate_indicators)
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
        f"✅ Snapshot: {len(latest)} rows | "
        f"{len(available_cols)} columns"
    )

    log("☁ Pushing to Google Sheet...")

    gc = get_gspread_client()

    sh = gc.open_by_key(SHEET_ID)

    worksheet = sh.get_worksheet(0)

    worksheet.clear()

    gd.set_with_dataframe(
        worksheet,
        latest
    )

    log(f"✅ Done — {len(latest)} rows pushed")

    return latest


if __name__ == "__main__":
    run()
