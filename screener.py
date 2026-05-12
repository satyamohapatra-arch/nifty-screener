"""
screener.py
Stable version for Streamlit screener.
Downloads OHLCV data, calculates indicators, pushes to Google Sheets.
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import gspread_dataframe as gd
import zoneinfo

from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials


# ── CONSTANTS ────────────────────────────────────────────────────────────────

SHEET_ID = "1JWHOhfTFhS0345GC4KMGHYCa1F8YEdDk2Skb85R2p5U"

MASTER_PATH = "master_data.csv"

NIFTY100_URL = "https://drive.google.com/uc?id=1SbcUYzWZPEd2zhK1kkNndYVmkDskp9fp"
LARGEMIDCAP_URL = "https://drive.google.com/uc?id=1BzI5KjtkkQ2H-LvUNnFXJDAki5IslJUP"

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

BASE_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Stock",
    "Universe",
]

COLS = [
    "Date",
    "Stock",
    "Universe",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "RSI_14",
    "MFI_14",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    "Prev_Close",
    "Returns",
    "Supertrend_Signal",
]


# ── AUTH ─────────────────────────────────────────────────────────────────────

def get_gspread_client():

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


# ── DATE HELPERS ─────────────────────────────────────────────────────────────

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

    return day.strftime("%Y-%m-%d")


# ── DOWNLOAD ────────────────────────────────────────────────────────────────

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

        try:

            existing = pd.read_csv(MASTER_PATH)

            required_cols = [
                "Date",
                "Stock",
                "Universe"
            ]

            if not all(
                col in existing.columns
                for col in required_cols
            ):
                raise ValueError("Corrupted cache")

            existing["Date"] = pd.to_datetime(
                existing["Date"]
            )

            last_date = existing[
                existing["Universe"] == universe_name
            ]["Date"].max()

            if pd.isna(last_date):

                start_date = "2021-01-01"

            else:

                start_date = (
                    last_date + timedelta(days=1)
                ).strftime("%Y-%m-%d")

        except Exception:

            if os.path.exists(MASTER_PATH):
                os.remove(MASTER_PATH)

            existing = None

            start_date = "2021-01-01"

    else:

        start_date = "2021-01-01"

    print(f"{universe_name} fetching from {start_date}")

    if start_date > end_date:

        return (
            existing[existing["Universe"] == universe_name]
            if existing is not None
            else pd.DataFrame(columns=BASE_COLUMNS)
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

            required_yf_cols = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            if not all(
                col in df.columns
                for col in required_yf_cols
            ):
                continue

            df = df.reset_index()[
                ["Date", "Open", "High", "Low", "Close", "Volume"]
            ]

            df["Stock"] = stock
            df["Universe"] = universe_name

            all_data.append(df)

        except Exception as e:

            print(f"{stock}: {e}")

    new_data = (
        pd.concat(all_data, ignore_index=True)
        if all_data
        else pd.DataFrame(columns=BASE_COLUMNS)
    )

    if existing is not None:

        combined = pd.concat(
            [existing, new_data],
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset=["Date", "Stock", "Universe"]
        )

        combined.to_csv(MASTER_PATH, index=False)

        return combined[
            combined["Universe"] == universe_name
        ]

    else:

        new_data.to_csv(MASTER_PATH, index=False)

        return new_data


# ── INDICATORS ───────────────────────────────────────────────────────────────

def calculate_indicators(data: pd.DataFrame):

    data = data.sort_values("Date").copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # SMA
    for w in [20, 50, 100, 200]:

        data[f"SMA_{w}"] = (
            close.rolling(w).mean()
        )

    # EMA
    for w in [10, 20, 50, 200]:

        data[f"EMA_{w}"] = close.ewm(
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

    data["RSI_14"] = 100 - (100 / (1 + rs))

    # MFI
    typical = (high + low + close) / 3

    money_flow = typical * volume

    positive_flow = money_flow.where(
        typical > typical.shift(1),
        0
    )

    negative_flow = money_flow.where(
        typical < typical.shift(1),
        0
    )

    positive_mf = positive_flow.rolling(14).sum()
    negative_mf = negative_flow.rolling(14).sum()

    mfr = positive_mf / (negative_mf + 1e-10)

    data["MFI_14"] = 100 - (100 / (1 + mfr))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD_line"] = ema12 - ema26

    data["MACD_signal"] = data[
        "MACD_line"
    ].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_hist"] = (
        data["MACD_line"]
        - data["MACD_signal"]
    )

    # Returns
    data["Prev_Close"] = close.shift(1)

    data["Returns"] = (
        close.pct_change() * 100
    )

    # Simple trend signal
    data["Supertrend_Signal"] = np.where(
        close > data["EMA_20"],
        "BUY",
        "SELL"
    )

    return data


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run(log=print):

    end_date = last_trading_day()

    fetch_end = (
        datetime.strptime(end_date, "%Y-%m-%d")
        + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    log(f"Target date: {end_date}")

    log("Downloading NIFTY100")

    download_universe(
        NIFTY100_URL,
        "NIFTY100",
        end_date,
        fetch_end
    )

    log("Downloading LARGEMIDCAP250")

    download_universe(
        LARGEMIDCAP_URL,
        "NIFTY_LARGEMIDCAP250",
        end_date,
        fetch_end
    )

    if not os.path.exists(MASTER_PATH):

        raise FileNotFoundError(
            f"{MASTER_PATH} missing"
        )

    df = pd.read_csv(MASTER_PATH)

    if df.empty:

        raise ValueError(
            "No data downloaded"
        )

    required_cols = [
        "Date",
        "Stock",
        "Universe"
    ]

    if not all(
        col in df.columns
        for col in required_cols
    ):
        raise ValueError(
            f"Missing columns: {required_cols}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    output_data = {}

    for u in df["Universe"].unique():

        log(f"Processing {u}")

        u_df = df[
            df["Universe"] == u
        ].copy()

        if u_df.empty:
            continue

        processed = (
            u_df.groupby(
                ["Stock", "Universe"],
                as_index=False,
                group_keys=False
            )
            .apply(calculate_indicators)
            .reset_index(drop=True)
        )

        output_data[u] = processed

    if not output_data:

        raise ValueError(
            "No processed data"
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
        .sort_values("Date")
        .groupby(
            ["Stock", "Universe"],
            as_index=False,
            group_keys=False
        )
        .tail(1)
        .reset_index(drop=True)
    )

    latest["Date"] = pd.to_datetime(
        latest["Date"]
    ).dt.strftime("%Y-%m-%d")

    log(
        f"Snapshot: {len(latest)} rows"
    )

    # ── PUSH TO GOOGLE SHEETS ──────────────────────────

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
