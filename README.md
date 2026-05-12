# Nifty Live Screener — Setup Guide

## Files
```
nifty-screener/
├── app.py                          # Streamlit UI
├── screener.py                     # Indicator engine + Google Sheet push
├── requirements.txt
├── .streamlit/config.toml          # Dark theme
└── .github/workflows/
    └── daily_screener.yml          # GitHub Actions cron (15:31 IST weekdays)
```

---

## Step 1 — Google Service Account

1. Go to https://console.cloud.google.com
2. Create project → **APIs & Services** → **Enable APIs**:
   - Google Sheets API
   - Google Drive API
3. **Credentials** → **Create Credentials** → **Service Account**
4. Give it a name, click through
5. **Keys** tab → **Add Key** → JSON → download `service_account.json`
6. Copy the `client_email` from that JSON
7. Open your Google Sheet → Share → paste that email → **Editor** access

---

## Step 2 — GitHub Repo

```bash
git init
git add .
git commit -m "init"
git remote add origin https://github.com/YOUR_USERNAME/nifty-screener.git
git push -u origin main
```

Then add the secret:
- Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
- Name: `GOOGLE_SERVICE_ACCOUNT_JSON`
- Value: paste the **entire contents** of `service_account.json`

---

## Step 3 — Deploy on Streamlit Cloud (free)

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. **New app** → pick your repo → branch: `main` → file: `app.py`
4. **Advanced settings** → **Secrets**:

```toml
GOOGLE_SERVICE_ACCOUNT_JSON = '''
{
  "type": "service_account",
  "project_id": "...",
  ...paste entire JSON here...
}
'''
```

5. Deploy → you get a public URL like `https://yourname-nifty-screener.streamlit.app`

---

## Step 4 — Daily Auto-Run

The GitHub Actions workflow (`.github/workflows/daily_screener.yml`) runs automatically:
- **Every weekday at 15:31 IST** (10:01 UTC)
- You can also trigger it manually: Repo → **Actions** → **Daily Nifty Screener** → **Run workflow**

---

## Running Locally (optional)

```bash
pip install -r requirements.txt

# Place service_account.json in project root
streamlit run app.py
```

Or trigger screener manually:
```bash
python screener.py
```

---

## Notes
- Screener writes data to Google Sheet; Streamlit app reads from it (cached 5 min)
- First run downloads from 2021-01-01; subsequent runs are incremental
- `master_data.csv` is the local OHLCV cache — add it to `.gitignore`
