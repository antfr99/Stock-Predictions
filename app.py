"""
Weekly Stock Prediction — Streamlit
===================================

Streamlit port of the Gradio / Hugging Face app.

Identical modelling pipeline to the original:
  • 6 years of weekly (W-FRI) OHLCV from yfinance
  • Same feature set: Close, Delta_Close, Pct_Change_Close, Volume,
    Delta_Volume, RSI14, Delta_RSI14, MA50, MA200, P/E
  • Returns-based Beta vs SPY (W-FRI anchored, 1y+ overlap required)
  • XGBoost walk-forward validation (n=100, depth=4, lr=0.05, seed=42)
    on the last 20% of history, then a final model on the full history
    for the next-week forecast
  • VADER sentiment with the same financial lexicon overlay and the
    same ticker-collision filtering
  • Same 3-point signal score (sentiment + predicted direction + trend)

Removed vs the original:
  • Supabase storage, stored prediction history, evaluation, grading,
    "was it correct?" tracking — this app only looks forward.
  • Fixed ecosystem / sector filter lists — any ticker can be typed in.
"""

import os
import re
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
import xgboost as xgb
from ta.momentum import RSIIndicator

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ══════════════════════════════════════════════════════════════════
# 0.  Page setup — white background
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Weekly Stock Prediction",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp, [data-testid="stHeader"], [data-testid="stSidebar"] > div:first-child {
          background-color: #ffffff;
      }
      [data-testid="stSidebar"] { border-right: 1px solid #e6e6e6; }
      [data-testid="stSidebar"] button[kind="secondary"] {
          background-color: #f0f0f0;
          border-color: #dcdcdc;
      }
      [data-testid="stSidebar"] button[kind="secondary"]:hover {
          background-color: #e6e6e6;
          border-color: #c9c9c9;
      }
      .block-container { padding-top: 2.2rem; max-width: 1500px; }
      h1, h2, h3 { color: #111111; letter-spacing: -0.01em; }
      [data-testid="stMetricValue"] { font-size: 1.45rem; }
      div[data-testid="stDataFrame"] { border: 1px solid #ececec; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Matplotlib defaults: white figure + white axes, no dark surprises.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#d0d0d0",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.8,
    "font.size": 10,
})


# ══════════════════════════════════════════════════════════════════
# 1.  Configuration (same constants as the original app)
# ══════════════════════════════════════════════════════════════════

DEFAULT_TICKERS = ["NVDA", "AMD", "TSM", "AVGO", "ASML"]

CORRECT_THRESHOLD = 0.02        # used by the walk-forward hit-rate stat
MIN_TRAIN_SIZE    = 50          # min weekly rows before the model will train

# Volatility-adjusted threshold (kept: it's a per-ticker "how close does a
# forecast have to be" bar, useful to display even without stored outcomes).
VOL_THRESHOLD_MULTIPLIER = 1.0
MIN_THRESHOLD_PCT        = 3.0
MAX_THRESHOLD_PCT        = 20.0

HISTORY_YEARS = 6               # 6y so the 200-week MA can populate
MAX_WORKERS   = 4

DEFAULT_NEWSAPI_KEY = "210ba50713b74c3e900a5154d843c11e"

_DIRECTION_ARROWS = {"Up": "🔼 Up", "Down": "🔽 Down", "Flat": "➡️ Flat"}


# ── News query configuration (verbatim from the original) ─────────
CUSTOM_SEARCH_QUERIES = {
    "ON":   '"ON Semiconductor" OR "ON Semi"',
    "MU":   '"Micron Technology" OR "Micron"',
    "TER":  '"Teradyne"',
    "TSM":  '"TSMC" OR "Taiwan Semiconductor"',
    "NVDA": '"Nvidia"',
    "WOLF": '"Wolfspeed"',
    "POET": '"POET Technologies"',
    "LITE": '"Lumentum"',
    "HON":  '"Honeywell"',
    "HQ":   '"Horizon Quantum"',
    "AMZN": '"Amazon.com" OR "Amazon Web Services" OR "Amazon stock"',
    "BABA": '"Alibaba"',
    "SNOW": '"Snowflake Inc" OR "Snowflake stock"',
}

SPECIAL_NAME_FILTERS = {
    "ON":   ["ON Semiconductor", "ON Semi"],
    "MU":   ["Micron"],
    "TSM":  ["TSMC", "Taiwan Semiconductor"],
    "WOLF": ["Wolfspeed"],
    "POET": ["POET Technologies"],
    "LITE": ["Lumentum"],
    "HON":  ["Honeywell"],
    "HQ":   ["Horizon Quantum"],
    "AMZN": ["Amazon.com", "Amazon Web Services"],
    "BABA": ["Alibaba"],
    "SNOW": ["Snowflake Inc", "Snowflake stock"],
}

STRICT_FINANCE_ONLY_TICKERS = {
    "WOLF", "POET", "LITE", "HON", "HQ", "AMZN", "BABA", "SNOW",
}

GENERIC_BAD_WORDS = {
    "inc", "ltd", "corp", "corporation", "company", "co",
    "holdings", "group", "plc", "sa", "nv", "llc", "limited", "the", "and",
}

# Suffixed / international tickers never appear as text in English
# headlines — search the formal company name instead.
NEWS_SEARCH_NAMES = {
    "005930.KS": ["Samsung Electronics"],
    "000660.KS": ["SK Hynix", "SK hynix"],
    "2344.TW":   ["Winbond"],
    "2408.TW":   ["Nanya Technology", "Nanya"],
    "285A.T":    ["Kioxia"],
    "2449.TW":   ["King Yuan Electronics", "KYEC"],
    "6239.TW":   ["Powertech Technology"],
    "0981.HK":   ["SMIC", "Semiconductor Manufacturing International"],
    "ASM.AS":    ["ASM International"],
    "BESI.AS":   ["BE Semiconductor", "Besi"],
    "6857.T":    ["Advantest"],
    "4063.T":    ["Shin-Etsu Chemical", "Shin-Etsu"],
    "3436.T":    ["SUMCO"],
    "6488.TWO":  ["GlobalWafers"],
    "2317.TW":   ["Foxconn", "Hon Hai Precision"],
    "2382.TW":   ["Quanta Computer"],
    "6669.TW":   ["Wiwynn"],
    "3017.TW":   ["Asia Vital Components"],
    "STM":       ["STMicroelectronics"],
    "NOK":       ["Nokia"],
    "ARM":       ["Arm Holdings"],
    "GFS":       ["GlobalFoundries"],
    "NBIS":      ["Nebius"],
}


# ══════════════════════════════════════════════════════════════════
# 2.  Utility helpers
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_sia() -> SentimentIntensityAnalyzer:
    """VADER analyser with the financial lexicon overlay from the original."""
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update({
        "beat": 2.0, "beats": 2.0, "beating": 2.0, "surged": 2.5, "surge": 2.5,
        "jumped": 2.0, "soars": 2.5, "strong": 2.0, "profit": 2.0, "growth": 2.0,
        "up": 1.5, "climb": 1.5, "bull": 2.0, "bullish": 2.0, "record": 1.5,
        "robust": 2.0, "confidence": 1.5, "breakthrough": 2.0,
        "missed": -2.0, "miss": -2.0, "fell": -2.0, "drop": -2.0, "down": -1.5,
        "dump": -2.5, "slump": -2.5, "loss": -2.0, "risk": -1.5, "warns": -2.0,
        "bear": -2.0, "bearish": -2.0, "uncertainty": -1.0, "weak": -2.0,
        "sued": -2.0, "lawsuit": -2.0, "fraud": -3.0,
    })
    return sia


def _retry(fn, retries: int = 3, base_delay: float = 1.5):
    """Call fn(), retrying with exponential back-off on transient failures."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance changes column casing across versions — force Title-Case."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    targets = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    remap = {c: t for c in df.columns for t in targets if str(c).lower() == t.lower()}
    return df.rename(columns=remap)


def _safe_info(ticker_obj) -> dict:
    """ticker.info with retry — parallel calls get rate-limited otherwise."""
    try:
        info = _retry(lambda: ticker_obj.info)
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _empty_fig(message: str = "No data available") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#888888")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _direction_word(base_price, price):
    if base_price is None or price is None:
        return None
    try:
        base_price, price = float(base_price), float(price)
    except (TypeError, ValueError):
        return None
    if pd.isna(base_price) or pd.isna(price):
        return None
    if price > base_price:
        return "Up"
    if price < base_price:
        return "Down"
    return "Flat"


def _direction_arrow(word):
    return _DIRECTION_ARROWS.get(word, "—")


def get_target_friday(date_value):
    """The Friday this prediction is for. If today is Friday → next Friday."""
    date_value = pd.Timestamp(date_value).normalize()
    days_until_friday = (4 - date_value.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    return date_value + pd.Timedelta(days=days_until_friday)


def clean_ticker(raw: str) -> str:
    """Normalise user input: strip whitespace, uppercase, drop junk chars."""
    t = (raw or "").strip().upper()
    t = re.sub(r"[^A-Z0-9.\-^=]", "", t)
    return t


# ── SPY cache, shared across ticker threads ───────────────────────
_spy_cache: dict = {}
_spy_lock = threading.Lock()


def _get_spy_weekly(start_date, end_date) -> pd.Series:
    """Weekly (W-FRI) SPY returns, downloaded once and reused."""
    key = (start_date.date(), end_date.date())
    with _spy_lock:
        if key in _spy_cache:
            return _spy_cache[key]

    try:
        spy_obj = yf.Ticker("SPY")
        spy = _retry(lambda: spy_obj.history(start=start_date, end=end_date))
        spy = _normalize_yf_columns(spy)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_convert(None)
        # W-FRI anchor so it lines up with each ticker's own weekly resample.
        result = spy["Close"].resample("W-FRI").last().pct_change().dropna()
    except Exception:
        result = pd.Series(dtype=float)

    with _spy_lock:
        _spy_cache[key] = result
    return result


def _compute_beta(close_weekly: pd.Series, start_date, end_date) -> float:
    """Beta = Cov(stock, SPY) / Var(SPY) from weekly returns."""
    spy_ret = _get_spy_weekly(start_date, end_date)
    stock_ret = close_weekly.pct_change().dropna()

    combined = pd.concat(
        [stock_ret.rename("s"), spy_ret.rename("m")], axis=1, sort=True
    ).dropna()

    if len(combined) < 30:
        return np.nan

    cov = combined.cov().iloc[0, 1]
    var = combined["m"].var()
    return round(float(cov / var), 4) if var > 0 else np.nan


def compute_volatility_threshold(
    weekly_df: pd.DataFrame,
    multiplier: float = VOL_THRESHOLD_MULTIPLIER,
    floor_pct: float = MIN_THRESHOLD_PCT,
    ceiling_pct: float = MAX_THRESHOLD_PCT,
) -> float:
    """
    Per-ticker accuracy bar, scaled to that ticker's own trailing weekly
    return volatility rather than one flat % for every stock.
    """
    if weekly_df is None or weekly_df.empty or "Pct_Change_Close" not in weekly_df.columns:
        return floor_pct

    returns = weekly_df["Pct_Change_Close"].dropna().tail(52)
    if len(returns) < 8:
        return floor_pct

    vol_pct = returns.std() * 100
    if pd.isna(vol_pct):
        return floor_pct

    return float(np.clip(vol_pct * multiplier, floor_pct, ceiling_pct))


# ══════════════════════════════════════════════════════════════════
# 3.  Weekly dataset builder
# ══════════════════════════════════════════════════════════════════

def compute_weekly_ml_data_with_target(ticker_list):
    """Weekly OHLCV + features + next-week target, per ticker."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=HISTORY_YEARS * 365)
    all_data, names = {}, {}

    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)

            info = _safe_info(stock)
            names[ticker] = (info.get("shortName")
                             or info.get("longName")
                             or ticker)

            trailing_eps = info.get("trailingEps", None)
            if (trailing_eps is None or trailing_eps == 0
                    or (isinstance(trailing_eps, float) and np.isnan(trailing_eps))):
                trailing_eps = 1.0

            hist = _retry(lambda: stock.history(start=start_date, end=end_date))
            hist = _normalize_yf_columns(hist)
            if hist.empty:
                continue
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

            weekly = hist.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna(subset=["Close"])

            if weekly.empty:
                continue

            beta_val = _compute_beta(weekly["Close"], start_date, end_date)
            if np.isnan(beta_val):
                raw = info.get("beta")
                try:
                    beta_val = float(raw) if raw is not None else np.nan
                except (TypeError, ValueError):
                    beta_val = np.nan

            weekly["MA50"]  = weekly["Close"].rolling(50).mean()
            weekly["MA200"] = weekly["Close"].rolling(200).mean()
            weekly["RSI14"] = RSIIndicator(weekly["Close"], window=14).rsi()
            weekly["P/E"]   = weekly["Close"] / trailing_eps

            weekly["Delta_Close"]      = weekly["Close"].diff()
            weekly["Pct_Change_Close"] = weekly["Close"].pct_change()
            weekly["Delta_Volume"]     = weekly["Volume"].diff()
            weekly["Delta_RSI14"]      = weekly["RSI14"].diff()
            weekly["Beta"]             = beta_val

            weekly["Target_Close_1w"]  = weekly["Close"].shift(-1)
            weekly["Target_Return_1w"] = (
                (weekly["Target_Close_1w"] - weekly["Close"]) / weekly["Close"]
            )
            weekly["Ticker"] = ticker
            all_data[ticker] = weekly.reset_index()

        except Exception:
            continue

    return all_data, names


# ══════════════════════════════════════════════════════════════════
# 4.  News sentiment (VADER)
# ══════════════════════════════════════════════════════════════════

def _quote_terms(terms):
    return " OR ".join(f'"{t}"' for t in terms)


def _ticker_is_symbol_only(ticker: str) -> bool:
    """True when the symbol can't work as a plain-text search term."""
    return ("." in ticker) or any(ch.isdigit() for ch in ticker)


def build_news_query(ticker: str, company_name: str = None) -> str:
    if ticker in CUSTOM_SEARCH_QUERIES:
        return CUSTOM_SEARCH_QUERIES[ticker]
    if ticker in NEWS_SEARCH_NAMES:
        return _quote_terms(NEWS_SEARCH_NAMES[ticker])
    if company_name:
        return f'"{company_name}"'
    if _ticker_is_symbol_only(ticker):
        return f'"{ticker.split(".")[0]}"'
    return ticker


def create_variants(company_name, ticker):
    if ticker in SPECIAL_NAME_FILTERS:
        return SPECIAL_NAME_FILTERS[ticker]

    if ticker in NEWS_SEARCH_NAMES:
        variants = {n.lower() for n in NEWS_SEARCH_NAMES[ticker]}
    else:
        variants = set()

    if not _ticker_is_symbol_only(ticker):
        variants.add(ticker.lower())

    full_name_clean = re.sub(r"[^\w\s]", "", (company_name or "").lower())
    if full_name_clean:
        variants.add(full_name_clean)
        parts = full_name_clean.split()
        if parts and len(parts[0]) > 3 and parts[0] not in GENERIC_BAD_WORDS:
            variants.add(parts[0])

    return [v for v in variants if v]


def contains_company(title, variants):
    title_clean = re.sub(r"[^a-zA-Z0-9 ]+", " ", title.lower())
    for v in variants:
        if re.search(r"\b" + re.escape(v.lower()) + r"\b", title_clean):
            return True
    return False


def contains_fin_keyword(title):
    FINANCE_KEYWORDS = [
        "stock", "earnings", "market", "price", "revenue", "guidance",
        "forecast", "investor", "profit", "loss", "results", "trading",
        "shares", "bull", "bear", "rally", "crash",
    ]
    return any(k in title.lower() for k in FINANCE_KEYWORDS)


def process_articles(articles, company_name, ticker, use_finance_keywords=False):
    sia = _get_sia()
    variants = create_variants(company_name, ticker)
    filtered, seen_titles = [], set()

    for article in articles:
        title = article.get("title", "")
        if not title:
            continue
        norm = re.sub(r"\W+", "", title).lower()
        if norm in seen_titles:
            continue
        if not contains_company(title, variants):
            continue
        if use_finance_keywords and not contains_fin_keyword(title):
            continue
        score = sia.polarity_scores(title)["compound"]
        if score == 0:
            continue
        seen_titles.add(norm)

        label = ("Positive" if score > 0.05 else
                 "Negative" if score < -0.05 else "Neutral")
        article_date = (article.get("publishedAt")
                        or article.get("pubDate")
                        or article.get("date") or "")
        filtered.append({
            "Ticker":    ticker,
            "Date":      article_date,
            "Score":     round(score, 4),
            "Sentiment": label,
            "Title":     title,
            "Source":    (article.get("source") or {}).get("name"),
            "URL":       article.get("url"),
        })
    return filtered


def get_news_sentiment_financial(ticker, api_key, company_name=None):
    if not api_key:
        return pd.DataFrame()

    base_url = "https://newsapi.org/v2/everything"
    query = build_news_query(ticker, company_name)
    url = (f"{base_url}?q={requests.utils.quote(query)}&language=en"
           f"&sortBy=publishedAt&pageSize=50&apiKey={api_key}")

    try:
        data = requests.get(url, timeout=10).json()
        if isinstance(data, dict) and data.get("status") == "error":
            return pd.DataFrame()

        articles = data.get("articles", [])
        use_fin_kw = ticker in STRICT_FINANCE_ONLY_TICKERS
        got = process_articles(articles, company_name=company_name,
                               ticker=ticker, use_finance_keywords=use_fin_kw)

        # Fallback: a suffixed/international ticker that came back empty
        # gets one retry on the bare short name.
        if not got and _ticker_is_symbol_only(ticker) and company_name \
                and f'"{company_name}"' != query:
            alt = f'"{company_name}"'
            url2 = (f"{base_url}?q={requests.utils.quote(alt)}&language=en"
                    f"&sortBy=publishedAt&pageSize=50&apiKey={api_key}")
            try:
                data2 = requests.get(url2, timeout=10).json()
                if data2.get("status") != "error":
                    got = process_articles(data2.get("articles", []),
                                           company_name=company_name,
                                           ticker=ticker,
                                           use_finance_keywords=use_fin_kw)
            except Exception:
                pass

        return pd.DataFrame(got)

    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# 5.  XGBoost walk-forward train & predict
# ══════════════════════════════════════════════════════════════════

FEATURES = [
    "Close", "Delta_Close", "Pct_Change_Close",
    "Volume", "Delta_Volume",
    "RSI14", "Delta_RSI14", "MA50", "MA200", "P/E",
]


def _make_model():
    """Same hyper-parameters as the Hugging Face app."""
    return xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        n_jobs=1,
    )


def train_predict_xgboost_filtered(df, correct_threshold=CORRECT_THRESHOLD):
    future_row = df[df["Target_Return_1w"].isna()].copy()
    history    = df.dropna(subset=["Target_Return_1w"]).copy()

    df = df.copy()
    df["Pred_Close_1w"] = np.nan
    df["Set"] = "N/A"

    if len(history) < MIN_TRAIN_SIZE:
        return df

    test_size = max(int(len(history) * 0.20), 1)
    start_test_index = max(len(history) - test_size, MIN_TRAIN_SIZE)

    # ── Walk-forward validation: retrain on everything up to week i,
    #    predict week i, step forward. No look-ahead.
    predictions_list = []
    for i in range(start_test_index, len(history)):
        X_train = history.iloc[:i][FEATURES].fillna(0)
        y_train = history.iloc[:i]["Target_Return_1w"]
        X_test  = history.iloc[i:i + 1][FEATURES].fillna(0)

        mdl = _make_model()
        mdl.fit(X_train, y_train)

        pred_return = mdl.predict(X_test)[0]
        row = history.iloc[i:i + 1].copy()
        row.loc[row.index, "Pred_Close_1w"] = row["Close"] * (1 + pred_return)
        row.loc[row.index, "Set"] = "Test"
        predictions_list.append(row)

    if predictions_list:
        wf = pd.concat(predictions_list)
        history.loc[wf.index, "Pred_Close_1w"] = wf["Pred_Close_1w"]
        history.loc[wf.index, "Set"] = "Test"
        history.loc[history["Set"] != "Test", "Set"] = "Train"

    # ── Final model on the full history → next-week forecast
    final_model = _make_model()
    final_model.fit(history[FEATURES].fillna(0), history["Target_Return_1w"])

    if not future_row.empty:
        pred_return_future = final_model.predict(future_row[FEATURES].fillna(0))
        future_row["Pred_Close_1w"] = future_row["Close"] * (1 + pred_return_future)
        future_row["Set"] = "Future"

    final_df = pd.concat([history, future_row]).sort_values("Date")
    final_df["Correct_Prediction"] = (
        (final_df["Pred_Close_1w"] - final_df["Target_Close_1w"]).abs()
        <= correct_threshold * final_df["Target_Close_1w"]
    ).astype(int)
    final_df.loc[final_df["Set"] == "Future", "Correct_Prediction"] = 0

    return final_df


def backtest_stats(df_pred: pd.DataFrame) -> dict:
    """Walk-forward accuracy for one ticker (in-app only, nothing stored)."""
    test = df_pred[df_pred["Set"] == "Test"].dropna(
        subset=["Pred_Close_1w", "Target_Close_1w"])
    if test.empty:
        return {"Backtest_MAPE_%": np.nan,
                "Backtest_Direction_%": np.nan,
                "Backtest_Weeks": 0}

    mape = ((test["Pred_Close_1w"] - test["Target_Close_1w"]).abs()
            / test["Target_Close_1w"]).mean() * 100

    pred_up   = test["Pred_Close_1w"] > test["Close"]
    actual_up = test["Target_Close_1w"] > test["Close"]
    dir_acc   = (pred_up == actual_up).mean() * 100

    return {"Backtest_MAPE_%": round(float(mape), 2),
            "Backtest_Direction_%": round(float(dir_acc), 1),
            "Backtest_Weeks": int(len(test))}


# ══════════════════════════════════════════════════════════════════
# 6.  Charts
# ══════════════════════════════════════════════════════════════════

def show_fig(fig: plt.Figure):
    """Render then close — otherwise figures pile up across reruns."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


SIGNAL_COLORS = {
    "Strong Buy":   "#1b7f3b",
    "Buy / Hold":   "#69b34c",
    "Hold / Weak":  "#e8a33d",
    "Avoid / Sell": "#c0392b",
}


def plot_latest_signal_all(df_display: pd.DataFrame) -> plt.Figure:
    if df_display.empty or "Total_Points" not in df_display.columns:
        return _empty_fig("No signal data available")

    d = df_display.sort_values("Total_Points", ascending=False)
    width = max(7, min(16, 1.3 * len(d) + 3))
    fig, ax = plt.subplots(figsize=(width, 4.6))

    colors = [SIGNAL_COLORS.get(s, "#999999") for s in d["Signal"]]
    bars = ax.bar(d["Ticker"], d["Total_Points"], color=colors, width=0.6)

    ax.set_title("Signal score — next week", fontsize=13, pad=12, loc="left")
    ax.set_ylabel("Points (0–3)")
    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for rect, val, sig in zip(bars, d["Total_Points"], d["Signal"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.08,
                f"{int(val)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(rect.get_x() + rect.get_width() / 2, 0.08, sig,
                ha="center", va="bottom", fontsize=8, color="white", rotation=90)

    fig.tight_layout()
    return fig


def plot_sentiment_score_chart(df_display: pd.DataFrame) -> plt.Figure:
    if df_display.empty or "Avg_Sentiment" not in df_display.columns:
        return _empty_fig("No sentiment data available")

    d = df_display.sort_values("Avg_Sentiment", ascending=False)
    width = max(7, min(16, 1.3 * len(d) + 3))
    fig, ax = plt.subplots(figsize=(width, 4.6))

    scores = d["Avg_Sentiment"].fillna(0)
    colors = ["#1b7f3b" if s >= 0 else "#c0392b" for s in scores]
    bars = ax.bar(d["Ticker"], scores, color=colors, width=0.6)

    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.axhline(0.05, color="#999999", linewidth=0.7, linestyle="--")
    ax.set_title("News sentiment — VADER composite", fontsize=13, pad=12, loc="left")
    ax.set_ylabel("Score (−1 to +1)")
    ax.set_ylim(-1, 1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for rect, val in zip(bars, scores):
        offset = 0.05 if val >= 0 else -0.11
        ax.text(rect.get_x() + rect.get_width() / 2, val + offset,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def plot_sentiment_distribution(df_news: pd.DataFrame) -> plt.Figure:
    if df_news.empty or "Sentiment" not in df_news.columns:
        return _empty_fig("No headlines available")

    counts = (df_news.groupby(["Ticker", "Sentiment"])
              .size().unstack(fill_value=0)
              .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0))

    width = max(7, min(16, 1.3 * len(counts) + 3))
    fig, ax = plt.subplots(figsize=(width, 4.2))

    bottom = np.zeros(len(counts))
    palette = {"Positive": "#1b7f3b", "Neutral": "#b0b0b0", "Negative": "#c0392b"}
    for label in ["Positive", "Neutral", "Negative"]:
        vals = counts[label].values
        ax.bar(counts.index, vals, bottom=bottom, label=label,
               color=palette[label], width=0.6)
        bottom += vals

    ax.set_title("Headline count by sentiment", fontsize=13, pad=12, loc="left")
    ax.set_ylabel("Headlines")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def plot_expected_move(df_display: pd.DataFrame) -> plt.Figure:
    if df_display.empty or "Expected_Return_%" not in df_display.columns:
        return _empty_fig("No prediction data available")

    d = df_display.sort_values("Expected_Return_%", ascending=False)
    width = max(7, min(16, 1.3 * len(d) + 3))
    fig, ax = plt.subplots(figsize=(width, 4.2))

    vals = d["Expected_Return_%"].fillna(0)
    colors = ["#1b7f3b" if v >= 0 else "#c0392b" for v in vals]
    bars = ax.bar(d["Ticker"], vals, color=colors, width=0.6)
    ax.axhline(0, color="#333333", linewidth=0.9)

    ax.set_title("Predicted 1-week move", fontsize=13, pad=12, loc="left")
    ax.set_ylabel("Expected return (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for rect, val in zip(bars, vals):
        offset = 0.06 * max(abs(vals.max()), abs(vals.min()), 1)
        ax.text(rect.get_x() + rect.get_width() / 2,
                val + (offset if val >= 0 else -offset * 1.8),
                f"{val:+.2f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def plot_price_history(df_hist: pd.DataFrame, ticker: str, weeks: int = 104) -> plt.Figure:
    d = df_hist[df_hist["Ticker"] == ticker].sort_values("Date").tail(weeks)
    if d.empty:
        return _empty_fig(f"No price history for {ticker}")

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(d["Date"], d["Close"], color="#14213d", linewidth=1.6, label="Close")
    if d["MA50"].notna().any():
        ax.plot(d["Date"], d["MA50"], color="#e8a33d", linewidth=1.2, label="MA50")
    if d["MA200"].notna().any():
        ax.plot(d["Date"], d["MA200"], color="#8e44ad", linewidth=1.2, label="MA200")

    ax.set_title(f"{ticker} — weekly close", fontsize=13, pad=12, loc="left")
    ax.set_ylabel("Price")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# 7.  Orchestration
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_market_data(tickers: tuple):
    """Weekly OHLCV + features for each ticker, fetched in parallel."""
    data, names = {}, {}

    def _one(t):
        return compute_weekly_ml_data_with_target([t])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_one, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                d, n = fut.result()
                if t in d and not d[t].empty:
                    data[t] = d[t]
                names.update(n)
            except Exception:
                continue

    return data, names


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_news(tickers: tuple, api_key: str, names: dict):
    """Filtered, VADER-scored headlines for each ticker, fetched in parallel."""
    if not api_key:
        return pd.DataFrame()

    frames = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(get_news_sentiment_financial, t, api_key, names.get(t)): t
            for t in tickers
        }
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if not res.empty:
                    frames.append(res)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True)
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def build_predictions(cache_key: tuple, _market_data: dict, _names: dict,
                      _news_df: pd.DataFrame, _progress=None):
    """
    Run the walk-forward model per ticker, attach sentiment, score the
    3-point signal, and return the next-week (Set == 'Future') rows.

    Walk-forward training is the slow part (one model fit per test week per
    ticker), so the result is cached on `cache_key`. Underscore-prefixed
    arguments are passed through without being hashed — Streamlit skips
    them — which keeps the key cheap instead of fingerprinting every frame.
    """
    market_data, names, news_df, progress = _market_data, _names, _news_df, _progress

    if not news_df.empty:
        ticker_sentiments = news_df.groupby("Ticker")["Score"].mean().to_dict()
        headline_counts = news_df.groupby("Ticker").size().to_dict()
    else:
        ticker_sentiments, headline_counts = {}, {}

    rows, full_frames = [], []
    total = max(len(market_data), 1)

    for idx, (t, df) in enumerate(market_data.items(), start=1):
        if progress:
            progress.progress(idx / total, text=f"Training walk-forward model — {t}")

        try:
            df_pred = train_predict_xgboost_filtered(df)
            if "Set" not in df_pred.columns or (df_pred["Set"] == "Future").sum() == 0:
                continue

            sentiment = ticker_sentiments.get(t, 0.0)
            df_pred["Avg_Sentiment"] = sentiment
            df_pred["Sentiment_Point"] = int(sentiment > 0.05)
            df_pred["Pred_Up_Point"] = (df_pred["Pred_Close_1w"] > df_pred["Close"]).astype(int)
            df_pred["Trend_Point"] = (df_pred["Close"] > df_pred["MA50"]).astype(int)
            df_pred["Total_Points"] = df_pred[
                ["Sentiment_Point", "Pred_Up_Point", "Trend_Point"]
            ].sum(axis=1)
            df_pred["Signal"] = df_pred["Total_Points"].apply(
                lambda x: "Strong Buy" if x >= 3
                else "Buy / Hold" if x == 2
                else "Hold / Weak" if x == 1
                else "Avoid / Sell"
            )
            df_pred["Direction"] = [
                _direction_arrow(_direction_word(base, pred))
                for base, pred in zip(df_pred["Close"], df_pred["Pred_Close_1w"])
            ]

            full_frames.append(df_pred)

            fut = df_pred[df_pred["Set"] == "Future"].iloc[-1]
            stats = backtest_stats(df_pred)

            rows.append({
                "Ticker": t,
                "Name": names.get(t, t),
                "Close": _safe_float(fut["Close"]),
                "Pred_Close_1w": _safe_float(fut["Pred_Close_1w"]),
                "Expected_Return_%": (
                    round((fut["Pred_Close_1w"] / fut["Close"] - 1) * 100, 2)
                    if fut["Close"] else np.nan
                ),
                "Direction": fut["Direction"],
                "Signal": fut["Signal"],
                "Total_Points": int(fut["Total_Points"]),
                "Sentiment_Point": int(fut["Sentiment_Point"]),
                "Pred_Up_Point": int(fut["Pred_Up_Point"]),
                "Trend_Point": int(fut["Trend_Point"]),
                "Avg_Sentiment": round(float(sentiment), 4),
                "Headlines": int(headline_counts.get(t, 0)),
                "RSI14": _safe_float(fut["RSI14"]),
                "P/E": _safe_float(fut["P/E"]),
                "MA50": _safe_float(fut["MA50"]),
                "MA200": _safe_float(fut["MA200"]),
                "Beta": _safe_float(fut["Beta"]),
                "Threshold_%": round(compute_volatility_threshold(df), 2),
                "Target_Friday": get_target_friday(fut["Date"]).date().isoformat(),
                **stats,
            })

        except Exception:
            continue

    df_display = pd.DataFrame(rows)
    if not df_display.empty:
        df_display = df_display.sort_values(
            ["Total_Points", "Expected_Return_%"], ascending=False
        ).reset_index(drop=True)

    df_full = (pd.concat(full_frames, ignore_index=True)
               if full_frames else pd.DataFrame())
    return df_display, df_full


# ══════════════════════════════════════════════════════════════════
# 8.  Sidebar — ticker entry + settings
# ══════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    try:
        if "NEWSAPI_KEY" in st.secrets:
            return str(st.secrets["NEWSAPI_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("NEWSAPI_KEY", "").strip()


if "tickers" not in st.session_state:
    st.session_state.tickers = DEFAULT_TICKERS.copy()

with st.sidebar:
    st.markdown("### Tickers")
    st.caption("Type any symbol — nothing is restricted to a preset list.")

    # Streamlit ≥1.45 lets a multiselect accept free-text entries. Older
    # versions fall back to a comma-separated text box so the app still runs.
    import inspect
    supports_new_options = (
        "accept_new_options" in inspect.signature(st.multiselect).parameters
    )

    if supports_new_options:
        chosen = st.multiselect(
            "Ticker list",
            options=sorted(set(st.session_state.tickers) | set(DEFAULT_TICKERS)),
            default=st.session_state.tickers,
            accept_new_options=True,
            label_visibility="collapsed",
            help="Start typing to add a symbol that isn't listed yet.",
        )
    else:
        raw = st.text_input(
            "Ticker list",
            value=", ".join(st.session_state.tickers),
            label_visibility="collapsed",
            help="Comma-separated, e.g. NVDA, AMD, 2330.TW",
        )
        chosen = raw.split(",")

    tickers = []
    for t in chosen:
        c = clean_ticker(t)
        if c and c not in tickers:
            tickers.append(c)
    st.session_state.tickers = tickers or DEFAULT_TICKERS.copy()

    if st.button("Reset to defaults", use_container_width=True):
        st.session_state.tickers = DEFAULT_TICKERS.copy()
        st.rerun()

    st.divider()
    st.markdown("### News")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "NewsAPI key", type="password", value=DEFAULT_NEWSAPI_KEY,
            help="Set NEWSAPI_KEY in secrets to skip this. "
                 "Without a key the app still runs — sentiment scores 0.",
        ).strip()
    else:
        st.caption("NewsAPI key loaded from secrets.")

    st.divider()
    run = st.button("Run predictions", type="primary", use_container_width=True)
    if st.button("Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")


# ══════════════════════════════════════════════════════════════════
# 9.  Main
# ══════════════════════════════════════════════════════════════════

st.title("Weekly stock prediction")
st.caption(
    "XGBoost walk-forward forecast of next Friday's close, combined with "
    "VADER news sentiment and a trend check. Forward-looking only — nothing "
    "is stored and no past prediction is scored."
)

if run:
    st.session_state.has_run = True

if not st.session_state.get("has_run"):
    st.info(
        f"Ready with {len(st.session_state.tickers)} tickers: "
        f"{', '.join(st.session_state.tickers)}. "
        "Edit the list in the sidebar, then choose **Run predictions**."
    )
    st.stop()

tickers = tuple(st.session_state.tickers)

progress = st.progress(0.0, text="Downloading weekly price history…")
market_data, names = fetch_market_data(tickers)

missing = [t for t in tickers if t not in market_data]
if not market_data:
    progress.empty()
    # Don't let a transient fetch failure (e.g. Yahoo Finance rate-limiting)
    # sit cached as "no data" for the full TTL — clear it so the next click
    # actually retries instead of replaying the same stale empty result.
    fetch_market_data.clear()
    st.error(
        "No price data came back for any of those symbols. This is usually a "
        "temporary Yahoo Finance hiccup — click **Run predictions** again. "
        "If it keeps happening, check the spelling — international listings "
        "need their exchange suffix (e.g. 2330.TW, ASML.AS)."
    )
    st.stop()

progress.progress(0.25, text="Fetching news headlines…")
news_df = fetch_news(tuple(market_data.keys()), api_key, names)

# Cheap fingerprint of the inputs so the cache invalidates when the ticker
# set or the fetched news actually changes, not on every widget interaction.
cache_key = (
    tuple(sorted(market_data.keys())),
    len(news_df),
    round(float(news_df["Score"].sum()), 4) if not news_df.empty else 0.0,
    max((pd.Timestamp(d["Date"].max()).date().isoformat()
         for d in market_data.values()), default=""),
)

df_display, df_full = build_predictions(
    cache_key, market_data, names, news_df, progress
)
progress.empty()

if missing:
    st.warning("No data for: " + ", ".join(missing))

if df_display.empty:
    st.error(
        f"Price data loaded, but no ticker had the {MIN_TRAIN_SIZE} weeks of "
        "history the model needs. Try longer-listed symbols."
    )
    st.stop()

# ── Header metrics ────────────────────────────────────────────────
target_friday = df_display["Target_Friday"].mode().iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Target date", target_friday)
c2.metric("Tickers modelled", len(df_display))
c3.metric("Strong Buy signals", int((df_display["Signal"] == "Strong Buy").sum()))
c4.metric("Headlines analysed", int(df_display["Headlines"].sum()))

if not api_key:
    st.info(
        "No NewsAPI key supplied, so every sentiment score is 0 and the "
        "sentiment point never fires — signals top out at 2 of 3."
    )

tab_pred, tab_news, tab_sent, tab_signal, tab_stock = st.tabs(
    ["Predictions", "News data", "News sentiment", "Signal", "Stock data"]
)


# ── Tab 1: Predictions ────────────────────────────────────────────
with tab_pred:
    st.subheader(f"Next week's forecast — week ending {target_friday}")

    core_cols = [
        "Ticker", "Name", "Close", "Pred_Close_1w", "Expected_Return_%",
        "Direction", "Signal", "Total_Points", "Avg_Sentiment",
        "Backtest_MAPE_%", "Backtest_Direction_%",
    ]
    st.dataframe(
        df_display[core_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Close": st.column_config.NumberColumn("Close", format="%.2f"),
            "Pred_Close_1w": st.column_config.NumberColumn("Predicted close", format="%.2f"),
            "Expected_Return_%": st.column_config.NumberColumn("Exp. return", format="%+.2f%%"),
            "Total_Points": st.column_config.NumberColumn("Points", format="%d"),
            "Avg_Sentiment": st.column_config.NumberColumn("Sentiment", format="%.3f"),
            "Backtest_MAPE_%": st.column_config.NumberColumn("Backtest MAPE", format="%.2f%%"),
            "Backtest_Direction_%": st.column_config.NumberColumn("Direction hit rate", format="%.1f%%"),
        },
    )

    show_fig(plot_expected_move(df_display))

    with st.expander("Full detail — points breakdown, fundamentals, thresholds"):
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.download_button(
        "Download predictions (CSV)",
        df_display.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{target_friday}.csv",
        mime="text/csv",
    )

    st.caption(
        "Backtest figures come from the walk-forward test window — the model is "
        "retrained week by week and predicts one week ahead each time, so there "
        "is no look-ahead. They describe past behaviour, not this forecast."
    )


# ── Tab 2: News data ──────────────────────────────────────────────
with tab_news:
    st.subheader("Headlines")

    if news_df.empty:
        st.info(
            "No headlines. Either no NewsAPI key is set, or nothing recent "
            "passed the company-name and finance-keyword filters."
        )
    else:
        pick = st.multiselect(
            "Filter by ticker",
            options=sorted(news_df["Ticker"].unique()),
            default=sorted(news_df["Ticker"].unique()),
        )
        sent_pick = st.multiselect(
            "Filter by sentiment",
            options=["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
        )

        view = news_df[
            news_df["Ticker"].isin(pick) & news_df["Sentiment"].isin(sent_pick)
        ].copy()
        view["Date"] = view["Date"].dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(
            view[["Ticker", "Date", "Sentiment", "Score", "Title", "Source", "URL"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.3f"),
                "URL": st.column_config.LinkColumn("Link", display_text="Open"),
            },
        )

        st.download_button(
            "Download headlines (CSV)",
            view.to_csv(index=False).encode("utf-8"),
            file_name="news_headlines.csv",
            mime="text/csv",
        )


# ── Tab 3: News sentiment ─────────────────────────────────────────
with tab_sent:
    st.subheader("Sentiment by ticker")
    show_fig(plot_sentiment_score_chart(df_display))
    st.caption(
        "Mean VADER compound score across each ticker's filtered headlines. "
        "The dashed line at 0.05 is the bar a ticker has to clear to earn its "
        "sentiment point."
    )

    show_fig(plot_sentiment_distribution(news_df))

    st.dataframe(
        df_display[["Ticker", "Name", "Avg_Sentiment", "Headlines", "Sentiment_Point"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg_Sentiment": st.column_config.NumberColumn("Avg sentiment", format="%.3f"),
        },
    )


# ── Tab 4: Signal ─────────────────────────────────────────────────
with tab_signal:
    st.subheader("Signal score")
    show_fig(plot_latest_signal_all(df_display))

    st.markdown(
        "Each ticker earns one point per condition met:\n\n"
        "- **Sentiment** — average headline score above 0.05\n"
        "- **Predicted direction** — model forecasts next Friday's close above today's\n"
        "- **Trend** — current close above the 50-week moving average\n\n"
        "3 points → Strong Buy · 2 → Buy / Hold · 1 → Hold / Weak · 0 → Avoid / Sell"
    )

    st.dataframe(
        df_display[["Ticker", "Name", "Sentiment_Point", "Pred_Up_Point",
                    "Trend_Point", "Total_Points", "Signal", "Direction"]],
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 5: Stock data ─────────────────────────────────────────────
with tab_stock:
    st.subheader("Weekly price and indicators")

    tk = st.selectbox("Ticker", options=sorted(market_data.keys()))
    show_fig(plot_price_history(df_full, tk))

    latest = df_display[df_display["Ticker"] == tk]
    if not latest.empty:
        r = latest.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close", f"{r['Close']:.2f}" if pd.notna(r["Close"]) else "—")
        m2.metric("RSI14", f"{r['RSI14']:.1f}" if pd.notna(r["RSI14"]) else "—")
        m3.metric("Beta", f"{r['Beta']:.2f}" if pd.notna(r["Beta"]) else "—")
        m4.metric("P/E", f"{r['P/E']:.1f}" if pd.notna(r["P/E"]) else "—")
        m5.metric("Vol. threshold", f"{r['Threshold_%']:.1f}%")

    hist_cols = ["Date", "Open", "High", "Low", "Close", "Volume",
                 "MA50", "MA200", "RSI14", "P/E", "Beta",
                 "Pct_Change_Close", "Pred_Close_1w", "Set"]
    d = df_full[df_full["Ticker"] == tk].sort_values("Date", ascending=False)
    d = d[[c for c in hist_cols if c in d.columns]]

    st.dataframe(d, use_container_width=True, hide_index=True)

    st.download_button(
        "Download weekly data (CSV)",
        d.to_csv(index=False).encode("utf-8"),
        file_name=f"{tk}_weekly.csv",
        mime="text/csv",
    )

    st.caption(
        "`Set` marks how each week was used: Train, Test (walk-forward "
        "validation), or Future (the row being forecast)."
    )
