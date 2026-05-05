from datetime import date
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    from supabase import create_client
except Exception:
    create_client = None

st.set_page_config(page_title="Wealth Accumulation Dashboard 2026", layout="wide")

ASSET_CLASSES = ["US Stock", "IDX Stock", "ETF", "Crypto"]
RISK_TIERS = ["Conservative", "Moderate", "Aggressive"]
BASELINE_ATR = {"US Stock": 2.0, "IDX Stock": 2.5, "ETF": 1.5, "Crypto": 6.0}


@st.cache_data(ttl=3600)
def fetch_yf_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna(how="all")
    except Exception as e:
        st.warning(f"Failed to fetch yfinance data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_coingecko_global() -> dict:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"CoinGecko global API unavailable: {e}")
        return {}


@st.cache_data(ttl=3600)
def fetch_defillama_pools() -> List[dict]:
    try:
        r = requests.get("https://yields.llama.fi/pools", timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        return data
    except Exception as e:
        st.warning(f"DefiLlama API unavailable: {e}")
        return []


def normalize_ticker(ticker: str, asset_class: str) -> str:
    t = ticker.strip().upper()
    if asset_class == "IDX Stock" and not t.endswith(".JK"):
        t += ".JK"
    return t


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def atr_pct(df: pd.DataFrame, length: int = 20) -> Optional[float]:
    if df.empty or len(df) < length + 1:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    val = atr.iloc[-1] / c.iloc[-1] * 100
    return float(val) if pd.notna(val) else None


def confluence_score(ticker: str) -> Tuple[int, Dict[str, bool], str]:
    score = 0
    details = {"trend": False, "setup": False, "entry": False}
    note = ""

    daily = fetch_yf_history(ticker, period="2y", interval="1d")
    if not daily.empty and len(daily) > 220:
        trend_val = wma(daily["Close"], 200).iloc[-1]
        if pd.notna(trend_val) and daily["Close"].iloc[-1] > trend_val:
            score += 1
            details["trend"] = True

    intraday = fetch_yf_history(ticker, period="60d", interval="60m")
    if intraday.empty or len(intraday) < 60:
        note = "Intraday unavailable, degraded scoring"
    else:
        recent = intraday.tail(48)
        prior = intraday.tail(120).head(72)
        if not prior.empty:
            prior_range = prior["High"].max() - prior["Low"].min()
            recent_range = recent["High"].max() - recent["Low"].min()
            vol_decline = recent["Volume"].mean() < prior["Volume"].mean()
            if prior_range > 0 and recent_range / prior_range < 0.7 and vol_decline:
                score += 1
                details["setup"] = True

    lowtf = fetch_yf_history(ticker, period="10d", interval="5m")
    if not lowtf.empty and len(lowtf) > 30:
        swing_low = lowtf["Low"].tail(20).min()
        last = lowtf.iloc[-1]
        if last["Low"] < swing_low * 0.997 and last["Close"] > swing_low:
            score += 1
            details["entry"] = True

    return score, details, note


def regime_status(ticker: str, asset_class: str) -> Tuple[str, str]:
    df = fetch_yf_history(ticker, period="1y", interval="1d")
    if df.empty or len(df) < 210:
        return "🟡 Neutral", "Insufficient data"
    s200 = sma(df["Close"], 200).iloc[-1]
    status = "🟢 Bull" if df["Close"].iloc[-1] > s200 else "🔴 Bear"
    return status, f"Close {df['Close'].iloc[-1]:.2f} vs SMA200 {s200:.2f}"


def relative_strength_20d(ticker: str, benchmark: str) -> float:
    a = fetch_yf_history(ticker, period="2mo", interval="1d")
    b = fetch_yf_history(benchmark, period="2mo", interval="1d")
    if a.empty or b.empty or len(a) < 21 or len(b) < 21:
        return 0.0
    ar = a["Close"].iloc[-1] / a["Close"].iloc[-21] - 1
    br = b["Close"].iloc[-1] / b["Close"].iloc[-21] - 1
    return float((ar - br) * 100)


def build_volume_profile(df: pd.DataFrame, buckets: int = 20) -> pd.DataFrame:
    d = df.tail(90).copy()
    if d.empty:
        return pd.DataFrame()
    pmin, pmax = d["Low"].min(), d["High"].max()
    edges = np.linspace(pmin, pmax, buckets + 1)
    vols = np.zeros(buckets)
    for _, row in d.iterrows():
        if row["High"] == row["Low"]:
            idx = np.searchsorted(edges, row["Close"], side="right") - 1
            idx = int(np.clip(idx, 0, buckets - 1))
            vols[idx] += row["Volume"]
            continue
        pos = (row["Close"] - row["Low"]) / (row["High"] - row["Low"])
        price = row["Low"] + pos * (row["High"] - row["Low"])
        idx = np.searchsorted(edges, price, side="right") - 1
        idx = int(np.clip(idx, 0, buckets - 1))
        vols[idx] += row["Volume"]
    mids = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"price": mids, "volume": vols}).sort_values("volume", ascending=False)


def get_supabase():
    if create_client is None:
        return None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        st.warning("Supabase not configured in st.secrets. Journal persistence disabled.")
        return None


def kelly_fraction(w: float, r: float) -> float:
    if r <= 0:
        return 0.0
    return max(0.0, (w * (r + 1) - 1) / r)


WATCHLIST_FILE = Path("watchlist.json")


def load_watchlist(sb):
    if sb:
        try:
            r = sb.table("watchlist").select("ticker,asset_class,risk_tier").execute()
            return r.data or []
        except Exception as e:
            st.warning(f"Supabase watchlist load failed: {e}")
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text())
        except Exception as e:
            st.warning(f"Local watchlist read failed: {e}")
    return []


def save_watchlist(sb, watchlist):
    if sb:
        try:
            sb.table("watchlist").delete().neq("ticker", "").execute()
            if watchlist:
                sb.table("watchlist").insert(watchlist).execute()
            return
        except Exception as e:
            st.warning(f"Supabase watchlist save failed: {e}")
    try:
        WATCHLIST_FILE.write_text(json.dumps(watchlist))
    except Exception as e:
        st.warning(f"Local watchlist save failed: {e}")


@st.cache_data(ttl=3600)
def search_tickers(query: str) -> List[dict]:
    if not query.strip():
        return []
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        r = requests.get(url, params={"q": query, "quotesCount": 10, "newsCount": 0}, timeout=15)
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
        return [{"symbol": q.get("symbol"), "name": q.get("shortname") or q.get("longname") or "", "exchange": q.get("exchange", "")} for q in quotes if q.get("symbol")]
    except Exception as e:
        st.warning(f"Ticker search unavailable: {e}")
        return []

st.title("2026 Wealth Accumulation & Risk Management Dashboard")

sb = get_supabase()
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist(sb)

with st.sidebar:
    st.header("Watchlist Manager")
    search_q = st.text_input("Find ticker", placeholder="e.g. Bank Central Asia, Tesla, Ethereum")
    candidates = search_tickers(search_q)
    selected_symbol = ""
    if candidates:
        labels = [f"{c['symbol']} | {c['name']} ({c['exchange']})" for c in candidates]
        picked = st.selectbox("Search results", labels, index=0)
        selected_symbol = picked.split(" | ")[0]
    t = st.text_input("Ticker", value=selected_symbol)
    ac = st.selectbox("Asset Class", ASSET_CLASSES)
    rt = st.selectbox("Risk Tier", RISK_TIERS)
    if st.button("Add Asset") and t:
        item = {"ticker": normalize_ticker(t, ac), "asset_class": ac, "risk_tier": rt}
        if item not in st.session_state.watchlist:
            st.session_state.watchlist.append(item)
            save_watchlist(sb, st.session_state.watchlist)
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []
        save_watchlist(sb, st.session_state.watchlist)
    if st.session_state.watchlist:
        st.dataframe(pd.DataFrame(st.session_state.watchlist), use_container_width=True)

    st.header("Account Settings")
    account_ccy = st.selectbox("Currency", ["USD", "IDR"])


tabs = st.tabs([
    "🎯 Sniper Pick",
    "🌍 Regime Dashboard",
    "🔍 Confluence Scanner",
    "⚖️ Position Sizer",
    "📊 Allocation Planner",
    "📓 Trade Journal",
    "📈 Volume Profile",
])

with tabs[0]:
    st.subheader("Best Opportunity Ranker — Daily Sniper Pick")
    st.caption("This score is a quantitative screen, not a buy signal. Always validate with your own chart read before entering.")
    rows = []
    for item in st.session_state.watchlist:
        ticker, asset_class = item["ticker"], item["asset_class"]
        regime, _ = regime_status(ticker, asset_class)
        if "Bear" in regime:
            continue
        cscore, _, _ = confluence_score(ticker)
        daily = fetch_yf_history(ticker, period="1y", interval="1d")
        if daily.empty or len(daily) < 60:
            continue
        close = daily["Close"].iloc[-1]
        s200 = sma(daily["Close"], 200).iloc[-1]
        s50 = sma(daily["Close"], 50).iloc[-1]
        trend_pts = (10 if close > s200 else 0) + (10 if close > s50 else 0)
        bench = "SPY" if asset_class in ["US Stock", "ETF"] else ("^JKSE" if asset_class == "IDX Stock" else "BTC-USD")
        rs = relative_strength_20d(ticker, bench)
        atrv = atr_pct(daily) or 1
        resistance = daily["High"].tail(20).max()
        var_pts = max(0, min(20, (resistance - close) / max(1e-9, close * atrv / 100) * 2))
        vol_ratio = daily["Volume"].iloc[-1] / max(1, daily["Volume"].tail(20).mean())
        vol_pts = 10 if vol_ratio > 1.5 else (5 if vol_ratio >= 1 else 0)
        rows.append({
            "ticker": ticker,
            "asset_class": asset_class,
            "regime": regime,
            "confluence": cscore,
            "rs": rs,
            "trend_pts": trend_pts,
            "conf_pts": cscore * 10,
            "var_pts": var_pts,
            "vol_pts": vol_pts,
        })

    if rows:
        df = pd.DataFrame(rows)
        df["rs_pts"] = pd.qcut(df["rs"].rank(method="first"), q=min(4, len(df)), labels=False, duplicates="drop")
        df["rs_pts"] = df["rs_pts"].fillna(0).astype(int).map({0: 0, 1: 10, 2: 10, 3: 20})
        df["score"] = df[["trend_pts", "conf_pts", "rs_pts", "var_pts", "vol_pts"]].sum(axis=1)
        df = df.sort_values(["score", "rs"], ascending=[False, False])
        top = df.iloc[0]
        st.success(f"🎯 TODAY'S SNIPER PICK: {top['ticker']} | {top['asset_class']} | Score {top['score']:.0f}/100 | {top['regime']}")
        st.dataframe(df[["ticker", "asset_class", "score", "regime", "confluence", "rs"]], use_container_width=True)
    else:
        st.info("No qualifying assets (regime gate may be excluding bear assets).")

with tabs[1]:
    st.subheader("Regime Dashboard")
    c1, c2, c3 = st.columns(3)
    spy = regime_status("SPY", "ETF")[0]
    c1.metric("SPY Macro Trend", spy)
    ihsg_df = fetch_yf_history("^JKSE", period="1y", interval="1d")
    if not ihsg_df.empty and len(ihsg_df) > 60:
        ihsg_bear = ihsg_df["Close"].iloc[-1] < ema(ihsg_df["Close"], 50).iloc[-1]
        c2.metric("IHSG Regime", "Bearish" if ihsg_bear else "Bullish")
        if ihsg_bear:
            st.error("IDX BEARISH REGIME — reduce exposure")
    btc_df = fetch_yf_history("BTC-USD", period="1y", interval="1d")
    if not btc_df.empty and len(btc_df) > 60:
        c3.metric("Crypto Proxy", "RISK-OFF" if btc_df["Close"].iloc[-1] < sma(btc_df["Close"], 50).iloc[-1] else "RISK-ON")

    global_data = fetch_coingecko_global().get("data", {})
    btc_dom = global_data.get("market_cap_percentage", {}).get("btc")
    if btc_dom is not None:
        st.write(f"BTC Dominance: {btc_dom:.2f}% (if rising: alt distribution warning)")

    out = []
    for item in st.session_state.watchlist:
        rg, detail = regime_status(item["ticker"], item["asset_class"])
        out.append({"Ticker": item["ticker"], "Asset Class": item["asset_class"], "Risk Tier": item["risk_tier"], "Regime": rg, "Detail": detail})
    if out:
        st.dataframe(pd.DataFrame(out), use_container_width=True)

with tabs[2]:
    st.subheader("Three-Screen Confluence Engine")
    tk = st.selectbox("Select asset", [w["ticker"] for w in st.session_state.watchlist] or ["SPY"], key="conf_ticker")
    score, details, note = confluence_score(tk)
    st.metric("Confluence Score", f"{score}/3")
    st.write(details)
    if note:
        st.warning(note)
    if score >= 2:
        st.success("Actionable signal (score ≥ 2)")

with tabs[3]:
    st.subheader("Kelly Position Sizer")
    tk = st.text_input("Ticker for sizing", "SPY")
    ac = st.selectbox("Asset Class", ASSET_CLASSES, key="size_ac")
    win = st.number_input("Win Rate %", 0.0, 100.0, 50.0) / 100
    rr = st.number_input("Average R:R", 0.1, 10.0, 2.0)
    acct = st.number_input(f"Account Size ({account_ccy})", 100.0, 1e12, 10000.0)
    fx = 1.0
    if account_ccy == "IDR":
        fxdf = fetch_yf_history("USDIDR=X", period="5d", interval="1d")
        if not fxdf.empty:
            fx = float(fxdf["Close"].iloc[-1])
            st.caption(f"USD/IDR live rate: {fx:.2f}")
    k = kelly_fraction(win, rr)
    full, half, fixed = k, k / 2, 0.10
    df = fetch_yf_history(normalize_ticker(tk, ac), period="1y", interval="1d")
    apct = atr_pct(df) or BASELINE_ATR[ac]
    scale = min(1.0, max(0.3, BASELINE_ATR[ac] / max(apct, 1e-6)))
    st.write({"Full Kelly": full, "Half Kelly (recommended)": half, "10% fixed": fixed, "Vol Scale": scale, "ATR%": apct})
    px = float(df["Close"].iloc[-1]) if not df.empty else 1.0
    recommended_value = acct * half * scale
    if account_ccy == "IDR":
        usd_value = recommended_value / fx
    else:
        usd_value = recommended_value
    qty = usd_value / px
    if ac == "IDX Stock":
        lots = int(qty // 100)
        st.info(f"Rounded IDX size: {lots} lots ({lots*100} shares)")
        if not df.empty:
            adv_val = float(df["Volume"].tail(20).mean())
            if adv_val < 5_000_000:
                st.error("LOW LIQUIDITY — ARB TRAP RISK. Verify exit feasibility before entry.")
    if ac == "Crypto":
        st.warning("Crypto position sized at reduced Kelly due to volatility regime. Never exceed 10% of total portfolio in any single crypto asset.")

with tabs[4]:
    st.subheader("Capital Allocation Planner")
    tiers = pd.DataFrame([
        {"Tier": "Core", "Asset Class": "US ETFs / IDX Blue Chips", "Target %": 50.0},
        {"Tier": "Growth", "Asset Class": "US/IDX Growth Stocks", "Target %": 25.0},
        {"Tier": "Speculative", "Asset Class": "Crypto / Small Caps / Gorengan", "Target %": 15.0},
        {"Tier": "Yield", "Asset Class": "Stablecoins / Bonds", "Target %": 10.0},
    ])
    edited = st.data_editor(tiers, use_container_width=True, key="alloc_editor")
    vals = {}
    total = 0.0
    for tname in edited["Tier"]:
        v = st.number_input(f"Current value: {tname}", 0.0, 1e12, 0.0)
        vals[tname] = v
        total += v
    if total > 0:
        calc = edited.copy()
        calc["Current"] = calc["Tier"].map(vals)
        calc["Current %"] = calc["Current"] / total * 100
        calc["Delta %"] = calc["Target %"] - calc["Current %"]
        calc["Action"] = calc["Delta %"].apply(lambda x: f"Add {x/100*total:,.0f}" if x > 5 else (f"Trim {-x/100*total:,.0f}" if x < -5 else "Within band"))
        st.dataframe(calc, use_container_width=True)

    pools = fetch_defillama_pools()
    stable = [p for p in pools if p.get("stablecoin") and (p.get("tvlUsd") or 0) > 50_000_000]
    top5 = sorted(stable, key=lambda x: x.get("apy", 0), reverse=True)[:5]
    if top5:
        sdf = pd.DataFrame([{"Pool": p.get("project"), "Chain": p.get("chain"), "APY": p.get("apy"), "TVL": p.get("tvlUsd")} for p in top5])
        st.write("Top stablecoin yields (TVL > $50M):")
        st.dataframe(sdf, use_container_width=True)
        irx = fetch_yf_history("^IRX", period="1mo", interval="1d")
        if not irx.empty:
            rf = float(irx["Close"].iloc[-1])
            best = float(sdf["APY"].max())
            if best < rf:
                st.error("No yield premium. Consider T-bills instead.")

with tabs[5]:
    st.subheader("Behavioral Guardrail Journal")
    cols = st.columns(3)
    jt = cols[0].text_input("Ticker", "SPY")
    jac = cols[1].selectbox("Asset Class", ASSET_CLASSES, key="j_ac")
    direction = cols[2].selectbox("Direction", ["Long", "Short"])
    c2 = st.columns(3)
    entry = c2[0].number_input("Entry Price", 0.0, 1e9, 0.0)
    exitp = c2[1].number_input("Exit Price", 0.0, 1e9, 0.0)
    ccy = c2[2].selectbox("Currency", ["USD", "IDR"])
    d1, d2 = st.columns(2)
    entry_date = d1.date_input("Entry Date", value=date.today())
    exit_date = d2.date_input("Exit Date", value=date.today())
    reason = st.selectbox("Exit Reason", ["Plan", "Extended Target", "Moved Stop", "Stopped Out", "External Force"])
    notes = st.text_area("Notes")
    if st.button("Save Journal Entry"):
        rec = {"ticker": jt, "asset_class": jac, "direction": direction, "entry_price": entry, "exit_price": exitp,
               "entry_date": str(entry_date), "exit_date": str(exit_date), "currency": ccy, "notes": notes, "exit_reason": reason}
        if sb:
            try:
                sb.table("trade_journal").insert(rec).execute()
                st.success("Saved to Supabase")
            except Exception as e:
                st.warning(f"Supabase insert failed: {e}")
        else:
            st.session_state.setdefault("local_journal", []).append(rec)
            st.success("Saved locally in session")

with tabs[6]:
    st.subheader("Volume Profile")
    tk = st.selectbox("Ticker", [w["ticker"] for w in st.session_state.watchlist] or ["SPY"], key="vp_t")
    df = fetch_yf_history(tk, period="6mo", interval="1d")
    vp = build_volume_profile(df)
    if not vp.empty:
        hvn = vp.head(3).sort_values("price")
        lvn = vp.tail(3).sort_values("price")
        st.write("HVN (Strong S/R):", hvn["price"].round(2).tolist())
        st.write("LVN (Fast move zone):", lvn["price"].round(2).tolist())
        fig = go.Figure(go.Bar(x=vp["volume"], y=vp["price"], orientation="h"))
        fig.update_layout(height=500, yaxis_title="Price", xaxis_title="Volume")
        st.plotly_chart(fig, use_container_width=True)
