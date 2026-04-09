from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
import pytz
import requests
import hashlib

st.set_page_config(
    page_title="Proyeksi Harga Kripto Metode Monte Carlo",
    layout="centered"
)

HORIZON_TO_PERIOD: dict = {
    3: 60,
    7: 60,
    30: 180,
    90: 365,
    365: 365,
}

HORIZONS = [3, 7, 30, 90, 365]
MAX_PERIOD = max(HORIZON_TO_PERIOD.values())

COINGECKO_MAP = {
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "BNB-USD": "binancecoin",
    "ADA-USD": "cardano", "SOL-USD": "solana", "XRP-USD": "ripple",
}

TICKER_OPTIONS = sorted(COINGECKO_MAP.keys())

# ───────── FORMAT ─────────
def format_angka_indonesia(val):
    try:
        val = float(val)
    except:
        return str(val)
    if abs(val) < 1:
        s = f"{val:,.8f}"
    else:
        s = f"{val:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def format_persen_indonesia(val):
    try:
        val = float(val)
    except:
        return str(val)
    return f"{val:.1f}".replace(".", ",") + "%"

# ───────── DATA ─────────
@st.cache_data(ttl=3600)
def ambil_data_harga(coin_id: str):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(MAX_PERIOD)}
    resp = requests.get(url, params=params)
    data = resp.json()

    prices = data.get("prices", [])
    if len(prices) < 60:
        raise ValueError("Data tidak cukup")

    dates = [datetime.fromtimestamp(p[0]/1000).date() for p in prices]
    closes = [p[1] for p in prices]

    return pd.DataFrame({"Date": dates, "Close": closes}).set_index("Date")

def hitung_parameter(df, periode):
    df_slice = df.iloc[-(periode+1):]
    log_ret = np.log(df_slice["Close"] / df_slice["Close"].shift(1)).dropna()
    return float(log_ret.mean()), float(log_ret.std())

def jalankan_simulasi(current_price, mu, sigma, days, seed):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(mu, sigma, size=(days, 100_000))
    return current_price * np.exp(np.sum(log_returns, axis=0))

# ───────── UI ─────────

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    ticker_input = st.selectbox("Pilih Kripto", TICKER_OPTIONS)
    selected_days = st.radio(
        "Horizon Waktu",
        HORIZONS,
        format_func=lambda x: f"{x} Hari"
    )

coin_id = COINGECKO_MAP[ticker_input]

# Header waktu
wib = pytz.timezone("Asia/Jakarta")
waktu = datetime.now(wib).strftime("%d %B %Y")
st.title("📊 Prediksi Harga Kripto (Monte Carlo)")
st.caption(f"Update: {waktu}")

# Ambil data
df = ambil_data_harga(coin_id)

if len(df) < 2:
    st.error("Data tidak cukup")
    st.stop()

current_price = df["Close"].iloc[-1]
harga_tampil = df["Close"].iloc[-2]

st.write(
    f"Harga terakhir: **US${format_angka_indonesia(harga_tampil)}**"
)

# Seed stabil
today_str = datetime.now().strftime("%Y-%m-%d")
seed_str = f"{ticker_input}-{today_str}-{round(current_price,6)}"
seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)

# ───────── SIMULASI ─────────

days = selected_days
periode = HORIZON_TO_PERIOD[days]

mu, sigma = hitung_parameter(df, periode)

with st.spinner("Menjalankan simulasi..."):
    finals = jalankan_simulasi(
        current_price,
        mu,
        sigma,
        days,
        seed + days
    )

# Distribusi sederhana
bins = np.linspace(finals.min(), finals.max(), 10)
counts, _ = np.histogram(finals, bins=bins)
probs = counts / len(finals) * 100
idx = np.argsort(probs)[::-1]

top3 = idx[:3]
low = bins[min(top3)]
high = bins[max(top3)+1]
total_prob = probs[top3].sum()

# Highlight
st.success(
    f"🎯 Peluang terbesar: {format_persen_indonesia(total_prob)} "
    f"di kisaran US${format_angka_indonesia(low)} – "
    f"US${format_angka_indonesia(high)}"
)

# Social text
social_text = (
    f"Simulasi Monte Carlo menunjukkan peluang {format_persen_indonesia(total_prob)} "
    f"{ticker_input} di kisaran US${format_angka_indonesia(low)} – "
    f"US${format_angka_indonesia(high)} dalam {days} hari."
)

st.text_area(
    "Teks Media Sosial",
    value=social_text,
    key=f"social_{ticker_input}_{days}"
)
