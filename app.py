from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
import pytz
import requests
import hashlib

# ════════════════════════════════════════════════
# KONFIGURASI
# ════════════════════════════════════════════════

st.set_page_config(
    page_title="Prediksi Harga Kripto (Monte Carlo)",
    layout="centered"
)

HORIZON_TO_PERIOD = {
    3: 60,
    7: 60,
    30: 180,
    90: 365,
    365: 365,
}

HORIZONS = [3, 7, 30, 90, 365]
MAX_PERIOD = max(HORIZON_TO_PERIOD.values())

COINGECKO_MAP = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "ADA-USD": "cardano",
    "BNB-USD": "binancecoin",
}

TICKER_OPTIONS = sorted(COINGECKO_MAP.keys())

# ════════════════════════════════════════════════
# FORMAT
# ════════════════════════════════════════════════

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

# ════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════

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

# ════════════════════════════════════════════════
# TABEL
# ════════════════════════════════════════════════

def buat_tabel_distribusi(finals):
    bins = np.linspace(finals.min(), finals.max(), 10)
    counts, _ = np.histogram(finals, bins=bins)
    probs = counts / len(finals) * 100
    idx_sorted = np.argsort(probs)[::-1]

    html = "<table><tr><th>Peluang</th><th>Rentang Harga</th></tr>"

    total = 0
    low = float("inf")
    high = 0

    for rank, i in enumerate(idx_sorted):
        if probs[i] == 0:
            continue
        l = bins[i]
        h = bins[i+1]
        html += f"<tr><td>{format_persen_indonesia(probs[i])}</td><td>{format_angka_indonesia(l)} – {format_angka_indonesia(h)}</td></tr>"
        if rank < 3:
            total += probs[i]
            low = min(low, l)
            high = max(high, h)

    html += "</table>"
    return html, total, low, high

def buat_tabel_statistik(finals):
    mean_log = np.mean(np.log(finals))
    harga_mean = np.exp(mean_log)
    chance = np.mean(finals > harga_mean) * 100
    std = np.std(finals)
    skew = pd.Series(finals).skew()

    html = f"""
    <table>
    <tr><th>Statistik</th><th>Nilai</th></tr>
    <tr><td>Mean Log</td><td>{format_angka_indonesia(mean_log)}</td></tr>
    <tr><td>Geometric Mean</td><td>US${format_angka_indonesia(harga_mean)}</td></tr>
    <tr><td>Peluang di atas Mean</td><td>{format_persen_indonesia(chance)}</td></tr>
    <tr><td>Std Dev</td><td>US${format_angka_indonesia(std)}</td></tr>
    <tr><td>Skewness</td><td>{format_angka_indonesia(skew)}</td></tr>
    </table>
    """
    return html, harga_mean, chance

# ════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Pengaturan")
    ticker_input = st.selectbox("Pilih Kripto", TICKER_OPTIONS)
    days = st.radio("Horizon", HORIZONS, format_func=lambda x: f"{x} Hari")

coin_id = COINGECKO_MAP[ticker_input]

wib = pytz.timezone("Asia/Jakarta")
today = datetime.now(wib).strftime("%d %B %Y")

st.title("📊 Prediksi Harga Kripto (Monte Carlo)")
st.caption(f"Update: {today}")

df = ambil_data_harga(coin_id)

if len(df) < 2:
    st.error("Data tidak cukup")
    st.stop()

current_price = df["Close"].iloc[-1]
harga_tampil = df["Close"].iloc[-2]

st.write(f"Harga terakhir: **US${format_angka_indonesia(harga_tampil)}**")

# Seed stabil
seed_str = f"{ticker_input}-{today}-{round(current_price,6)}"
seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)

# Simulasi
periode = HORIZON_TO_PERIOD[days]
mu, sigma = hitung_parameter(df, periode)

with st.spinner("Simulasi berjalan..."):
    finals = jalankan_simulasi(current_price, mu, sigma, days, seed + days)

# Distribusi
tabel_dist, total, low, high = buat_tabel_distribusi(finals)

# Highlight
st.success(
    f"🎯 Peluang terbesar: {format_persen_indonesia(total)} "
    f"di kisaran US${format_angka_indonesia(low)} – {format_angka_indonesia(high)}"
)

st.divider()

# Tabel distribusi
st.markdown("### 📈 Distribusi Peluang")
st.markdown(tabel_dist, unsafe_allow_html=True)

st.divider()

# Statistik
st.markdown("### 📊 Statistik")
tabel_stat, harga_mean, chance = buat_tabel_statistik(finals)
st.markdown(tabel_stat, unsafe_allow_html=True)

st.divider()

# Social text
# Hitung potensi perubahan (%)
change_low = ((low - current_price) / current_price) * 100
change_high = ((high - current_price) / current_price) * 100

arah_low = "turun" if change_low < 0 else "naik"
arah_high = "turun" if change_high < 0 else "naik"

# Social text (UPDATED)
social_text = (
    f"Simulasi Monte Carlo menunjukkan peluang {format_persen_indonesia(total)} "
    f"{ticker_input} di kisaran US${format_angka_indonesia(low)} – "
    f"US${format_angka_indonesia(high)} dalam {days} hari, "
    f"dengan potensi {arah_low} {format_persen_indonesia(abs(change_low))} "
    f"hingga {arah_high} {format_persen_indonesia(abs(change_high))} "
    f"dari harga saat ini."
)

st.text_area(
    "Teks Media Sosial",
    value=social_text,
    key=f"social_{ticker_input}_{days}"
)
