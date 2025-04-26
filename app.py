import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
import requests

# ————————————————————
# Fungsi utility: format angka & persen Indonesia
# ————————————————————

def format_angka_indonesia(val: float) -> str:
    s = f"{val:,.0f}"  # <-- sekarang tanpa desimal
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def format_persen_indonesia(val: float) -> str:
    s = f"{val:.1f}"
    return s.replace(".", ",") + "%"

# ————————————————————
# Konfigurasi halaman Streamlit
# ————————————————————

st.set_page_config(page_title="Proyeksi Harga Kripto Metode Monte Carlo", layout="centered")

# CSS Global untuk Tampilan Font Rapi
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #f0f0f0;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
        color: #f9a825;
    }
    p {
        font-family: 'Arial', sans-serif;
        font-size: 15px;
        margin-bottom: 10px;
        text-align: justify;
    }
    th {
        background-color: #424242;
        color: #fff;
        font-family: 'Verdana', sans-serif;
    }
    td {
        font-family: 'Verdana', sans-serif;
    }
    .highlight-green {
        background-color: #5B5B5B;
        font-weight: bold;
        font-size: 15px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Tampilkan waktu realtime di atas
waktu_sekarang = datetime.now().strftime("%A, %d %B %Y")
st.markdown(f"""
<div style='background-color: #5B5B5B; padding: 8px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 16px;'>
⏰ {waktu_sekarang}
</div>
""", unsafe_allow_html=True)

st.title("Proyeksi Harga Kripto Metode Monte Carlo")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga kripto selama beberapa hari ke depan, menggunakan metode Monte Carlo. Harga yang digunakan adalah harga penutupan selama 365 hari terakhir dari CoinGecko._",
    unsafe_allow_html=True
)

# ————————————————————
# CSS global untuk styling hasil
# ————————————————————

st.markdown("""
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid white;
        padding: 6px;
        text-align: left;
    }
    .highlight-green {
        background-color: #5B5B5B;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ————————————————————
# Daftar ticker dan mapping ke CoinGecko
# ————————————————————

ticker_options = ["BTC-USD", "ETH-USD", "BNB-USD", "USDT-USD", "SOL-USD", "XRP-USD", "TON-USD", "DOGE-USD",
    "ADA-USD", "AVAX-USD", "SHIB-USD", "WETH-USD", "DOT-USD", "TRX-USD", "WBTC-USD", "LINK-USD",
    "MATIC-USD", "ICP-USD", "LTC-USD", "BCH-USD", "NEAR-USD", "UNI-USD", "PEPE-USD", "LEO-USD",
    "DAI-USD", "APT-USD", "STETH-USD", "XLM-USD", "OKB-USD", "ETC-USD", "CRO-USD", "FIL-USD",
    "RNDR-USD", "ATOM-USD", "HBAR-USD", "KAS-USD", "IMX-USD", "TAO-USD", "VET-USD", "MNT-USD",
    "FET-USD", "LDO-USD", "TONCOIN-USD", "AR-USD", "INJ-USD", "GRT-USD", "BTCB-USD", "USDC-USD",
    "SUI-USD", "BGB-USD", "XTZ-USD"]

coingecko_map = {
    "BTC-USD":"bitcoin", "ETH-USD":"ethereum", "BNB-USD":"binancecoin", "USDT-USD":"tether", "SOL-USD":"solana",
    "XRP-USD":"ripple", "TON-USD":"toncoin", "DOGE-USD":"dogecoin", "ADA-USD":"cardano", "AVAX-USD":"avalanche-2",
    "SHIB-USD":"shiba-inu", "WETH-USD":"weth", "DOT-USD":"polkadot", "TRX-USD":"tron", "WBTC-USD":"wrapped-bitcoin",
    "LINK-USD":"chainlink", "MATIC-USD":"matic-network", "ICP-USD":"internet-computer", "LTC-USD":"litecoin",
    "BCH-USD":"bitcoin-cash", "NEAR-USD":"near", "UNI-USD":"uniswap", "PEPE-USD":"pepe", "LEO-USD":"leo-token",
    "DAI-USD":"dai", "APT-USD":"aptos", "STETH-USD":"staked-ether", "XLM-USD":"stellar", "OKB-USD":"okb",
    "ETC-USD":"ethereum-classic", "CRO-USD":"crypto-com-chain", "FIL-USD":"filecoin", "RNDR-USD":"render-token",
    "ATOM-USD":"cosmos", "HBAR-USD":"hedera-hashgraph", "KAS-USD":"kaspa", "IMX-USD":"immutable-x",
    "TAO-USD":"bittensor", "VET-USD":"vechain", "MNT-USD":"mantle", "FET-USD":"fetch-ai", "LDO-USD":"lido-dao",
    "TONCOIN-USD":"toncoin", "AR-USD":"arweave", "INJ-USD":"injective-protocol", "GRT-USD":"the-graph",
    "BTCB-USD":"bitcoin-bep2", "USDC-USD":"usd-coin", "SUI-USD":"sui", "BGB-USD":"bitget-token", "XTZ-USD":"tezos"
}

# ————————————————————
# Input pengguna
# ————————————————————

ticker_input = st.selectbox("Pilih simbol kripto:", ticker_options)
if not ticker_input:
    st.stop()

# ————————————————————
# Logika simulasi
# ————————————————————

try:
    coin_id = coingecko_map[ticker_input]

    resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency":"usd","days":"365"}
    )
    resp.raise_for_status()
    prices = resp.json()["prices"]
    dates = [datetime.fromtimestamp(p[0]/1000).date() for p in prices]
    closes = [p[1] for p in prices]

    df = pd.DataFrame({"Date":dates, "Close":closes}).set_index("Date")
    if len(df) < 2:
        st.warning("Data historis tidak mencukupi untuk simulasi.")
        st.stop()

    log_ret = np.log(df["Close"]/df["Close"].shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()

    try:
        r2 = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        )
        r2.raise_for_status()
        current_price = r2.json()[coin_id]["usd"]
    except:
        current_price = df["Close"].iloc[-2]

    harga_penutupan = format_angka_indonesia(current_price)
    st.write(f"**Harga penutupan {ticker_input} terakhir: US${harga_penutupan}**")

    for days in [7, 30, 90]:
        st.subheader(f"Proyeksi Harga Kripto {ticker_input} untuk {days} Hari ke Depan")
        sims = np.zeros((days, 1000))
        for i in range(1000):
            rw = np.random.normal(mu, sigma, days)
            sims[:, i] = current_price * np.exp(np.cumsum(rw))
        finals = sims[-1, :
