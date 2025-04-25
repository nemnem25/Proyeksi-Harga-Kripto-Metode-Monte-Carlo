import streamlit as st  
import numpy as np  
import pandas as pd  
from datetime import datetime  
import requests  

# —————————————————————————
# Fungsi utility: format angka & persen Indonesia
# —————————————————————————
def format_angka_indonesia(val: float) -> str:
    """
    Format angka dengan titik sebagai pemisah ribuan dan koma sebagai desimal.
    Contoh: 12345.67 → "12.345,67"
    """
    s = f"{val:,.2f}"       # US style: 1,234,567.89
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_persen_indonesia(val: float) -> str:
    """
    Format persentase dengan koma sebagai desimal dan tambahkan simbol "%".
    Contoh: 24.7 → "24,7%"
    """
    s = f"{val:.1f}"        # e.g. "24.7"
    return s.replace(".", ",") + "%"

# —————————————————————————
# Konfigurasi halaman Streamlit
# —————————————————————————
st.set_page_config(page_title="Proyeksi Harga Kripto Metode Monte Carlo", layout="centered")
st.title("📈 Proyeksi Harga Kripto Metode Monte Carlo")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga kripto selama beberapa hari ke depan. Simulasi menggunakan metode Monte Carlo. Harga yang digunakan adalah harga penutupan sehari sebelumnya dari CoinGecko._",
    unsafe_allow_html=True
)

# —————————————————————————
# CSS global untuk styling hasil
# —————————————————————————
st.markdown("""
    <style>
    .highlight {
        font-size: 16px;
        font-weight: bold;
        color: #2ecc71;
        margin-bottom: 8px;
    }
    .normal {
        font-size: 15px;
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# —————————————————————————
# Daftar ticker dan mapping ke CoinGecko
# —————————————————————————
ticker_options = [
    "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOT-USD", "DOGE-USD",
    # ... tambahkan lainnya sesuai kebutuhan ...
]
coingecko_map = {
    "BTC-USD":"bitcoin", "ETH-USD":"ethereum", "BNB-USD":"binancecoin",
    "ADA-USD":"cardano", "SOL-USD":"solana", # ... dst ...
}

# —————————————————————————
# Input pengguna
# —————————————————————————
ticker_input = st.selectbox("Pilih simbol kripto:", ticker_options)
if not ticker_input:
    st.stop()

# —————————————————————————
# Logika simulasi
# —————————————————————————
try:
    st.write(f"📥 Mengambil data harga {ticker_input} dari CoinGecko...")
    coin_id = coingecko_map[ticker_input]

    # Ambil data historis 1 tahun
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

    # Coba ambil harga real-time, fallback ke kemarin
    try:
        r2 = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        )
        r2.raise_for_status()
        current_price = r2.json()[coin_id]["usd"]
    except:
        current_price = df["Close"].iloc[-2]

    # Simulasi untuk horizon 7,30,90 hari
    for days in [7, 30, 90]:
        st.subheader(f"🔮 Proyeksi Harga Kripto {ticker_input} untuk {days} Hari ke Depan")
        sims = np.zeros((days, 1000))
        for i in range(1000):
            rw = np.random.normal(mu, sigma, days)
            sims[:, i] = current_price * np.exp(np.cumsum(rw))
        finals = sims[-1, :]

        # Hitung probabilitas tiap rentang harga
        bins = np.linspace(finals.min(), finals.max(), 10)
        counts, _ = np.histogram(finals, bins=bins)
        probs = counts / len(finals) * 100
        idx_sorted = np.argsort(probs)[::-1]

        # Tampilkan hasil dengan format Indonesia
        for rank, idx in enumerate(idx_sorted):
            low = format_angka_indonesia(bins[idx])
            high = format_angka_indonesia(bins[idx+1]) if idx+1 < len(bins) else "N/A"
            pct = format_persen_indonesia(probs[idx])
            html = (
                f"<div class='{ 'highlight' if rank==0 else 'normal' }'>"
                f"{pct} peluang harga berada di antara: US${low} dan US${high}"  
                f"</div>"
            )
            st.markdown(html, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
