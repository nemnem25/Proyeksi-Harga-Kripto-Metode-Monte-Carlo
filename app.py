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
    .kesimpulan {
        font-family: 'Arial', sans-serif;
        font-size: 15px;
        line-height: 1.8;
        text-align: justify;
        color: #f0f0f0;
        margin-top: 20px;
        padding: 10px;
        background-color: #424242;
        border-radius: 8px;
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
# Daftar ticker dan mapping ke CoinGecko
# ————————————————————

ticker_options = ["BTC-USD", "ETH-USD", "BNB-USD", "USDT-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "SHIB-USD"]
coingecko_map = {
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "BNB-USD": "binancecoin", "USDT-USD": "tether",
    "SOL-USD": "solana", "XRP-USD": "ripple", "DOGE-USD": "dogecoin", "ADA-USD": "cardano", "SHIB-USD": "shiba-inu"
}

# Input pengguna
ticker_input = st.selectbox("Pilih simbol kripto:", ticker_options)
if not ticker_input:
    st.stop()

# Logika simulasi
try:
    coin_id = coingecko_map[ticker_input]
    resp = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", params={"vs_currency": "usd", "days": "365"})
    resp.raise_for_status()
    prices = resp.json()["prices"]
    dates = [datetime.fromtimestamp(p[0]/1000).date() for p in prices]
    closes = [p[1] for p in prices]
    
    df = pd.DataFrame({"Date": dates, "Close": closes}).set_index("Date")
    if len(df) < 2:
        st.warning("Data historis tidak mencukupi untuk simulasi.")
        st.stop()
    
    log_ret = np.log(df["Close"]/df["Close"].shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()
    
    try:
        r2 = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd")
        r2.raise_for_status()
        current_price = r2.json()[coin_id]["usd"]
    except:
        current_price = df["Close"].iloc[-2]
    
    for days in [7, 30, 90]:
        sims = np.zeros((days, 1000))
        for i in range(1000):
            rw = np.random.normal(mu, sigma, days)
            sims[:, i] = current_price * np.exp(np.cumsum(rw))
        finals = sims[-1, :]
        
        bins = np.linspace(finals.min(), finals.max(), 10)
        counts, _ = np.histogram(finals, bins=bins)
        probs = counts / len(finals) * 100
        
        st.subheader(f"Proyeksi untuk {days} hari")
        st.write(f"**Simulasi selesai! Rentang harga akhir: {format_angka_indonesia(finals.min())} - {format_angka_indonesia(finals.max())}**")
    
    # Kesimpulan
    rentang_bawah = finals.min()
    rentang_atas = finals.max()
    total_peluang = sum(probs[:3])
    skewness = (np.sum((finals - finals.mean()) ** 3) / len(finals)) / (finals.std() ** 3)
    
    kesimpulan_html = f"""
        <div class="kesimpulan">
            <b>Kesimpulan:</b> <br>
            Dari hasil simulasi, diperkirakan harga akan berada dalam kisaran <b>US${format_angka_indonesia(rentang_bawah)}</b> hingga <b>US${format_angka_indonesia(rentang_atas)}</b> dengan peluang kumulatif sebesar <b>{format_persen_indonesia(total_peluang)}</b>. <br>
            Skewness positif sebesar <b>{skewness:.3f}</b> menunjukkan bahwa harga lebih condong untuk bergerak naik, meskipun fluktuasi tajam tidak diharapkan dalam waktu dekat.
        </div>
    """
    st.markdown(kesimpulan_html, unsafe_allow_html=True)

# Penjelasan Statistik Lengkap
def buat_tabel_penjelasan_statistik():
    tabel_html = """
        <table>
            <thead><tr><th>Statistik</th><th>Penjelasan</th></tr></thead>
            <tbody>
                <tr><td>Mean (Harga Logaritmik)</td><td>Rata-rata perubahan harga dalam bentuk logaritmik, menggambarkan tren harga secara umum dalam jangka waktu tertentu.</td></tr>
                <tr><td>Harga Berdasarkan Mean</td><td>Perkiraan harga berdasarkan rata-rata logaritmik, memberikan titik acuan harga yang lebih stabil.</td></tr>
                <tr><td>Chance Above Mean</td><td>Probabilitas harga bergerak lebih tinggi dari rata-rata historis.</td></tr>
                <tr><td>Standard Deviation</td><td>Ukuran volatilitas, menunjukkan seberapa besar fluktuasi harga dari rata-rata.</td></tr>
                <tr><td>Skewness</td><td>Mengukur kecenderungan pergerakan harga; nilai positif menunjukkan kecenderungan naik.</td></tr>
                <tr><td>Rentang Harga (Kumulatif Tertinggi)</td><td>Kisaran harga yang memiliki peluang tertinggi dalam simulasi, bersama dengan persentase kumulatif.</td></tr>
            </tbody>
        </table>
    """
    st.markdown(tabel_html, unsafe_allow_html=True)

# Panggilan fungsi untuk tabel penjelasan
buat_tabel_penjelasan_statistik()
