
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import requests

st.set_page_config(page_title="Simulasi Monte Carlo Kripto", layout="centered")
st.title("📈 Simulasi Monte Carlo Harga Kripto")

ticker_input = st.text_input(
    "Masukkan simbol ticker cryptocurrency (misal: BTC-USD untuk Bitcoin):",
    value="BTC-USD"
)

if ticker_input:
    try:
        st.write(f"📥 Mengambil data historis {ticker_input} selama satu tahun terakhir...")
        data = yf.download(ticker_input, period="1y", interval="1d", auto_adjust=True)

        if data.empty:
            st.error("Data tidak ditemukan atau kosong. Periksa kembali simbol ticker.")
        else:
            close_prices = data["Close"]
            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

            mu = float(log_returns.mean())
            sigma = float(log_returns.std())

            last_price = float(close_prices.iloc[-1])
            last_date = data.index[-1].date()
            now_wib = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%H:%M:%S")

            # Ambil harga real-time dari CoinGecko
            try:
                coingecko_map = {
                    "BTC-USD": "bitcoin",
                    "ETH-USD": "ethereum",
                    "BNB-USD": "binancecoin"
                }
                coin_id = coingecko_map.get(ticker_input.upper(), "bitcoin")
                response = requests.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                )
                response.raise_for_status()
                coingecko_price = response.json()[coin_id]["usd"]
            except Exception as e:
                coingecko_price = None
                st.warning(f"Gagal mengambil harga real-time dari CoinGecko: {e}")

            # Tampilkan info statistik
            st.success(f"✅ Rata-rata log returns harian: {mu:.5f}, Standar deviasi harian: {sigma:.5f}")

            # Tampilkan harga penutupan terakhir dan harga real-time
            st.info(
    f"""💰 Harga penutupan terakhir (Yahoo Finance): US${last_price:,.2f}  
📅 Tanggal penutupan: {last_date.strftime('%d %B %Y')}  
🕒 Waktu akses (WIB): {now_wib}"""
)

            if coingecko_price:
                st.success(f"⚡️ Harga real-time (CoinGecko): US${coingecko_price:,.2f}")

            # Gunakan harga real-time jika tersedia
            start_price = coingecko_price if coingecko_price else last_price

            for num_days in [7, 30, 90]:
                st.divider()
                st.subheader(f"🔮 Simulasi Monte Carlo untuk {num_days} hari ke depan")

                num_simulations = 1000
                simulations = np.zeros((num_days, num_simulations))

                for i in range(num_simulations):
                    rand_returns = np.random.normal(mu, sigma, num_days)
                    price_path = start_price * np.exp(np.cumsum(rand_returns))
                    simulations[:, i] = price_path

                final_prices = simulations[-1, :]
                bins = np.linspace(final_prices.min(), final_prices.max(), num=10)
                bin_labels = [f"{bins[i]:,.2f} dan {bins[i+1]:,.2f}" for i in range(len(bins) - 1)]
                counts, _ = np.histogram(final_prices, bins=bins)
                probabilities = counts / len(final_prices) * 100

                max_prob_index = np.argmax(probabilities)
                output = [
                    f"{probabilities[i]:.1f}% chance price between {bin_labels[i]}"
                    for i in range(len(probabilities))
                ]
                output[max_prob_index] = f"⬅️ [PROBABILITAS TERTINGGI] {probabilities[max_prob_index]:.1f}% chance price between {bin_labels[max_prob_index]}"

                start_date = last_date
                end_date = last_date + timedelta(days=num_days)
                st.markdown(f"[PROYEKSI HARGA {ticker_input} {num_days} HARI KE DEPAN]")
                st.markdown(f"Tanggal awal: {start_date.strftime('%d %B %Y')}")
                st.markdown(f"Tanggal akhir: {end_date.strftime('%d %B %Y')}")
                st.code("\n".join(output), language="markdown")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
