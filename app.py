import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import requests

st.set_page_config(page_title="Simulasi Monte Carlo Kripto", layout="centered")
st.title("📈 Simulasi Monte Carlo Harga Kripto")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga cryptocurrency selama beberapa hari ke depan. Harga yang digunakan adalah harga penutupan sehari sebelumnya._",
    unsafe_allow_html=True
)

# Dropdown untuk memilih simbol ticker
ticker_options = ["BTC-USD", "ETH-USD", "BNB-USD"]
ticker_input = st.selectbox(
    "Pilih simbol ticker cryptocurrency yang didukung:",
    ticker_options
)
st.caption("⚠️ Simbol ticker yang didukung: BTC-USD (Bitcoin), ETH-USD (Ethereum), BNB-USD (Binance Coin)")

if ticker_input:
    try:
        st.write(f"📥 Mengambil data harga {ticker_input} dari CoinGecko...")
        
        coingecko_map = {
            "BTC-USD": "bitcoin",
            "ETH-USD": "ethereum",
            "BNB-USD": "binancecoin"
        }
        coin_id = coingecko_map.get(ticker_input.upper(), "bitcoin")
        
        # Ambil data harga historis (365 hari terakhir) dari CoinGecko
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": "365"}
        )
        response.raise_for_status()
        prices = response.json()["prices"]
        dates = [datetime.fromtimestamp(price[0] / 1000).date() for price in prices]
        close_prices = [price[1] for price in prices]
        
        data = pd.DataFrame({"Date": dates, "Close": close_prices}).set_index("Date")
        log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

        mu = float(log_returns.mean())
        sigma = float(log_returns.std())

        yesterday_price = float(data["Close"].iloc[-2])  # Harga penutupan sehari sebelumnya
        yesterday_date = data.index[-2]  # Tanggal sehari sebelumnya
        now_wib = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%H:%M:%S")

        # Tampilkan informasi harga penutupan sehari sebelumnya
        st.info(
            f"""💰 **Harga Penutupan Sehari Sebelumnya (CoinGecko):** US${yesterday_price:,.2f}  
📅 **Tanggal Harga Penutupan:** {yesterday_date.strftime('%d %B %Y')}"""
        )

        # Tampilkan harga real-time jika tersedia
        try:
            response_realtime = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            )
            response_realtime.raise_for_status()
            coingecko_price = response_realtime.json()[coin_id]["usd"]
            st.success(
                f"""⚡️ **Harga Real-Time Saat Ini (CoinGecko):** US${coingecko_price:,.2f}  
📅 **Tanggal Harga Real-Time:** {datetime.now().strftime('%d %B %Y')}"""
            )
            # Gunakan harga real-time jika tersedia
            start_price = coingecko_price
        except Exception as e:
            st.warning(f"Gagal mengambil harga real-time: {e}")
            start_price = yesterday_price  # Default ke harga penutupan

        st.subheader("📅 Informasi Tanggal")
        st.markdown(
            f"""
            - **Harga berdasarkan tanggal penutupan:** {yesterday_date.strftime('%d %B %Y')}  
            - **Waktu akses program:** {now_wib}
            """
        )
        st.caption(
            "⚠️ Harga berasal dari penutupan sehari sebelumnya berdasarkan waktu Indonesia (WIB). "
            "Informasi ini berbeda dengan waktu akses program."
        )

        # Simulasi Monte Carlo
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

            start_date = yesterday_date
            end_date = yesterday_date + timedelta(days=num_days)
            st.markdown(f"[PROYEKSI HARGA {ticker_input} {num_days} HARI KE DEPAN]")
            st.markdown(f"Tanggal awal: {start_date.strftime('%d %B %Y')}")
            st.markdown(f"Tanggal akhir: {end_date.strftime('%d %B %Y')}")
            st.code("\n".join(output), language="markdown")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
