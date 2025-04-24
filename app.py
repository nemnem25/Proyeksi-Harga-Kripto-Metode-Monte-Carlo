import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

st.title("📈 Simulasi Monte Carlo Harga Kripto")

ticker = st.text_input("Masukkan simbol ticker cryptocurrency (misal: BTC-USD, ETH-USD):", "BTC-USD")

if ticker:
    try:
        st.write(f"Mengambil data historis {ticker} selama satu tahun terakhir...")
        data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
        if data.empty:
            st.error("Data tidak ditemukan. Pastikan ticker benar.")
        else:
            close_prices = data['Close']
            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            mu = log_returns.mean()
            sigma = log_returns.std()

            st.write(f"📊 Rata-rata log returns harian: `{mu:.5f}`")
            st.write(f"📊 Standar deviasi harian: `{sigma:.5f}`")
            last_price = close_prices.iloc[-1]

            days_options = [7, 30, 90]
            for num_days in days_options:
                st.subheader(f"🔮 Proyeksi {ticker} untuk {num_days} hari ke depan")
                simulations = np.zeros((num_days, 1000))
                for i in range(1000):
                    rand_returns = np.random.normal(mu, sigma, num_days)
                    price_path = last_price * np.exp(np.cumsum(rand_returns))
                    simulations[:, i] = price_path

                final_prices = simulations[-1, :]
                bins = np.linspace(final_prices.min(), final_prices.max(), 10)
                bin_labels = [f"{bins[i]:,.2f} dan {bins[i+1]:,.2f}" for i in range(len(bins) - 1)]
                counts, _ = np.histogram(final_prices, bins=bins)
                probabilities = counts / len(final_prices) * 100

                max_prob_index = np.argmax(probabilities)

                tanggal_awal = data.index[-1].strftime('%d %B %Y')
                tanggal_akhir = (data.index[-1] + timedelta(days=num_days)).strftime('%d %B %Y')

                st.markdown(f"📅 Tanggal awal: **{tanggal_awal}**")
                st.markdown(f"📅 Tanggal akhir: **{tanggal_akhir}**")

                for i, label in enumerate(bin_labels):
                    if i == max_prob_index:
                        st.markdown(f"<span style='color:green'><b><-- [PROBABILITAS TERTINGGI]</b> {probabilities[i]:.1f}% kemungkinan harga antara {label}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"{probabilities[i]:.1f}% kemungkinan harga antara {label}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
