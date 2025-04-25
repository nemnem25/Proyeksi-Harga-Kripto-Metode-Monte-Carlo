import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Simulasi Monte Carlo Kripto", layout="centered")
st.title("📈 Simulasi Monte Carlo Harga Kripto")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga kripto selama beberapa hari ke depan. Harga yang digunakan adalah harga penutupan sehari sebelumnya._",
    unsafe_allow_html=True
)

ticker_options = ["BTC-USD", "ETH-USD", "BNB-USD"]
ticker_input = st.selectbox("Pilih simbol ticker kripto yang didukung:", ticker_options)

if ticker_input:
    try:
        st.write(f"📥 Mengambil data harga {ticker_input} dari CoinGecko...")
        coingecko_map = {
            "BTC-USD": "bitcoin",
            "ETH-USD": "ethereum",
            "BNB-USD": "binancecoin"
        }
        coin_id = coingecko_map.get(ticker_input.upper(), "bitcoin")

        # Ambil data harga historis
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": "365"}
        )
        response.raise_for_status()
        prices = response.json()["prices"]
        dates = [datetime.fromtimestamp(price[0] / 1000).date() for price in prices]
        close_prices = [price[1] for price in prices]

        # Proses data menjadi DataFrame
        data = pd.DataFrame({"Date": dates, "Close": close_prices}).set_index("Date")
        yesterday_price = float(data["Close"].iloc[-2])  # Harga penutupan

        # Harga real-time
        try:
            response_realtime = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            )
            response_realtime.raise_for_status()
            coingecko_price = response_realtime.json()[coin_id]["usd"]
        except Exception as e:
            coingecko_price = yesterday_price  # Fallback ke harga penutupan

        # Tampilkan harga dengan HTML
        st.markdown(
            f"""
            <div style="font-family: Arial, sans-serif; font-size: 16px; color: black; background-color: #f0f9e8; padding: 10px; border-radius: 5px;">
                💰 <b>Harga Penutupan Sehari Sebelumnya (CoinGecko):</b> US${yesterday_price:,.2f}<br>
                ⚡️ <b>Harga Real-Time Saat Ini (CoinGecko):</b> US${coingecko_price:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
