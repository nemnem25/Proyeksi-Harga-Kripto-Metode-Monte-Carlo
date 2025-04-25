import streamlit as st 
import numpy as np
import pandas as pd
from datetime import datetime
import requests

st.set_page_config(page_title="Proyeksi Harga Kripto Metode Monte Carlo", layout="centered")
st.title("📈 Proyeksi Harga Kripto Metode Monte Carlo")
st.markdown(
    "_Simulasi berbasis data historis untuk memproyeksikan harga kripto selama beberapa hari ke depan. Simulasi menggunakan metode Monte Carlo. Harga yang digunakan adalah harga penutupan sehari sebelumnya dari CoinGecko._",
    unsafe_allow_html=True
)

ticker_options = [
    "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOT-USD", "DOGE-USD",
    "MATIC-USD", "LTC-USD", "TRX-USD", "SHIB-USD", "AVAX-USD", "UNI-USD", "LINK-USD",
    "ATOM-USD", "ETC-USD", "XLM-USD", "BCH-USD", "ALGO-USD", "FTM-USD", "SAND-USD",
    "MANA-USD", "AXS-USD", "GALA-USD", "APT-USD", "HBAR-USD", "ICP-USD", "NEAR-USD",
    "AAVE-USD", "CAKE-USD", "EOS-USD", "KSM-USD", "ZIL-USD", "QNT-USD", "DYDX-USD",
    "CHZ-USD", "GRT-USD", "VET-USD", "1INCH-USD", "CRV-USD", "RUNE-USD", "FIL-USD",
    "XTZ-USD", "ENS-USD", "FLOW-USD", "LRC-USD", "SUSHI-USD", "COMP-USD", "YFI-USD"
]

coingecko_map = {
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "BNB-USD": "binancecoin", "ADA-USD": "cardano",
    "SOL-USD": "solana", "XRP-USD": "ripple", "DOT-USD": "polkadot", "DOGE-USD": "dogecoin",
    "MATIC-USD": "matic-network", "LTC-USD": "litecoin", "TRX-USD": "tron", "SHIB-USD": "shiba-inu",
    "AVAX-USD": "avalanche-2", "UNI-USD": "uniswap", "LINK-USD": "chainlink", "ATOM-USD": "cosmos",
    "ETC-USD": "ethereum-classic", "XLM-USD": "stellar", "BCH-USD": "bitcoin-cash",
    "ALGO-USD": "algorand", "FTM-USD": "fantom", "SAND-USD": "the-sandbox", "MANA-USD": "decentraland",
    "AXS-USD": "axie-infinity", "GALA-USD": "gala", "APT-USD": "aptos", "HBAR-USD": "hedera",
    "ICP-USD": "internet-computer", "NEAR-USD": "near", "AAVE-USD": "aave", "CAKE-USD": "pancakeswap-token",
    "EOS-USD": "eos", "KSM-USD": "kusama", "ZIL-USD": "zilliqa", "QNT-USD": "quant",
    "DYDX-USD": "dydx", "CHZ-USD": "chiliz", "GRT-USD": "the-graph", "VET-USD": "vechain",
    "1INCH-USD": "1inch", "CRV-USD": "curve-dao-token", "RUNE-USD": "thorchain",
    "FIL-USD": "filecoin", "XTZ-USD": "tezos", "ENS-USD": "ethereum-name-service",
    "FLOW-USD": "flow", "LRC-USD": "loopring", "SUSHI-USD": "sushi", "COMP-USD": "compound",
    "YFI-USD": "yearn-finance"
}

ticker_input = st.selectbox("Pilih simbol kripto:", ticker_options)

if ticker_input:
    try:
        st.write(f"📅 Mengambil data harga {ticker_input} dari CoinGecko...")
        coin_id = coingecko_map.get(ticker_input.upper(), "bitcoin")

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
        yesterday_price = float(data["Close"].iloc[-2])

        try:
            response_realtime = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            )
            response_realtime.raise_for_status()
            coingecko_price = response_realtime.json()[coin_id]["usd"]
        except Exception:
            coingecko_price = yesterday_price

        for num_days in [7, 30, 90]:
            st.subheader(f"🔮 Proyeksi Harga Kripto {coin_id} untuk {num_days} Hari ke Depan")
            simulations = np.zeros((num_days, 1000))
            for i in range(1000):
                rand_returns = np.random.normal(mu, sigma, num_days)
                price_path = coingecko_price * np.exp(np.cumsum(rand_returns))
                simulations[:, i] = price_path

            final_prices = simulations[-1, :]
            bins = np.linspace(final_prices.min(), final_prices.max(), num=10)
            counts, _ = np.histogram(final_prices, bins=bins)
            probabilities = counts / len(final_prices) * 100

            sorted_indices = np.argsort(probabilities)[::-1]
            for idx in range(len(probabilities)):
                low_range = f"{bins[sorted_indices[idx]]:,.2f}".replace(",", ".").replace(".", ",")
                high_range = f"{bins[sorted_indices[idx] + 1]:,.2f}".replace(",", ".").replace(".", ",") if idx + 1 < len(bins) else "N/A"

                if idx == 0:
                    st.markdown(
                        f"**<span style='color:green;'>{probabilities[sorted_indices[idx]]:.1f}% peluang harga berada di antara: US${low_range} dan US${high_range}</span>**",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"{probabilities[sorted_indices[idx]]:.1f}% peluang harga berada di antara: **US${low_range}** dan **US${high_range}**"
                    )
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
