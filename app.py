from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
import pytz
import requests

# ════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ════════════════════════════════════════════════

st.set_page_config(
    page_title="Proyeksi Harga Kripto Metode Monte Carlo",
    layout="centered"
)

# ════════════════════════════════════════════════
# KONSTANTA
# ════════════════════════════════════════════════

# Horizon proyeksi (hari) → periode data historis yang dipakai untuk
# mengestimasi mu dan sigma. Semakin pendek horizon, semakin pendek
# periode agar estimasi volatilitas mencerminkan kondisi pasar terkini.
HORIZON_TO_PERIOD: dict = {
    3:   60,
    7:   60,
    30:  180,
    90:  365,
    365: 365,   # batas maksimum API CoinGecko gratis (granularitas harian)
}

HORIZONS = [3, 7, 30, 90, 365]

# Periode terpanjang yang dibutuhkan — dipakai saat memanggil API
MAX_PERIOD = max(HORIZON_TO_PERIOD.values())

COINGECKO_MAP = {
    "BTC-USD": "bitcoin",          "ETH-USD": "ethereum",         "BNB-USD": "binancecoin",
    "USDT-USD": "tether",          "SOL-USD": "solana",            "XRP-USD": "ripple",
    "TON-USD": "toncoin",          "DOGE-USD": "dogecoin",         "ADA-USD": "cardano",
    "AVAX-USD": "avalanche-2",     "SHIB-USD": "shiba-inu",        "WETH-USD": "weth",
    "DOT-USD": "polkadot",         "TRX-USD": "tron",              "WBTC-USD": "wrapped-bitcoin",
    "LINK-USD": "chainlink",       "MATIC-USD": "matic-network",   "ICP-USD": "internet-computer",
    "LTC-USD": "litecoin",         "BCH-USD": "bitcoin-cash",      "NEAR-USD": "near",
    "UNI-USD": "uniswap",          "PEPE-USD": "pepe",             "LEO-USD": "leo-token",
    "DAI-USD": "dai",              "APT-USD": "aptos",             "STETH-USD": "staked-ether",
    "XLM-USD": "stellar",          "OKB-USD": "okb",               "ETC-USD": "ethereum-classic",
    "CRO-USD": "crypto-com-chain", "FIL-USD": "filecoin",          "RNDR-USD": "render-token",
    "ATOM-USD": "cosmos",          "HBAR-USD": "hedera-hashgraph", "KAS-USD": "kaspa",
    "IMX-USD": "immutable-x",      "TAO-USD": "bittensor",         "VET-USD": "vechain",
    "MNT-USD": "mantle",           "FET-USD": "fetch-ai",          "LDO-USD": "lido-dao",
    "TONCOIN-USD": "toncoin",      "AR-USD": "arweave",            "INJ-USD": "injective-protocol",
    "GRT-USD": "the-graph",        "BTCB-USD": "bitcoin-bep2",     "USDC-USD": "usd-coin",
    "SUI-USD": "sui",              "BGB-USD": "bitget-token",      "XTZ-USD": "tezos",
    "MUBARAK-USD": "mubarakcoin",
}

TICKER_OPTIONS = sorted(COINGECKO_MAP.keys())


# ════════════════════════════════════════════════
# UTILITAS FORMAT ANGKA INDONESIA
# ════════════════════════════════════════════════

def format_angka_indonesia(val) -> str:
    """Format angka ke format Indonesia (titik=ribuan, koma=desimal)."""
    try:
        val = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(val) < 1:
        s = f"{val:,.8f}"
    else:
        s = f"{val:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_persen_indonesia(val) -> str:
    """Format nilai persen ke format Indonesia."""
    try:
        val = float(val)
    except (TypeError, ValueError):
        return str(val)
    return f"{val:.1f}".replace(".", ",") + "%"


def interpretasi_skewness(skewness: float) -> str:
    """Kembalikan interpretasi tekstual yang kondisional dari nilai skewness."""
    skew_fmt = format_angka_indonesia(skewness)
    if skewness > 0.5:
        return (
            f"Dengan <strong>Skewness</strong> sebesar <strong>{skew_fmt}</strong>, "
            "distribusi harga condong ke kanan (<em>positively skewed</em>), "
            "artinya peluang harga naik secara signifikan lebih besar daripada turun."
        )
    elif skewness < -0.5:
        return (
            f"Dengan <strong>Skewness</strong> sebesar <strong>{skew_fmt}</strong>, "
            "distribusi harga condong ke kiri (<em>negatively skewed</em>), "
            "artinya peluang harga turun secara signifikan lebih besar daripada naik."
        )
    else:
        return (
            f"Dengan <strong>Skewness</strong> sebesar <strong>{skew_fmt}</strong>, "
            "distribusi harga relatif simetris, "
            "artinya peluang naik dan turun hampir seimbang."
        )


# ════════════════════════════════════════════════
# PENGAMBILAN DATA DARI COINGECKO (DENGAN CACHE)
# ════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def ambil_data_harga(coin_id: str) -> pd.DataFrame:
    """
    Ambil data harga historis harian dari CoinGecko.
    Cache selama 1 jam untuk menghindari rate-limiting.

    Selalu mengambil MAX_PERIOD hari (365 hari) agar satu panggilan API
    cukup untuk semua horizon. Slicing per horizon dilakukan di luar fungsi ini.

    Returns:
        DataFrame dengan kolom 'Close' dan index 'Date' (datetime.date).

    Raises:
        ConnectionError: Untuk berbagai kegagalan jaringan / API.
        ValueError: Jika data yang diterima tidak mencukupi.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(MAX_PERIOD)}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise ConnectionError(
            "Permintaan ke CoinGecko habis waktu (timeout). "
            "Coba lagi beberapa saat."
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 429:
            raise ConnectionError(
                "Batas permintaan API CoinGecko terlampaui (429 Too Many Requests). "
                "Tunggu beberapa menit lalu coba lagi."
            )
        raise ConnectionError(
            f"API CoinGecko mengembalikan error HTTP {status}. "
            "Periksa koneksi internet atau coba lagi."
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Tidak dapat terhubung ke CoinGecko. "
            "Periksa koneksi internet Anda."
        )

    data = resp.json()
    prices = data.get("prices", [])

    if len(prices) < 60:
        raise ValueError(
            "Data historis yang diterima tidak mencukupi untuk simulasi (minimal 60 hari). "
            "Coba pilih koin yang berbeda."
        )

    dates = [datetime.fromtimestamp(p[0] / 1000).date() for p in prices]
    closes = [p[1] for p in prices]

    df = pd.DataFrame({"Date": dates, "Close": closes}).set_index("Date")
    return df


def hitung_parameter(df: pd.DataFrame, periode: int) -> Tuple[float, float]:
    """
    Hitung mu dan sigma dari slice N hari terakhir DataFrame.

    Args:
        df: DataFrame harga lengkap (365 hari).
        periode: Jumlah hari historis yang dipakai untuk estimasi.

    Returns:
        Tuple (mu, sigma) dari log-return harian.
    """
    # Tambah 1 baris untuk kebutuhan shift log-return
    n_slice = min(periode + 1, len(df))
    df_slice = df.iloc[-n_slice:]

    log_ret = np.log(df_slice["Close"] / df_slice["Close"].shift(1)).dropna()
    mu = float(log_ret.mean())
    sigma = float(log_ret.std())
    return mu, sigma


# ════════════════════════════════════════════════
# SIMULASI MONTE CARLO (VEKTORISASI PENUH)
# ════════════════════════════════════════════════

def jalankan_simulasi(
    current_price: float,
    mu: float,
    sigma: float,
    days: int,
    n_sim: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Jalankan simulasi Monte Carlo dengan geometric Brownian motion.

    Menggunakan vektorisasi NumPy penuh — jauh lebih cepat daripada loop Python.

    Args:
        current_price: Harga awal simulasi.
        mu: Rata-rata log-return harian (dari periode yang relevan).
        sigma: Standar deviasi log-return harian (dari periode yang relevan).
        days: Jumlah hari proyeksi ke depan.
        n_sim: Jumlah jalur simulasi.
        seed: Random seed untuk reproduktibilitas hasil.

    Returns:
        Array 1D berisi harga akhir dari setiap simulasi (shape: n_sim,).
    """
    rng = np.random.default_rng(seed)
    # Shape (days, n_sim): vektorisasi penuh, tanpa loop Python
    log_returns = rng.normal(mu, sigma, size=(days, n_sim))
    finals = current_price * np.exp(np.sum(log_returns, axis=0))
    return finals


# ════════════════════════════════════════════════
# PEMBUATAN HTML TABEL
# ════════════════════════════════════════════════

def buat_tabel_distribusi(finals: np.ndarray) -> Tuple[str, float, float, float]:
    """
    Buat tabel HTML distribusi peluang dari hasil simulasi.

    Returns:
        Tuple (html_string, total_peluang_top3, rentang_bawah, rentang_atas).
    """
    bins = np.linspace(finals.min(), finals.max(), 10)
    counts, _ = np.histogram(finals, bins=bins)
    probs = counts / len(finals) * 100
    idx_sorted = np.argsort(probs)[::-1]

    table_html = (
        "<table><thead><tr>"
        "<th>Peluang</th><th>Rentang Harga (US$)</th>"
        "</tr></thead><tbody>"
    )

    total_peluang = 0.0
    rentang_bawah = float("inf")
    rentang_atas = 0.0

    for rank, id_sort in enumerate(idx_sorted):
        if probs[id_sort] == 0:
            continue
        low = bins[id_sort]
        high = bins[id_sort + 1] if id_sort + 1 < len(bins) else bins[-1]
        pct = format_persen_indonesia(probs[id_sort])
        table_html += (
            f"<tr><td>{pct}</td>"
            f"<td>{format_angka_indonesia(low)} – {format_angka_indonesia(high)}</td></tr>"
        )
        if rank < 3:
            total_peluang += probs[id_sort]
            rentang_bawah = min(rentang_bawah, low)
            rentang_atas = max(rentang_atas, high)

    table_html += (
        f"<tr><td colspan='2'>"
        f"Peluang kumulatif tiga rentang teratas: "
        f"<strong>{format_persen_indonesia(total_peluang)}</strong>, "
        f"kisaran US${format_angka_indonesia(rentang_bawah)} – "
        f"US${format_angka_indonesia(rentang_atas)}."
        f"</td></tr>"
        "</tbody></table>"
    )

    return table_html, total_peluang, rentang_bawah, rentang_atas


def buat_tabel_statistik(finals: np.ndarray) -> Tuple[str, float, float, float, float, float]:
    """
    Hitung statistik dan buat tabel HTML ringkasan dari hasil simulasi.

    Returns:
        Tuple (html_string, harga_mean, chance_above_mean, std_dev, skewness, mean_log).
    """
    mean_log = float(np.mean(np.log(finals)))
    harga_mean = float(np.exp(mean_log))
    chance_above_mean = float(np.mean(finals > harga_mean) * 100)
    std_dev = float(np.std(finals))
    skewness = float(pd.Series(finals).skew())

    kesimpulan_skew = interpretasi_skewness(skewness)

    kesimpulan = (
        f"Berdasarkan hasil simulasi, <strong>median geometrik</strong> harga diperkirakan "
        f"<strong>US${format_angka_indonesia(harga_mean)}</strong>. "
        f"Terdapat peluang <strong>{format_persen_indonesia(chance_above_mean)}</strong> "
        f"harga berada di atas angka tersebut. "
        f"Fluktuasi harga tercermin dari <strong>Standard Deviation</strong> sebesar "
        f"<strong>US${format_angka_indonesia(std_dev)}</strong>. "
        f"{kesimpulan_skew}"
    )

    stat_html = f"""
<br>
<table>
<thead><tr><th>Statistik</th><th>Nilai</th></tr></thead>
<tbody>
<tr><td>Mean Log-Return Kumulatif</td><td>{format_angka_indonesia(mean_log)}</td></tr>
<tr><td>Median Geometrik Simulasi</td><td>US${format_angka_indonesia(harga_mean)}</td></tr>
<tr><td>Peluang di Atas Median Geometrik</td><td>{format_persen_indonesia(chance_above_mean)}</td></tr>
<tr><td>Standard Deviation</td><td>US${format_angka_indonesia(std_dev)}</td></tr>
<tr><td>Skewness</td><td>{format_angka_indonesia(skewness)}</td></tr>
<tr><td colspan="2"><strong>Kesimpulan:</strong><br>{kesimpulan}</td></tr>
</tbody>
</table>
"""
    return stat_html, harga_mean, chance_above_mean, std_dev, skewness, mean_log


# ════════════════════════════════════════════════
# TAMPILAN ANTARMUKA UTAMA
# ════════════════════════════════════════════════

# Header waktu realtime WIB
wib = pytz.timezone("Asia/Jakarta")
waktu_sekarang = datetime.now(wib).strftime("%A, %d %B %Y")
st.markdown(
    f"""
    <div style='background-color:#5B5B5B;padding:8px;border-radius:8px;
                text-align:center;font-weight:bold;font-size:16px;color:white;'>
        ⏰ {waktu_sekarang}
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Proyeksi Harga Kripto Metode Monte Carlo")
st.markdown(
    "_Simulasi Monte Carlo berbasis data historis harga penutupan dari CoinGecko, "
    "dengan periode data yang disesuaikan per horizon proyeksi "
    "untuk estimasi volatilitas yang lebih relevan._"
)

# CSS global tabel
st.markdown(
    """
    <style>
    table { width:100%; border-collapse:collapse; }
    th {
        background-color:#5B5B5B; font-weight:bold; color:white;
        padding:6px; text-align:left; border:1px solid white;
    }
    td { border:1px solid white; padding:6px; text-align:left; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Input ───
ticker_input = st.selectbox("Pilih simbol kripto:", TICKER_OPTIONS)
if not ticker_input:
    st.stop()

coin_id = COINGECKO_MAP[ticker_input]

# ─── Ambil Data (satu kali, di-cache) ───
with st.spinner("Mengambil data historis dari CoinGecko…"):
    try:
        df = ambil_data_harga(coin_id)
    except (ConnectionError, ValueError) as e:
        st.error(str(e))
        st.stop()

# Harga penutupan terbaru
current_price = df["Close"].iloc[-1]
tanggal_terakhir = df.index[-1].strftime("%d %B %Y")

st.write(
    f"**Harga penutupan {ticker_input} per {tanggal_terakhir}: "
    f"US${format_angka_indonesia(current_price)}**"
)

# Random seed deterministik: ticker + tanggal + harga penutupan
today_str = datetime.now().strftime("%Y-%m-%d")
seed = hash((ticker_input, today_str, round(current_price, 6))) % (2 ** 32)

# ─── Simulasi untuk Setiap Horizon Waktu ───
for days in HORIZONS:
    periode = HORIZON_TO_PERIOD[days]

    # Hitung mu dan sigma dari periode yang relevan untuk horizon ini
    mu, sigma = hitung_parameter(df, periode)

    st.subheader(f"Proyeksi {ticker_input} — {days} Hari ke Depan")
    st.caption(f"Parameter volatilitas dihitung dari {periode} hari terakhir.")

    with st.spinner(f"Menjalankan 100.000 simulasi untuk {days} hari…"):
        finals = jalankan_simulasi(
            current_price=current_price,
            mu=mu,
            sigma=sigma,
            days=days,
            n_sim=100_000,
            seed=seed,
        )

    # Tabel distribusi peluang
    tabel_dist_html, total_peluang, rentang_bawah, rentang_atas = buat_tabel_distribusi(finals)
    st.markdown(tabel_dist_html, unsafe_allow_html=True)

    # Tabel statistik & kesimpulan
    tabel_stat_html, harga_mean, chance_above_mean, _, _, _ = buat_tabel_statistik(finals)
    st.markdown(tabel_stat_html, unsafe_allow_html=True)

    # Teks media sosial
    social_text = (
        f"Berdasarkan simulasi Monte Carlo, ada peluang {format_persen_indonesia(total_peluang)} "
        f"bagi {ticker_input} bergerak antara US${format_angka_indonesia(rentang_bawah)} "
        f"hingga US${format_angka_indonesia(rentang_atas)} dalam {days} hari ke depan, "
        f"dengan peluang {format_persen_indonesia(chance_above_mean)} berada di atas "
        f"median geometrik US${format_angka_indonesia(harga_mean)}."
    )
    st.text_area(
        label="Teks untuk Media Sosial",
        value=social_text,
        height=100,
        key=f"social_{days}",
    )
