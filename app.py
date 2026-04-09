from __future__ import annotations

import hashlib
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

HORIZON_TO_PERIOD: dict = {
    3:   60,
    7:   60,
    30:  180,
    90:  365,
    365: 365,
}

HORIZONS = [3, 7, 30, 90, 365]
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
# CSS GLOBAL
# ════════════════════════════════════════════════

st.markdown("""
<style>
table { width: 100%; border-collapse: collapse; }
th {
    background-color: #5B5B5B;
    font-weight: bold;
    color: white;
    padding: 6px 10px;
    text-align: left;
    border: 1px solid rgba(255,255,255,0.3);
}
td {
    border: 1px solid rgba(128,128,128,0.3);
    padding: 6px 10px;
    text-align: left;
}
/* Baris highlight: hijau tua (#3B6D11) agar teks gelap terbaca di atas background */
tr.top-row td {
    background-color: #C0DD97;
    color: #173404;
    font-weight: bold;
}
/* Baris keterangan di bawah tabel distribusi */
tr.keterangan-row td {
    background-color: #EAF3DE;
    color: #3B6D11;
    font-size: 0.85em;
    font-style: italic;
    border-top: 2px solid #97C459;
}
/* Baris persentil: merah untuk turun, hijau untuk naik */
td.chg-up   { color: #27500A; font-weight: bold; }
td.chg-down { color: #791F1F; font-weight: bold; }
/* Baris kesimpulan statistik */
tr.kesimpulan-row td {
    background-color: rgba(128,128,128,0.08);
    font-size: 0.9em;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# UTILITAS FORMAT
# ════════════════════════════════════════════════

def fmt(val) -> str:
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


def pct(val) -> str:
    """Format persen ke format Indonesia."""
    try:
        val = float(val)
    except (TypeError, ValueError):
        return str(val)
    return f"{val:.1f}".replace(".", ",") + "%"


def pct_chg(val: float, base: float) -> Tuple[str, bool]:
    """
    Hitung persentase perubahan dari base.
    Returns (teks_format, is_up).
    """
    p = (val - base) / base * 100
    is_up = p >= 0
    arah = "naik" if is_up else "turun"
    return f"{arah} {abs(p):.1f}%".replace(".", ","), is_up


def interpretasi_skewness(skewness: float) -> str:
    skew_fmt = fmt(skewness)
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
# DATA
# ════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def ambil_data_harga(coin_id: str) -> pd.DataFrame:
    """
    Ambil data harga historis harian dari CoinGecko.
    Cache selama 1 jam. Satu panggilan untuk semua horizon.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(MAX_PERIOD)}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise ConnectionError(
            "Permintaan ke CoinGecko habis waktu (timeout). Coba lagi beberapa saat."
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 429:
            raise ConnectionError(
                "Batas permintaan API CoinGecko terlampaui (429). "
                "Tunggu beberapa menit lalu coba lagi."
            )
        raise ConnectionError(
            f"API CoinGecko mengembalikan error HTTP {status}. "
            "Periksa koneksi internet atau coba lagi."
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Tidak dapat terhubung ke CoinGecko. Periksa koneksi internet Anda."
        )

    prices = resp.json().get("prices", [])
    if len(prices) < 60:
        raise ValueError(
            "Data historis tidak mencukupi (minimal 60 hari). Coba pilih koin lain."
        )

    dates  = [datetime.fromtimestamp(p[0] / 1000).date() for p in prices]
    closes = [p[1] for p in prices]
    return pd.DataFrame({"Date": dates, "Close": closes}).set_index("Date")


def hitung_parameter(df: pd.DataFrame, periode: int) -> Tuple[float, float]:
    """Hitung mu & sigma log-return dari N hari terakhir."""
    n_slice = min(periode + 1, len(df))
    df_slice = df.iloc[-n_slice:]
    log_ret = np.log(df_slice["Close"] / df_slice["Close"].shift(1)).dropna()
    return float(log_ret.mean()), float(log_ret.std())


def jalankan_simulasi(
    current_price: float,
    mu: float,
    sigma: float,
    days: int,
    seed: int,
) -> np.ndarray:
    """Monte Carlo GBM, vektorisasi penuh (100.000 jalur)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(mu, sigma, size=(days, 100_000))
    return current_price * np.exp(np.sum(log_returns, axis=0))

# ════════════════════════════════════════════════
# KOMPONEN HTML — FITUR 1: METRIC CARDS
# ════════════════════════════════════════════════

def render_metric_cards(
    median: float,
    chance: float,
    std: float,
    current_price: float,
) -> None:
    """Tampilkan tiga metric card: median geometrik, peluang naik, std dev."""
    med_chg, med_up = pct_chg(median, current_price)
    std_pct = abs(std / current_price * 100)

    med_color  = "#27500A" if med_up else "#791F1F"
    pct_color  = "#27500A" if chance >= 50 else "#791F1F"

    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:1rem;">
  <div style="background:var(--background-color,#f8f9fa);border-radius:8px;padding:14px 16px;border:0.5px solid rgba(128,128,128,0.2);">
    <div style="font-size:12px;color:gray;margin-bottom:4px;">Median geometrik</div>
    <div style="font-size:18px;font-weight:600;">US${fmt(median)}</div>
    <div style="font-size:11px;color:{med_color};margin-top:2px;">{med_chg} dari harga kini</div>
  </div>
  <div style="background:var(--background-color,#f8f9fa);border-radius:8px;padding:14px 16px;border:0.5px solid rgba(128,128,128,0.2);">
    <div style="font-size:12px;color:gray;margin-bottom:4px;">Peluang di atas median</div>
    <div style="font-size:18px;font-weight:600;color:{pct_color};">{pct(chance)}</div>
    <div style="font-size:11px;color:gray;margin-top:2px;">dari seluruh simulasi</div>
  </div>
  <div style="background:var(--background-color,#f8f9fa);border-radius:8px;padding:14px 16px;border:0.5px solid rgba(128,128,128,0.2);">
    <div style="font-size:12px;color:gray;margin-bottom:4px;">Std deviation</div>
    <div style="font-size:18px;font-weight:600;">US${fmt(std)}</div>
    <div style="font-size:11px;color:gray;margin-top:2px;">±{pct(std_pct)} dari harga kini</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# FITUR 3: SKENARIO BULL / BASE / BEAR
# ════════════════════════════════════════════════

def render_skenario(finals: np.ndarray, current_price: float, days: int) -> None:
    """Tampilkan tiga kartu skenario berdasarkan P10, P50, P90."""
    p10 = float(np.percentile(finals, 10))
    p50 = float(np.percentile(finals, 50))
    p90 = float(np.percentile(finals, 90))

    bear_chg, _  = pct_chg(p10, current_price)
    base_chg, _  = pct_chg(p50, current_price)
    bull_chg, _  = pct_chg(p90, current_price)

    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:1rem;">
  <div style="background:#FCEBEB;border:0.5px solid #F09595;border-radius:8px;padding:14px 16px;">
    <div style="font-size:11px;font-weight:600;color:#A32D2D;margin-bottom:4px;">🐻 Bear — P10</div>
    <div style="font-size:16px;font-weight:600;color:#791F1F;">US${fmt(p10)}</div>
    <div style="font-size:11px;color:#A32D2D;margin-top:2px;">{bear_chg}</div>
  </div>
  <div style="background:#E6F1FB;border:0.5px solid #85B7EB;border-radius:8px;padding:14px 16px;">
    <div style="font-size:11px;font-weight:600;color:#185FA5;margin-bottom:4px;">📊 Base — P50</div>
    <div style="font-size:16px;font-weight:600;color:#0C447C;">US${fmt(p50)}</div>
    <div style="font-size:11px;color:#185FA5;margin-top:2px;">{base_chg}</div>
  </div>
  <div style="background:#EAF3DE;border:0.5px solid #97C459;border-radius:8px;padding:14px 16px;">
    <div style="font-size:11px;font-weight:600;color:#3B6D11;margin-bottom:4px;">🐂 Bull — P90</div>
    <div style="font-size:16px;font-weight:600;color:#27500A;">US${fmt(p90)}</div>
    <div style="font-size:11px;color:#3B6D11;margin-top:2px;">{bull_chg}</div>
  </div>
</div>
<p style="font-size:11px;color:gray;margin-top:-6px;margin-bottom:1rem;">
  Berdasarkan persentil hasil simulasi · horizon {days} hari
</p>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# FITUR 4: TABEL PERSENTIL
# ════════════════════════════════════════════════

def render_tabel_persentil(finals: np.ndarray, current_price: float) -> None:
    """Tabel P10–P90 dengan warna merah/hijau pada kolom perubahan."""
    persentil_list = [10, 25, 50, 75, 90]
    rows = ""
    for p in persentil_list:
        val = float(np.percentile(finals, p))
        chg_txt, is_up = pct_chg(val, current_price)
        cls = "chg-up" if is_up else "chg-down"
        rows += (
            f"<tr>"
            f"<td>P{p}</td>"
            f"<td>US${fmt(val)}</td>"
            f"<td class='{cls}'>{chg_txt}</td>"
            f"</tr>"
        )

    st.markdown(f"""
<table>
  <thead>
    <tr>
      <th>Persentil</th>
      <th>Harga (US$)</th>
      <th>Perubahan dari harga kini</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TABEL DISTRIBUSI PELUANG
# ════════════════════════════════════════════════

def render_tabel_distribusi(
    finals: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Tabel distribusi 9 rentang harga diurutkan dari peluang tertinggi.
    Baris teratas (peluang max) diberi warna hijau dengan teks gelap.
    Returns (total_peluang_top3, rentang_bawah, rentang_atas).
    """
    bins  = np.linspace(finals.min(), finals.max(), 10)
    counts, _ = np.histogram(finals, bins=bins)
    probs = counts / len(finals) * 100
    idx_sorted = np.argsort(probs)[::-1]

    total_peluang = 0.0
    rentang_bawah = float("inf")
    rentang_atas  = 0.0
    rows = ""

    for rank, id_sort in enumerate(idx_sorted):
        if probs[id_sort] == 0:
            continue
        low  = bins[id_sort]
        high = bins[id_sort + 1] if id_sort + 1 < len(bins) else bins[-1]

        # Kelas CSS top-row hanya untuk baris dengan peluang TERTINGGI
        row_class = ' class="top-row"' if rank == 0 else ""
        rows += (
            f"<tr{row_class}>"
            f"<td>{pct(probs[id_sort])}</td>"
            f"<td>{fmt(low)} – {fmt(high)}</td>"
            f"</tr>"
        )

        if rank < 3:
            total_peluang += probs[id_sort]
            rentang_bawah  = min(rentang_bawah, low)
            rentang_atas   = max(rentang_atas, high)

    # Baris keterangan warna
    rows += (
        "<tr class='keterangan-row'>"
        "<td colspan='2'>"
        "Baris hijau = rentang dengan peluang tertinggi"
        "</td></tr>"
    )

    st.markdown(f"""
<table>
  <thead>
    <tr><th>Peluang</th><th>Rentang harga (US$)</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

    st.markdown(
        f"Peluang kumulatif tiga rentang teratas: **{pct(total_peluang)}**, "
        f"kisaran US${fmt(rentang_bawah)} – US${fmt(rentang_atas)}."
    )

    return total_peluang, rentang_bawah, rentang_atas

# ════════════════════════════════════════════════
# TABEL STATISTIK
# ════════════════════════════════════════════════

def render_tabel_statistik(finals: np.ndarray) -> Tuple[float, float]:
    """Tabel statistik ringkasan + kesimpulan. Returns (harga_mean, chance)."""
    mean_log  = float(np.mean(np.log(finals)))
    harga_mean = float(np.exp(mean_log))
    chance    = float(np.mean(finals > harga_mean) * 100)
    std_dev   = float(np.std(finals))
    skewness  = float(pd.Series(finals).skew())

    kesimpulan = (
        f"Median geometrik diperkirakan <strong>US${fmt(harga_mean)}</strong>. "
        f"Terdapat peluang <strong>{pct(chance)}</strong> harga berada di atas angka tersebut. "
        f"Fluktuasi tercermin dari std deviation <strong>US${fmt(std_dev)}</strong>. "
        f"{interpretasi_skewness(skewness)}"
    )

    st.markdown(f"""
<table>
  <thead><tr><th>Statistik</th><th>Nilai</th></tr></thead>
  <tbody>
    <tr><td>Mean log-return kumulatif</td><td>{fmt(mean_log)}</td></tr>
    <tr><td>Median geometrik simulasi</td><td>US${fmt(harga_mean)}</td></tr>
    <tr><td>Peluang di atas median geometrik</td><td>{pct(chance)}</td></tr>
    <tr><td>Standard deviation</td><td>US${fmt(std_dev)}</td></tr>
    <tr><td>Skewness</td><td>{fmt(skewness)}</td></tr>
    <tr class="kesimpulan-row">
      <td colspan="2"><strong>Kesimpulan:</strong><br>{kesimpulan}</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

    return harga_mean, chance

# ════════════════════════════════════════════════
# FITUR 2: GRAFIK DISTRIBUSI (PLOTLY)
# ════════════════════════════════════════════════

def render_grafik_distribusi(
    finals: np.ndarray,
    current_price: float,
    harga_mean: float,
    days: int,
) -> None:
    """
    Histogram distribusi 100.000 harga akhir simulasi menggunakan Plotly.
    Garis vertikal biru = harga terkini, hijau = median geometrik.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly untuk menampilkan grafik distribusi.")
        return

    bins   = np.linspace(finals.min(), finals.max(), 30)
    counts, edges = np.histogram(finals, bins=bins)
    probs  = counts / len(finals) * 100
    labels = [fmt(e) for e in edges[:-1]]

    # Warna: bar tertinggi lebih gelap
    max_idx = int(np.argmax(probs))
    colors  = ["#85B7EB"] * len(probs)
    colors[max_idx] = "#185FA5"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(probs))),
        y=probs.tolist(),
        marker_color=colors,
        hovertemplate="Rentang: %{customdata}<br>Peluang: %{y:.1f}%<extra></extra>",
        customdata=labels,
        name="Distribusi",
    ))

    # Garis vertikal: harga terkini
    cur_bin = int(np.searchsorted(edges, current_price, side="right")) - 1
    cur_bin = max(0, min(cur_bin, len(probs) - 1))
    fig.add_vline(
        x=cur_bin,
        line_dash="dash",
        line_color="#185FA5",
        line_width=1.5,
        annotation_text="Harga kini",
        annotation_font_size=11,
        annotation_font_color="#185FA5",
    )

    # Garis vertikal: median geometrik
    med_bin = int(np.searchsorted(edges, harga_mean, side="right")) - 1
    med_bin = max(0, min(med_bin, len(probs) - 1))
    if med_bin != cur_bin:
        fig.add_vline(
            x=med_bin,
            line_dash="dot",
            line_color="#3B6D11",
            line_width=1.5,
            annotation_text="Median",
            annotation_font_size=11,
            annotation_font_color="#3B6D11",
        )

    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            tickvals=list(range(0, len(probs), 4)),
            ticktext=[labels[i] for i in range(0, len(probs), 4)],
            tickfont=dict(size=10),
            showgrid=False,
        ),
        yaxis=dict(
            title="Peluang (%)",
            tickfont=dict(size=10),
            gridcolor="rgba(128,128,128,0.1)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.05,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bar biru gelap = rentang peluang tertinggi · "
        "Garis biru putus = harga terkini · "
        "Garis hijau titik = median geometrik"
    )

# ════════════════════════════════════════════════
# FITUR 5: EKSPANDER METODOLOGI + DISCLAIMER
# ════════════════════════════════════════════════

def render_ekspander_metodologi(periode: int, days: int) -> None:
    with st.expander("ℹ️ Cara kerja simulasi ini"):
        st.markdown(f"""
**Metode:** Geometric Brownian Motion (GBM) — model standar pergerakan harga aset keuangan.
Setiap jalur simulasi dibangun dari log-return harian acak yang terdistribusi normal
dengan parameter **mu** (rata-rata return) dan **sigma** (volatilitas).

**Periode data:** Parameter untuk horizon **{days} hari** dihitung dari
**{periode} hari terakhir** — bukan seluruh 365 hari — agar estimasi volatilitas
mencerminkan kondisi pasar terkini, bukan rata-rata jangka panjang yang sudah tidak relevan.

**Jumlah simulasi:** 100.000 jalur independen, menghasilkan distribusi harga akhir
yang stabil secara statistik dan mendekati distribusi log-normal teoritis.

**Seed deterministik:** Hasil simulasi untuk ticker, tanggal, dan harga yang sama
selalu menghasilkan angka yang identik — sehingga bisa direproduksi dan dibandingkan.
""")

    st.warning(
        "⚠️ **Disclaimer:** Hasil simulasi ini bersifat probabilistik dan **bukan saran investasi**. "
        "Kripto adalah aset dengan risiko tinggi dan volatilitas ekstrem. "
        "Pergerakan harga aktual bisa berbeda jauh dari proyeksi simulasi.",
        icon=None,
    )

# ════════════════════════════════════════════════
# DOWNLOAD CSV
# ════════════════════════════════════════════════

def buat_csv(
    finals: np.ndarray,
    current_price: float,
    ticker: str,
    days: int,
) -> str:
    """Buat string CSV dari persentil dan distribusi peluang."""
    lines = [f"Proyeksi Monte Carlo — {ticker} — {days} hari\n"]

    lines.append("Persentil,Harga (USD),Perubahan (%)")
    for p in [10, 25, 50, 75, 90]:
        val = float(np.percentile(finals, p))
        chg = (val - current_price) / current_price * 100
        lines.append(f"P{p},{val:.2f},{chg:.2f}%")

    lines.append("\nPeluang (%),Rentang Bawah (USD),Rentang Atas (USD)")
    bins = np.linspace(finals.min(), finals.max(), 10)
    counts, _ = np.histogram(finals, bins=bins)
    probs = counts / len(finals) * 100
    for i, p in enumerate(probs):
        lines.append(f"{p:.2f}%,{bins[i]:.2f},{bins[i+1]:.2f}")

    return "\n".join(lines)

# ════════════════════════════════════════════════
# ANTARMUKA UTAMA
# ════════════════════════════════════════════════

# Header tanggal WIB
wib = pytz.timezone("Asia/Jakarta")
today_wib = datetime.now(wib)
waktu_str  = today_wib.strftime("%A, %d %B %Y")

st.markdown(f"""
<div style="background-color:#5B5B5B;padding:8px;border-radius:8px;
            text-align:center;font-weight:bold;font-size:15px;color:white;
            margin-bottom:1rem;">
    ⏰ {waktu_str}
</div>
""", unsafe_allow_html=True)

st.title("Proyeksi Harga Kripto Metode Monte Carlo")
st.markdown(
    "_Simulasi Monte Carlo berbasis data historis harga penutupan dari CoinGecko, "
    "dengan periode data yang disesuaikan per horizon proyeksi "
    "untuk estimasi volatilitas yang lebih relevan._"
)

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Pengaturan")

    ticker_input = st.selectbox("Pilih simbol kripto", TICKER_OPTIONS)

    days = st.radio(
        "Horizon proyeksi",
        HORIZONS,
        format_func=lambda x: f"{x} Hari",
    )

    st.divider()
    st.caption(
        f"Periode data untuk {days} hari: "
        f"**{HORIZON_TO_PERIOD[days]} hari terakhir**"
    )
    st.caption("Data: CoinGecko · Cache: 1 jam")

# ─── Ambil Data ───
coin_id = COINGECKO_MAP[ticker_input]

with st.spinner("Mengambil data historis dari CoinGecko…"):
    try:
        df = ambil_data_harga(coin_id)
    except (ConnectionError, ValueError) as e:
        st.error(str(e))
        st.stop()

# Harga terkini (titik awal simulasi) dan harga kemarin (sudah final, untuk ditampilkan)
current_price  = df["Close"].iloc[-1]
harga_tampil   = df["Close"].iloc[-2]
tanggal_tampil = df.index[-2].strftime("%d %B %Y")

st.write(
    f"**Harga penutupan {ticker_input} per {tanggal_tampil}: "
    f"US${fmt(harga_tampil)}**"
    f" _(simulasi dimulai dari harga terkini: US${fmt(current_price)})_"
)

# ─── Parameter & Seed ───
periode = HORIZON_TO_PERIOD[days]
mu, sigma = hitung_parameter(df, periode)

today_str = today_wib.strftime("%Y-%m-%d")
seed_str  = f"{ticker_input}-{today_str}-{round(current_price, 6)}"
seed      = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)

# ─── Simulasi ───
with st.spinner(f"Menjalankan 100.000 simulasi untuk {days} hari…"):
    finals = jalankan_simulasi(current_price, mu, sigma, days, seed + days)

# ─── Highlight peluang terbesar ───
bins_tmp  = np.linspace(finals.min(), finals.max(), 10)
counts_tmp, _ = np.histogram(finals, bins=bins_tmp)
probs_tmp = counts_tmp / len(finals) * 100
top_idx   = int(np.argmax(probs_tmp))
top_low   = bins_tmp[top_idx]
top_high  = bins_tmp[top_idx + 1]

st.success(
    f"Peluang terbesar: **{pct(probs_tmp[top_idx])}** "
    f"— kisaran US${fmt(top_low)} hingga US${fmt(top_high)}"
)

st.divider()

# ─── 1. Metric Cards ───
st.subheader(f"Proyeksi {ticker_input} — {days} Hari ke Depan")
st.caption(f"Parameter volatilitas dihitung dari {periode} hari terakhir · 100.000 simulasi")

harga_mean_tmp = float(np.exp(np.mean(np.log(finals))))
chance_tmp     = float(np.mean(finals > harga_mean_tmp) * 100)
std_tmp        = float(np.std(finals))

render_metric_cards(harga_mean_tmp, chance_tmp, std_tmp, current_price)

st.divider()

# ─── 2. Grafik Distribusi ───
st.markdown("**Grafik distribusi simulasi**")
st.caption(
    "Distribusi 100.000 harga akhir simulasi. "
    "Bar biru gelap = peluang tertinggi · "
    "Garis biru = harga kini · Garis hijau = median geometrik."
)
render_grafik_distribusi(finals, current_price, harga_mean_tmp, days)

st.divider()

# ─── 3. Skenario Bull / Base / Bear ───
st.markdown("**Skenario Bull / Base / Bear**")
render_skenario(finals, current_price, days)

st.divider()

# ─── 4. Tabel Persentil ───
st.markdown("**Tabel persentil**")
render_tabel_persentil(finals, current_price)

st.divider()

# ─── Distribusi Peluang (tabel lengkap) ───
st.markdown("**Distribusi peluang**")
total_peluang, rentang_bawah, rentang_atas = render_tabel_distribusi(finals)

st.divider()

# ─── Statistik ───
st.markdown("**Statistik**")
harga_mean, chance = render_tabel_statistik(finals)

st.divider()

# ─── 5. Ekspander Metodologi + Disclaimer ───
render_ekspander_metodologi(periode, days)

st.divider()

# ─── Teks Media Sosial ───
chg_low_txt,  _  = pct_chg(rentang_bawah, current_price)
chg_high_txt, _  = pct_chg(rentang_atas,  current_price)

social_text = (
    f"Simulasi Monte Carlo menunjukkan peluang {pct(total_peluang)} "
    f"bagi {ticker_input} bergerak antara US${fmt(rentang_bawah)} "
    f"hingga US${fmt(rentang_atas)} dalam {days} hari ke depan, "
    f"dengan potensi {chg_low_txt} hingga {chg_high_txt} dari harga saat ini."
)

st.text_area(
    label="Teks untuk media sosial",
    value=social_text,
    height=90,
    key=f"social_{ticker_input}_{days}",
)

# ─── Download CSV ───
csv_data = buat_csv(finals, current_price, ticker_input, days)
st.download_button(
    label="⬇️ Unduh hasil sebagai CSV",
    data=csv_data,
    file_name=f"monte_carlo_{ticker_input}_{days}hari.csv",
    mime="text/csv",
)
