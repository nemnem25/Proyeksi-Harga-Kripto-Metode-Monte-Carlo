import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Fungsi untuk format angka dengan format Indonesia
def format_angka_indonesia(angka):
    return f"{angka:,.0f}".replace(",", ".")

# Fungsi untuk format persen dengan format Indonesia
def format_persen_indonesia(persen):
    return f"{persen:,.2f}".replace(",", ".")

# Fungsi untuk simulasi Monte Carlo harga kripto
def simulasi_monte_carlo(df, n_simulasi, n_hari):
    # Simulasi Monte Carlo berdasarkan data harga penutupan
    hasil_simulasi = np.zeros((n_simulasi, n_hari))
    for i in range(n_simulasi):
        # Ambil harga penutupan acak dan lakukan simulasi harga
        harga_awal = df['Close'].iloc[-1]
        log_returns = np.diff(np.log(df['Close']))  # Log returns untuk analisis
        mean_return = np.mean(log_returns)
        std_deviation = np.std(log_returns)
        
        for t in range(n_hari):
            random_return = np.random.normal(mean_return, std_deviation)
            hasil_simulasi[i, t] = harga_awal * np.exp(np.cumsum(random_return))
            harga_awal = hasil_simulasi[i, t]
    
    return hasil_simulasi

# Fungsi untuk menghitung statistik dari hasil simulasi
def hitung_statistik(simulasi):
    # Statistik harga logaritmik
    log_harga = np.log(simulasi[:, -1])
    mean_harga = np.mean(log_harga)
    std_harga = np.std(log_harga)
    
    # Skewness
    skewness = skew(log_harga)
    
    # Rentang harga (kumulatif tertinggi)
    rentang_atas = np.max(simulasi[:, -1])
    rentang_bawah = np.min(simulasi[:, -1])

    return mean_harga, std_harga, skewness, rentang_atas, rentang_bawah

# Fungsi untuk menghitung peluang
def hitung_peluang(rentang_atas, rentang_bawah, hasil_simulasi):
    # Hitung peluang harga berada dalam rentang yang diharapkan
    peluang_atas = np.mean(hasil_simulasi[:, -1] >= rentang_atas) * 100
    peluang_bawah = np.mean(hasil_simulasi[:, -1] <= rentang_bawah) * 100
    total_peluang = peluang_atas + peluang_bawah
    return total_peluang, peluang_atas, peluang_bawah

# Proyeksi harga kripto menggunakan simulasi Monte Carlo
df = pd.read_csv('data_kripto.csv')  # Ganti dengan path data yang sesuai
n_simulasi = 1000
n_hari = 30

hasil_simulasi = simulasi_monte_carlo(df, n_simulasi, n_hari)
mean_harga, std_harga, skewness, rentang_atas, rentang_bawah = hitung_statistik(hasil_simulasi)
total_peluang, peluang_atas, peluang_bawah = hitung_peluang(rentang_atas, rentang_bawah, hasil_simulasi)

# Hitung persentase harga penutupan hari sebelumnya dengan proyeksi harga tertinggi
current_price = df['Close'].iloc[-1]
persentase_harga_penutupan = (current_price / rentang_atas) * 100
persentase_harga_penutupan_fmt = format_persen_indonesia(persentase_harga_penutupan)

# Menyusun tampilan statistik
mean_harga_fmt = format_angka_indonesia(np.exp(mean_harga))
std_harga_fmt = format_angka_indonesia(np.exp(std_harga))
rentang_atas_fmt = format_angka_indonesia(rentang_atas)
rentang_bawah_fmt = format_angka_indonesia(rentang_bawah)
total_peluang_fmt = format_persen_indonesia(total_peluang)
peluang_atas_fmt = format_persen_indonesia(peluang_atas)
peluang_bawah_fmt = format_persen_indonesia(peluang_bawah)

# Menyusun tabel statistik dengan format HTML
table_html = f"""
<table>
    <tr><td>Mean (Harga Logaritmik)</td><td>US${mean_harga_fmt}</td></tr>
    <tr><td>Standard Deviation</td><td>US${std_harga_fmt}</td></tr>
    <tr><td>Skewness</td><td>{skewness:.2f}</td></tr>
    <tr><td>Rentang Harga (Tertinggi)</td><td>US${rentang_atas_fmt}</td></tr>
    <tr><td>Rentang Harga (Terendah)</td><td>US${rentang_bawah_fmt}</td></tr>
    <tr class='highlight-green'><td colspan='2'>
    Peluang kumulatif dari tiga rentang harga tertinggi mencapai {total_peluang_fmt}, dengan kisaran harga US${rentang_bawah_fmt} hingga US${rentang_atas_fmt}. Artinya, berdasarkan simulasi, ada kemungkinan besar harga akan bergerak dalam kisaran tersebut dalam {n_hari} hari ke depan. Persentase harga penutupan hari sebelumnya dengan proyeksi harga tertinggi adalah {persentase_harga_penutupan_fmt}.
    </td></tr>
</table>
"""

# Tampilkan hasil proyeksi harga kripto
print(table_html)

# Visualisasi hasil simulasi
plt.plot(hasil_simulasi.T, color='blue', alpha=0.1)
plt.title(f"Proyeksi Harga Kripto - {n_hari} Hari Ke Depan")
plt.xlabel("Hari")
plt.ylabel("Harga (US$)")
plt.show()
