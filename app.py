    if data.empty:
        st.error("Data tidak ditemukan atau kosong. Periksa kembali simbol ticker.")
    else:
        close_prices = data["Close"]
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

        # Konversi ke float agar tidak error saat format
        mu = float(log_returns.mean())
        sigma = float(log_returns.std())

        # Informasi harga terkini dan waktu lokal
        last_price = float(close_prices.iloc[-1])
        last_date = data.index[-1].date()
        now_wib = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%H:%M:%S")

        st.success(f"✅ Rata-rata log returns harian: {mu:.5f}, Standar deviasi harian: {sigma:.5f}")
        st.info(f"💰 Harga terakhir: US${last_price:,.2f} | 🗓️ Tanggal: {last_date.strftime('%d %B %Y')} | 🕒 Waktu (WIB): {now_wib}")

        for num_days in [7, 30, 90]:
            st.divider()
            st.subheader(f"🔮 Simulasi Monte Carlo untuk {num_days} hari ke depan")

            num_simulations = 1000
            simulations = np.zeros((num_days, num_simulations))

            for i in range(num_simulations):
                rand_returns = np.random.normal(mu, sigma, num_days)
                price_path = last_price * np.exp(np.cumsum(rand_returns))
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

            st.markdown(f"**[PROYEKSI HARGA {ticker_input} {num_days} HARI KE DEPAN]**")
            st.markdown(f"Tanggal awal: {start_date.strftime('%d %B %Y')}")
            st.markdown(f"Tanggal akhir: {end_date.strftime('%d %B %Y')}")
            st.code("\n".join(output), language="markdown")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
