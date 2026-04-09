"""Halaman 3: Tentang Penelitian"""
import streamlit as st
import pandas as pd
from pages.utils import load_evaluasi, COLORS

def show():
    data = load_evaluasi()
    ev   = data["evaluasi"]
    dm   = data["dm_test"]

    st.title("📋 Tentang Penelitian")
    st.markdown(
        "Dokumentasi lengkap penelitian komparasi metode forecasting "
        "pada data volume penumpang harian KRL Yogyakarta–Solo."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Latar Belakang & Dataset",
        "⚙️ Metodologi CRISP-DM",
        "🤖 Detail Tiga Model",
        "📊 Hasil & Kesimpulan",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([3, 2])

        with col_a:
            st.subheader("Latar Belakang")
            st.markdown(
                "KRL Yogyakarta–Solo merupakan layanan transportasi commuter penting "
                "di koridor Yogyakarta–Klaten–Solo. Pengelolaan kapasitas operasional "
                "yang efisien membutuhkan prediksi volume penumpang yang akurat untuk "
                "mendukung keputusan seperti penjadwalan gerbong, alokasi petugas, "
                "dan perencanaan pemeliharaan.\n\n"
                "Penelitian ini membandingkan tiga pendekatan forecasting time series — "
                "**Prophet**, **SARIMAX**, dan **Hybrid Sequential** — dengan penekanan "
                "pada penanganan efek Lebaran dalam kalender Masehi, yang menjadi "
                "tantangan utama dalam pemodelan data transportasi Indonesia."
            )

            st.subheader("Rumusan Masalah")
            st.markdown(
                "1. Bagaimana proses penerapan metode **Hybrid SARIMAX-Prophet** "
                "dalam memprediksi jumlah penumpang harian KRL Yogyakarta-Solo?\n\n"
                "2. Model mana yang memberikan **performa terbaik** dalam konteks "
                "dataset KRL Yogyakarta-Solo ini?"
            )

            st.subheader("Tujuan Penelitian")
            st.markdown(
                "1. Menerapkan model Prophet, SARIMAX, dan Hybrid SARIMAX-Prophet "
                "untuk prediksi penumpang harian KRL Yogyakarta-Solo.\n\n"
                "2. Menganalisis performa ketiga model berdasarkan metrik evaluasi "
                "MAE, RMSE, MAPE, SMAPE, dan Coverage CI.\n\n"
                "3. Menentukan model terbaik berdasarkan akurasi dan stabilitas prediksi "
                "menggunakan Diebold-Mariano Test."
            )

        with col_b:
            st.subheader("Informasi Dataset")
            info = {
                "Sumber Data":       "PT Kereta Commuter Indonesia (KCI)",
                "Akses Data":        "Website Resmi PT KCI",
                "Periode":           "1 Jan 2025 – 31 Jan 2026",
                "Total Observasi":   "396 hari",
                "Rata-rata Harian":  "±25.840 penumpang",
                "Volume Minimum":    "14.888 penumpang",
                "Volume Maksimum":   "45.536 penumpang",
                "Missing Values":    "0 (data bersih)",
                "Granularitas":      "Harian",
            }
            for k, v in info.items():
                st.markdown(f"**{k}:** {v}")

            st.divider()
            st.subheader("Preprocessing Utama")
            st.markdown(
                "- Lebaran 2025 (17 hari) diinterpolasi linear\n"
                "- Dummy granular: pra/puncak/pasca Lebaran\n"
                "- Split kronologis 80-20 (tidak random)\n"
                "- Protokol CV identik untuk semua model"
            )

    # ── Tab 2 ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Kerangka CRISP-DM")
        st.markdown(
            "Cross Industry Standard Process for Data Mining — "
            "6 fase iteratif yang digunakan dalam penelitian ini."
        )

        cols = st.columns(6)
        fase_info = [
            ("1", "Business\nUnderstanding"),
            ("2", "Data\nUnderstanding"),
            ("3", "Data\nPreparation"),
            ("4", "Modeling"),
            ("5", "Evaluation"),
            ("6", "Deployment"),
        ]
        for col, (num, title) in zip(cols, fase_info):
            with col:
                st.metric(label=f"Fase {num}", value=title.replace("\n", " "))

        st.divider()

        fase_detail = [
            ("Fase 1 — Business Understanding", [
                "Identifikasi kebutuhan prediksi volume penumpang KRL untuk operasional KAI Commuter.",
                "Rumusan masalah: perbandingan tiga metode forecasting.",
                "Penentuan metrik evaluasi: MAE, RMSE, MAPE, SMAPE, Coverage CI.",
            ]),
            ("Fase 2 — Data Understanding", [
                "EDA: time series plot, distribusi per hari dan bulan.",
                "Identifikasi outlier: spike Lebaran mencapai 45.536 penumpang.",
                "ADF Test untuk stasioneritas (d=0, data sudah stasioner).",
                "ACF & PACF untuk identifikasi parameter SARIMAX.",
            ]),
            ("Fase 3 — Data Preparation", [
                "Interpolasi linear periode Lebaran (17 hari).",
                "Pembuatan dummy granular: is_lebaran_pra, is_lebaran_puncak, is_lebaran_pasca.",
                "Train-test split kronologis 80-20 (TIDAK random — wajib untuk time series).",
                "Protokol CV identik: initial=180 hari, period=30 hari, horizon=7 hari.",
            ]),
            ("Fase 4 — Modeling", [
                "Prophet: fourier_order=3, changepoint_prior_scale=0.05, holidays_prior_scale=30, multiplicative.",
                "SARIMAX(2,0,0)(2,1,1)[7]: parameter dari Auto ARIMA, exog=is_holiday (satu-satunya signifikan).",
                "Hybrid Sequential: Prophet(Layer1) + SARIMAX(residual_clipped)(Layer2).",
                "Perbaikan hybrid: residual di-clip IQR [−8.472, 7.946] sebelum masuk SARIMAX Layer 2.",
            ]),
            ("Fase 5 — Evaluation", [
                "Metrik: MAE, RMSE, MAPE, SMAPE, Coverage 95% CI.",
                "Cross Validation rolling window dengan protokol identik untuk semua model.",
                "Diebold-Mariano Test (α=0.05) — wajib untuk jurnal ilmiah.",
                "Analisis MAPE per hari dalam seminggu.",
                "Kesimpulan: SARIMAX terbaik (MAPE 9.75%, Coverage 97.5%).",
            ]),
            ("Fase 6 — Deployment", [
                "Implementasi sebagai aplikasi web Streamlit 3 halaman.",
                "Dashboard perbandingan model (data statis dari hasil penelitian).",
                "Halaman prediksi: horizon 7/14/30 hari, rekomendasi operasional otomatis.",
                "Hosting: Streamlit Community Cloud.",
            ]),
        ]

        for title, points in fase_detail:
            with st.expander(title, expanded=False):
                for p in points:
                    st.markdown(f"- {p}")

    # ── Tab 3 ─────────────────────────────────────────────────
    with tab3:
        st.subheader("Detail Tiga Model yang Dibandingkan")

        for model, color in COLORS.items():
            if model == "aktual":
                continue
            is_best = model == "SARIMAX"
            with st.expander(
                f"{'🏆 ' if is_best else ''}{model} — {ev[model]['type']}",
                expanded=is_best,
            ):
                col_l, col_r = st.columns([3, 2])
                with col_l:
                    if model == "Prophet":
                        st.markdown(
                            "Dikembangkan oleh Meta (Facebook) Research. Mendekomposisi "
                            "time series menjadi komponen tren, musiman, dan holiday secara "
                            "eksplisit. Sangat baik untuk data dengan pola musiman kompleks.\n\n"
                            "**Konfigurasi yang digunakan:**\n"
                            "- `yearly_seasonality=False`, custom `fourier_order=3`\n"
                            "- `changepoint_prior_scale=0.05` (trend stabil)\n"
                            "- `seasonality_mode='multiplicative'`\n"
                            "- `holidays_prior_scale=30`\n"
                            "- Holiday granular: lebaran_pra, lebaran_puncak, lebaran_pasca\n\n"
                            "**Kelebihan:** Interpretabilitas tinggi, komponen visual mudah dijelaskan.\n\n"
                            "**Keterbatasan:** MAPE 15.88%, trend menurun tidak sesuai logika bisnis, "
                            "tidak stabil di CV (18.75%)."
                        )
                    elif model == "SARIMAX":
                        st.markdown(
                            "Model statistik klasik yang menangkap autokorelasi jangka pendek "
                            "dan pola musiman. Parameter dipilih melalui Auto ARIMA berdasarkan AIC.\n\n"
                            "**Konfigurasi yang digunakan:**\n"
                            "- `SARIMAX(2, 0, 0)(2, 1, 1)[7]`\n"
                            "- `exog = is_holiday` (satu-satunya yang signifikan, p=0.000)\n"
                            "- `is_weekend` dihapus (p=1.000, sudah tertangkap seasonal AR)\n"
                            "- `is_lebaran` dihapus (p=0.820, data di-exclude saat training)\n\n"
                            "**Kelebihan:** MAPE 9.75%, Coverage 97.5%, CV stabil 7.54%, "
                            "residual sudah white noise (Ljung-Box p=0.83).\n\n"
                            "**Terbukti terbaik** berdasarkan DM Test vs kedua model lainnya."
                        )
                    else:
                        st.markdown(
                            "Menggabungkan Prophet (Layer 1) untuk pola makro dan SARIMAX (Layer 2) "
                            "untuk autokorelasi residual.\n\n"
                            "**Arsitektur:**\n"
                            "Prophet → Residual Prophet → Clip IQR → SARIMAX → Koreksi → "
                            "Prediksi Final = Prediksi Prophet + Koreksi SARIMAX\n\n"
                            "**Konfigurasi:**\n"
                            "- Layer 1: Prophet (sama seperti model Prophet)\n"
                            "- Clipping residual: [-8,472 ; 7,946] (13 hari dari 290)\n"
                            "- Layer 2: SARIMAX(1,0,0)(2,1,0)[7] pada residual (tanpa exog)\n\n"
                            "**Hasil:** CV MAPE 7.50% (hampir setara SARIMAX 7.54%), "
                            "namun MAPE test 16.72%. DM Test: SARIMAX signifikan lebih baik (p=0.0000)."
                        )
                with col_r:
                    st.metric("MAE", f"{ev[model]['MAE']:,}")
                    st.metric("MAPE", f"{ev[model]['MAPE']:.2f}%")
                    st.metric("Coverage", f"{ev[model]['Coverage']:.1f}%")
                    st.metric("CV MAPE", f"{ev[model]['CV_MAPE']:.2f}%")

    # ── Tab 4 ─────────────────────────────────────────────────
    with tab4:
        st.subheader("Hasil Evaluasi Lengkap")

        rows_tbl = []
        metrik_list = [
            ("MAE","MAE",False), ("RMSE","RMSE",False),
            ("MAPE","MAPE (%)",False), ("SMAPE","SMAPE (%)",False),
            ("Coverage","Coverage 95% CI (%)",True),
            ("CV_MAPE","CV MAPE rata-rata (%)",False),
        ]
        for mk, lbl, hb in metrik_list:
            all_v  = {m: ev[m][mk] for m in ev}
            best_v = max(all_v.values()) if hb else min(all_v.values())
            row = {"Metrik": lbl}
            for m in ["Prophet","SARIMAX","Hybrid"]:
                v   = all_v[m]
                fmt = f"{v:,.0f}" if mk in ["MAE","RMSE"] else f"{v:.2f}"
                row[m] = fmt + " ★" if v == best_v else fmt
            rows_tbl.append(row)

        st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)
        st.caption(
            "★ = nilai terbaik | Data test: 13 Nov 2025–31 Jan 2026 (79 hari) | "
            "CV: initial=180d, period=30d, horizon=7d"
        )

        st.divider()
        st.subheader("Diebold-Mariano Test (α = 0.05)")

        rows_dm = []
        for dm_row in dm:
            rows_dm.append({
                "Perbandingan": dm_row["pair"],
                "DM Statistik": f"{dm_row['stat']:+.4f}",
                "p-value": f"{dm_row['pval']:.4f}",
                "Signifikan?": "✅ Ya (p < 0.05)" if dm_row["sig"] else "❌ Tidak",
                "Model Lebih Baik": dm_row["better"],
            })
        st.dataframe(pd.DataFrame(rows_dm), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Kesimpulan Penelitian")

        st.success(
            "🏆 **SARIMAX(2,0,0)(2,1,1)[7] adalah model terbaik.**\n\n"
            "MAPE 9.75% (di bawah threshold 10%), Coverage 97.5% (melampaui target 85%), "
            "CV MAPE 7.54% (stabil di semua horizon). Terbukti signifikan lebih baik "
            "dari kedua model lain berdasarkan DM Test (p=0.0000)."
        )
        st.info(
            "📊 **Prophet unggul dalam interpretabilitas.**\n\n"
            "Komponen visual (tren, seasonal, holiday) mudah dijelaskan ke stakeholder "
            "non-teknis meskipun MAPE 15.88% tidak kompetitif."
        )
        st.warning(
            "🔄 **Hybrid Sequential tidak lebih baik untuk data ini.**\n\n"
            "CV MAPE Hybrid 7.50% hampir setara SARIMAX 7.54%, namun MAPE test 16.72% "
            "dan DM Test menunjukkan SARIMAX signifikan lebih baik (p=0.0000). "
            "Sequential Decomposition kurang efektif untuk dataset dengan 1 siklus tahunan."
        )

        st.divider()
        st.subheader("Limitasi Penelitian")
        st.markdown(
            "- Data hanya 1 siklus tahunan — yearly seasonality belum terverifikasi lintas tahun\n"
            "- Tidak ada external regressors (cuaca, jumlah perjalanan, harga tiket)\n"
            "- Posisi Lebaran bergeser ~11 hari/tahun — model perlu diperbarui setiap siklus\n"
            "- Spike error besar di 30 Des 2025 dan 13 Jan 2026 terjadi di semua model "
            "(kemungkinan event tidak terdokumentasi)"
        )

        st.subheader("Rekomendasi Pengembangan")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Jangka Pendek**")
            st.markdown(
                "Retrain model SARIMAX setiap bulan dengan data terbaru. "
                "Update definisi tanggal Lebaran sesuai penetapan pemerintah."
            )
        with col_b:
            st.markdown("**Jangka Menengah**")
            st.markdown(
                "Kumpulkan data 2–3 tahun. Eksplorasi penambahan external regressors "
                "(data cuaca, jadwal perjalanan KRL)."
            )
        with col_c:
            st.markdown("**Jangka Panjang**")
            st.markdown(
                "Integrasi dengan sistem operasional KAI secara real-time. "
                "Eksplorasi model deep learning (LSTM) dengan data yang lebih banyak."
            )

        st.divider()
        st.caption(
            "Penelitian Forecasting Volume Penumpang KRL Yogyakarta–Solo · "
            "Data: PT KCI Jan 2025–Jan 2026 · "
            "Metodologi: CRISP-DM · "
            "Model: Prophet | SARIMAX | Hybrid"
        )
