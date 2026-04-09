"""Halaman 1: Dashboard Perbandingan Model"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pages.utils import load_evaluasi, load_forecast_test, load_cv, load_historis, COLORS

def show():
    st.title("📊 Dashboard Perbandingan Model")
    st.markdown(
        "Hasil evaluasi komprehensif tiga model forecasting pada data volume penumpang "
        "harian KRL Yogyakarta–Solo · Periode test: **13 Nov 2025 – 31 Jan 2026**"
    )

    data   = load_evaluasi()
    ev     = data["evaluasi"]
    dm     = data["dm_test"]
    mpdh   = data["mape_per_hari"]
    cph    = data["cv_per_horizon"]
    df_test = load_forecast_test()
    df_hist = load_historis()

    # ── Kartu metrik ringkasan ─────────────────────────────────
    st.subheader("Ringkasan Performa di Data Test")
    st.caption("★ = nilai terbaik pada metrik tersebut")

    metrik_list = [
        ("MAE", "penumpang/hari", False),
        ("RMSE", "penumpang/hari", False),
        ("MAPE", "%", False),
        ("SMAPE", "%", False),
        ("Coverage", "% (95% CI)", True),
        ("CV_MAPE", "% (CV 7-hari)", False),
    ]

    cols = st.columns(3)
    for idx_model, model in enumerate(["Prophet", "SARIMAX", "Hybrid"]):
        with cols[idx_model]:
            is_best = model == "SARIMAX"
            badge   = "🏆 **Model Terbaik**" if is_best else f"*{ev[model]['type']}*"
            st.markdown(f"### {model}")
            st.caption(badge)
            st.caption(f"`{ev[model]['order']}`")
            st.divider()
            for key, unit, higher in metrik_list:
                val       = ev[model][key]
                all_vals  = [ev[m][key] for m in ev]
                is_best_v = (val == min(all_vals)) if not higher else (val == max(all_vals))
                star = " ★" if is_best_v else ""
                label_map = {
                    "MAE": "MAE", "RMSE": "RMSE", "MAPE": "MAPE",
                    "SMAPE": "SMAPE", "Coverage": "Coverage 95% CI", "CV_MAPE": "CV MAPE"
                }
                fmt = f"{val:.2f}{unit}" if key not in ["MAE", "RMSE"] else f"{val:,.0f} {unit}"
                st.metric(label=label_map[key] + star, value=fmt)

    st.divider()

    # ── Tab utama ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Prediksi vs Aktual",
        "📊 Metrik Perbandingan",
        "🔁 Cross Validation",
        "📅 MAPE per Hari",
        "🔬 Diebold-Mariano Test",
    ])

    # ─ Tab 1: Plot prediksi vs aktual ─────────────────────────
    with tab1:
        st.subheader("Prediksi vs Aktual — Periode Test")
        st.caption(
            "Plot berikut menggunakan data prediksi yang dihasilkan pipeline penelitian. "
            "Silakan pilih model yang ingin ditampilkan."
        )

        model_pilih = st.multiselect(
            "Tampilkan model:",
            ["Prophet", "SARIMAX", "Hybrid"],
            default=["SARIMAX"],
            key="model_test_multi",
        )
        tampil_ci = st.checkbox("Tampilkan 95% Confidence Interval", value=True)

        fig = go.Figure()

        # Data historis 30 hari sebelum test
        df_pre = df_hist.tail(30)
        fig.add_trace(go.Scatter(
            x=df_pre["tgl"], y=df_pre["volume"],
            mode="lines",
            name="Historis (sebelum test)",
            line=dict(color=COLORS["aktual"], width=1.2, dash="dot"),
            opacity=0.5,
        ))

        # Aktual test
        fig.add_trace(go.Scatter(
            x=df_test["tanggal"], y=df_test["aktual"],
            mode="lines+markers",
            name="Aktual (test)",
            line=dict(color=COLORS["aktual"], width=2),
            marker=dict(size=4),
        ))

        for m in model_pilih:
            col_pred = f"pred_{m.lower()}"
            col_lo   = f"ci_lower_{m.lower()}"
            col_hi   = f"ci_upper_{m.lower()}"

            if tampil_ci:
                fig.add_trace(go.Scatter(
                    x=pd.concat([df_test["tanggal"], df_test["tanggal"][::-1]]),
                    y=pd.concat([df_test[col_hi], df_test[col_lo][::-1]]),
                    fill="toself",
                    fillcolor=COLORS[m] + "22",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{m} 95% CI",
                    showlegend=True,
                ))

            fig.add_trace(go.Scatter(
                x=df_test["tanggal"], y=df_test[col_pred],
                mode="lines",
                name=f"{m} (MAPE={ev[m]['MAPE']:.2f}%)",
                line=dict(color=COLORS[m], width=2),
            ))

        fig.add_vline(
            x="2025-11-13", line_dash="dash",
            line_color="gray", opacity=0.6,
            annotation_text="Mulai Periode Test",
        )
        fig.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Volume Penumpang",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            height=420,
            hovermode="x unified",
            plot_bgcolor="white",
        )
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Cara membaca:** Garis hitam solid = volume aktual penumpang. "
            "Semakin dekat garis prediksi ke garis aktual, semakin akurat modelnya. "
            "Band berwarna = interval kepercayaan 95% — aktual idealnya berada di dalam band."
        )

    # ─ Tab 2: Bar chart metrik ─────────────────────────────────
    with tab2:
        st.subheader("Perbandingan Metrik Evaluasi")

        metrik_pilih = st.selectbox(
            "Pilih metrik:",
            ["MAPE (%)", "MAE", "RMSE", "SMAPE (%)", "Coverage (%)", "CV MAPE (%)"],
            index=0,
        )
        key_map = {
            "MAPE (%)": "MAPE", "MAE": "MAE", "RMSE": "RMSE",
            "SMAPE (%)": "SMAPE", "Coverage (%)": "Coverage", "CV MAPE (%)": "CV_MAPE",
        }
        mkey   = key_map[metrik_pilih]
        models = list(ev.keys())
        vals   = [ev[m][mkey] for m in models]
        colors = [COLORS[m] for m in models]

        fig2 = go.Figure(go.Bar(
            x=models, y=vals,
            marker_color=colors,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            width=0.5,
        ))

        # Threshold dan anotasi
        if mkey == "MAPE":
            fig2.add_hline(y=10, line_dash="dash", line_color="red",
                           opacity=0.6, annotation_text="Threshold 10%",
                           annotation_position="right")
        if mkey == "Coverage":
            fig2.add_hline(y=85, line_dash="dash", line_color="green",
                           opacity=0.6, annotation_text="Target 85%",
                           annotation_position="right")

        higher_better = mkey == "Coverage"
        catatan = "(lebih rendah = lebih baik)" if not higher_better else "(lebih tinggi = lebih baik)"
        fig2.update_layout(
            title=f"{metrik_pilih} — {catatan}",
            yaxis_title=metrik_pilih,
            height=380,
            plot_bgcolor="white",
            showlegend=False,
        )
        fig2.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig2, use_container_width=True)

        # Tabel lengkap
        st.subheader("Tabel Perbandingan Lengkap")
        rows = []
        labels = {"MAE":"MAE", "RMSE":"RMSE", "MAPE":"MAPE (%)", "SMAPE":"SMAPE (%)",
                  "Coverage":"Coverage 95% CI (%)", "CV_MAPE":"CV MAPE (%)"}
        for mk, lbl in labels.items():
            all_v  = {m: ev[m][mk] for m in ev}
            hb     = mk == "Coverage"
            best_v = max(all_v.values()) if hb else min(all_v.values())
            row    = {"Metrik": lbl}
            for m in ["Prophet", "SARIMAX", "Hybrid"]:
                v   = all_v[m]
                fmt = f"{v:,.0f}" if mk in ["MAE","RMSE"] else f"{v:.2f}"
                row[m] = fmt + " ★" if v == best_v else fmt
            rows.append(row)
        df_tbl = pd.DataFrame(rows)
        st.dataframe(df_tbl, use_container_width=True, hide_index=True)
        st.caption("★ = nilai terbaik untuk metrik tersebut")

    # ─ Tab 3: Cross Validation ────────────────────────────────
    with tab3:
        st.subheader("Cross Validation MAPE per Horizon")
        st.caption(
            "Protokol identik untuk semua model: initial=180 hari, period=30 hari, horizon=7 hari. "
            "Menunjukkan performa model pada berbagai jendela waktu prediksi."
        )

        df_cv = pd.DataFrame([
            {"horizon": h, **cph[str(h) if str(h) in cph else h]}
            for h in range(1, 8)
        ])

        # Gunakan data dari JSON
        fig3 = go.Figure()
        for m in ["Prophet", "SARIMAX", "Hybrid"]:
            vals_cv = [cph[str(h)][m] if str(h) in cph else cph[h][m] for h in range(1,8)]
            fig3.add_trace(go.Scatter(
                x=list(range(1, 8)),
                y=vals_cv,
                mode="lines+markers",
                name=f"{m} (avg {ev[m]['CV_MAPE']:.1f}%)",
                line=dict(color=COLORS[m], width=2.5),
                marker=dict(size=8),
            ))

        fig3.add_hline(y=10, line_dash="dot", line_color="red",
                       opacity=0.5, annotation_text="Threshold 10%",
                       annotation_position="right")
        fig3.update_layout(
            xaxis_title="Horizon Prediksi (hari ke depan)",
            yaxis_title="MAPE (%)",
            xaxis=dict(tickmode="linear", dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            height=380,
            plot_bgcolor="white",
            hovermode="x unified",
        )
        fig3.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig3, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.success(
                "**SARIMAX & Hybrid:** MAPE stabil di ~7.5% untuk semua horizon 1–7 hari. "
                "Sangat konsisten dan andal untuk prediksi jangka pendek."
            )
        with col_b:
            st.warning(
                "**Prophet:** MAPE meningkat drastis dari 14% (hari 1) hingga 30% (hari 6–7). "
                "Tidak stabil untuk horizon yang lebih jauh."
            )

    # ─ Tab 4: MAPE per hari ───────────────────────────────────
    with tab4:
        st.subheader("MAPE per Hari dalam Seminggu")
        st.caption("Menunjukkan akurasi prediksi berdasarkan karakteristik hari.")

        hari_list  = list(mpdh.keys())
        fig4 = go.Figure()
        for m in ["Prophet", "SARIMAX", "Hybrid"]:
            fig4.add_trace(go.Bar(
                name=m,
                x=hari_list,
                y=[mpdh[h][m] for h in hari_list],
                marker_color=COLORS[m],
                text=[f"{mpdh[h][m]:.1f}%" for h in hari_list],
                textposition="outside",
            ))

        fig4.add_hline(y=10, line_dash="dot", line_color="red", opacity=0.4)
        fig4.update_layout(
            barmode="group",
            xaxis_title="Hari",
            yaxis_title="MAPE (%)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=400,
            plot_bgcolor="white",
        )
        fig4.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig4, use_container_width=True)

        # Tabel
        rows_day = []
        for hari in hari_list:
            best_m = min(mpdh[hari], key=mpdh[hari].get)
            rows_day.append({
                "Hari": hari,
                "Prophet (%)": f"{mpdh[hari]['Prophet']:.2f}",
                "SARIMAX (%)": f"{mpdh[hari]['SARIMAX']:.2f}",
                "Hybrid (%)":  f"{mpdh[hari]['Hybrid']:.2f}",
                "Terbaik": best_m,
            })
        st.dataframe(pd.DataFrame(rows_day), use_container_width=True, hide_index=True)
        st.info(
            "SARIMAX unggul di 6 dari 7 hari. Hybrid hanya unggul di hari Minggu (7.11% vs SARIMAX 8.40%). "
            "Perbedaan ini tidak cukup mengubah kesimpulan keseluruhan."
        )

    # ─ Tab 5: DM Test ─────────────────────────────────────────
    with tab5:
        st.subheader("Diebold-Mariano Test (α = 0.05)")
        st.caption(
            "Uji signifikansi statistik perbedaan akurasi antar model. "
            "H₀: tidak ada perbedaan signifikan. p < 0.05 → perbedaan signifikan."
        )

        st.info(
            "**Mengapa DM Test penting?** MAPE 9.75% vs 10.72% secara visual terlihat berbeda, "
            "tapi apakah perbedaan ini signifikan secara statistik atau sekadar kebetulan variasi sampling? "
            "DM Test menjawab pertanyaan ini — wajib ada dalam penelitian komparasi forecasting untuk jurnal ilmiah."
        )

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

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.success(
                "**Prophet vs SARIMAX** — p = 0.0000\n\n"
                "SARIMAX terbukti secara statistik lebih akurat dari Prophet."
            )
        with col_b:
            st.warning(
                "**Prophet vs Hybrid** — p = 0.8055\n\n"
                "Tidak signifikan. Prophet dan Hybrid setara secara statistik — "
                "Hybrid tidak memberikan nilai tambah dibanding Prophet."
            )
        with col_c:
            st.success(
                "**SARIMAX vs Hybrid** — p = 0.0000\n\n"
                "SARIMAX terbukti secara statistik lebih akurat dari Hybrid."
            )

        st.divider()
        st.markdown(
            "**Kesimpulan:** SARIMAX adalah satu-satunya model yang terbukti secara statistik "
            "lebih baik dari kedua model lainnya. Kompleksitas Hybrid Sequential tidak memberikan "
            "improvement bermakna untuk karakteristik data KRL dengan 1 siklus tahunan."
        )
