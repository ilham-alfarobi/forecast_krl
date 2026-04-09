import streamlit as st

st.set_page_config(
    page_title="KRL Forecast — KAI Commuter",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: navigasi dengan radio ────────────────────────────
with st.sidebar:
    st.title("🚆 KRL Forecast")
    st.caption("Prediksi Volume Penumpang\nKRL Yogyakarta–Solo")
    st.divider()

    halaman = st.radio(
        "Pilih Halaman",
        options=["📊 Dashboard Model", "🔮 Prediksi Penumpang", "📋 Tentang Penelitian"],
        index=0,
        key="nav",
    )

    st.divider()
    st.caption(
        "**Sumber Data:** PT KCI\n\n"
        "**Periode:** Jan 2025 – Jan 2026\n\n"
        "**Model Terbaik:** SARIMAX\n"
        "MAPE: 9.75% | Coverage: 97.5%\n\n"
        "**Metodologi:** CRISP-DM"
    )

# ── Routing ───────────────────────────────────────────────────
if halaman == "📊 Dashboard Model":
    from pages.dashboard import show
    show()
elif halaman == "🔮 Prediksi Penumpang":
    from pages.prediksi import show
    show()
elif halaman == "📋 Tentang Penelitian":
    from pages.penelitian import show
    show()
