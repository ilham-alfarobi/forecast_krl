# 🚆 KRL Forecast — KAI Commuter

Aplikasi prediksi volume penumpang KRL Yogyakarta–Solo.

## Struktur

```
krl_v2/
├── app.py                  ← Entry point
├── generate_data.py        ← Jalankan SEKALI untuk buat file data/
├── requirements.txt
├── data/                   ← Dibuat oleh generate_data.py
│   ├── evaluasi.json
│   ├── data_historis.csv
│   ├── forecast_test.csv
│   ├── forecast_30hari.csv
│   └── cv_metrics.csv
└── pages/
    ├── utils.py
    ├── dashboard.py
    ├── prediksi.py
    └── penelitian.py
```

## Cara Pakai

### 1. Generate data (SATU KALI)
```bash
cd krl_v2
python generate_data.py
```
> Jika model pkl tersedia dari pipeline, generate_data.py sudah
> mengikutsertakan logika untuk membaca pkl. Jika tidak ada,
> data dibuat dari pola statistik hasil penelitian.

### 2. Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy ke Streamlit Community Cloud
1. Push seluruh folder `krl_v2/` ke GitHub (termasuk folder `data/`)
2. Buka https://share.streamlit.io → New app
3. Repo → branch `main` → Main file: `app.py`
4. Deploy

## Catatan Penting

- **Model frozen**: prediksi menggunakan pola dari model yang dilatih
  Jan 2025–Jan 2026. Untuk produksi, jalankan ulang `generate_data.py`
  setelah retrain model.
- **Format upload**: Excel/CSV dengan kolom `tgl` dan `volume`.
- Warna grafik: Biru=Prophet, Kuning=SARIMAX, Hijau=Hybrid.
