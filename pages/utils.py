"""utils.py — Load data CSV/JSON hasil pipeline"""
import json, os
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LIBUR_NASIONAL = {
    "2025-01-01": "Tahun Baru 2025",
    "2025-01-27": "Isra Mi'raj",
    "2025-01-29": "Tahun Baru Imlek",
    "2025-03-28": "Nyepi",
    "2025-03-30": "Idul Fitri 1446H",
    "2025-03-31": "Idul Fitri 1446H",
    "2025-04-01": "Idul Fitri 1446H (H+1)",
    "2025-04-18": "Wafat Isa Al-Masih",
    "2025-05-01": "Hari Buruh",
    "2025-05-12": "Waisak",
    "2025-05-29": "Kenaikan Isa Al-Masih",
    "2025-06-01": "Hari Lahir Pancasila",
    "2025-08-17": "HUT RI ke-80",
    "2025-09-05": "Maulid Nabi",
    "2025-12-25": "Natal",
    "2025-12-26": "Cuti Bersama Natal",
    "2026-01-01": "Tahun Baru 2026",
    "2026-03-20": "Idul Fitri 1447H (perkiraan)",
    "2026-03-21": "Idul Fitri 1447H H+1 (perkiraan)",
}

COLORS = {
    "Prophet": "#2563EB",
    "SARIMAX": "#D97706",
    "Hybrid":  "#16A34A",
    "aktual":  "#111827",
}

@st.cache_data
def load_evaluasi():
    path = os.path.join(DATA_DIR, "evaluasi.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_forecast_test():
    path = os.path.join(DATA_DIR, "forecast_test.csv")
    df = pd.read_csv(path, parse_dates=["tanggal"])
    return df

@st.cache_data
def load_forecast_30():
    path = os.path.join(DATA_DIR, "forecast_30hari.csv")
    df = pd.read_csv(path, parse_dates=["tanggal"])
    return df

@st.cache_data
def load_historis():
    path = os.path.join(DATA_DIR, "data_historis.csv")
    df = pd.read_csv(path, parse_dates=["tgl"])
    return df

@st.cache_data
def load_cv():
    path = os.path.join(DATA_DIR, "cv_metrics.csv")
    df = pd.read_csv(path)
    return df

def label_hari(d):
    return ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][d.weekday()]

def get_keterangan(d):
    ds = str(d.date()) if hasattr(d, "date") else str(d)
    return LIBUR_NASIONAL.get(ds, "")
