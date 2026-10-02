"""Logika prediksi (tanpa Streamlit) agar mudah dites terpisah dari tampilan."""
import json, os
import numpy as np
import pandas as pd
import joblib

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Nama program studi menurut dokumentasi dataset (verifikasi dengan data institusi Anda)
COURSES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (kelas malam)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (kelas malam)",
}


ASSETS = os.path.dirname(os.path.abspath(__file__))


def load_assets():

    models = {
        "early": joblib.load(
            os.path.join(ASSETS, "model_dropout_early_warning.joblib")
        ),
        "full": joblib.load(
            os.path.join(ASSETS, "model_dropout.joblib")
        ),
    }

    with open(os.path.join(ASSETS, "defaults.json")) as f:
        meta = json.load(f)

    return models, meta["defaults"], meta["columns"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering IDENTIK dengan notebook (bagian Data Preparation)."""
    df = df.copy()
    for s in ["1st", "2nd"]:
        enr = df[f"Curricular_units_{s}_sem_enrolled"].replace(0, np.nan)
        df[f"Approval_rate_{s}_sem"] = (df[f"Curricular_units_{s}_sem_approved"] / enr).fillna(0)
        df[f"Eval_to_enrolled_{s}_sem"] = (df[f"Curricular_units_{s}_sem_evaluations"] / enr).fillna(0)
    df["Total_approved"] = df["Curricular_units_1st_sem_approved"] + df["Curricular_units_2nd_sem_approved"]
    df["Approved_trend"] = df["Curricular_units_2nd_sem_approved"] - df["Curricular_units_1st_sem_approved"]
    df["Grade_trend"] = df["Curricular_units_2nd_sem_grade"] - df["Curricular_units_1st_sem_grade"]
    return df


def risk_tier(score: float, threshold: float) -> str:
    if score < threshold:
        return "Rendah"
    return "Sedang" if score < 0.5 else "Tinggi"


def predict(df_raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """df_raw: kolom mentah (36 fitur). Mengembalikan df dengan risk_score, risk_tier, predicted_dropout."""
    feats = add_features(df_raw)
    score = bundle["model"].predict_proba(feats[bundle["feature_cols"]])[:, 1]
    out = df_raw.copy()
    out["risk_score"] = score.round(4)
    out["risk_tier"] = [risk_tier(s, bundle["threshold"]) for s in score]
    out["predicted_dropout"] = (score >= bundle["threshold"]).astype(int)
    return out


def risk_factors(row: dict) -> list:
    """Penjelasan sederhana berbasis aturan dari temuan EDA (bukan penjelasan model per individu)."""
    f = []
    if row.get("Tuition_fees_up_to_date") == 0: f.append("UKT menunggak (dropout 86,6% pada kelompok ini)")
    if row.get("Debtor") == 1: f.append("Berstatus debitur (dropout 62,0%)")
    if row.get("Curricular_units_1st_sem_approved", 1) == 0: f.append("Tidak ada MK lulus di semester 1 (dropout 79,4%)")
    if row.get("Scholarship_holder") == 0: f.append("Tidak menerima beasiswa (dropout 38,7% vs 12,2%)")
    if row.get("Age_at_enrollment", 0) >= 25: f.append("Usia saat mendaftar ≥ 25 tahun (dropout sekitar 55-60%)")
    if row.get("Daytime_evening_attendance") == 0: f.append("Kelas malam (dropout 42,9%)")
    return f
