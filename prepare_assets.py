"""Langkah 2: buat assets/defaults.json (nilai default untuk fitur yang tidak ditanyakan di form).
Jalankan sekali:  python prepare_assets.py path/ke/data.csv
"""
import json, sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
df = pd.read_csv(path, sep=";", encoding="utf-8-sig").drop(columns=["Status"])

nominal = ["Marital_status", "Application_mode", "Course", "Previous_qualification", "Nacionality",
           "Mothers_qualification", "Fathers_qualification", "Mothers_occupation", "Fathers_occupation"]
defaults = {}
for c in df.columns:
    v = df[c].mode().iloc[0] if c in nominal else df[c].median()
    defaults[c] = float(v) if df[c].dtype == "float64" else int(v)

with open("defaults.json", "w") as f:
    json.dump({"defaults": defaults, "columns": list(df.columns)}, f, indent=2)
print("OK ->", "defaults.json", f"({len(defaults)} kolom)")
