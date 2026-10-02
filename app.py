import io
import pandas as pd
import streamlit as st
import predictor as P

st.set_page_config(page_title="Deteksi Risiko Dropout - Jaya Jaya Institut", page_icon="🎓", layout="wide")


@st.cache_resource
def get_assets():
    return P.load_assets()


models, defaults, columns = get_assets()

# ---------------------------------------------------------------- Sidebar
st.sidebar.title("🎓 Jaya Jaya Institut")
st.sidebar.caption("Sistem peringatan dini risiko dropout")
mode = st.sidebar.radio(
    "Model yang dipakai",
    ["early", "full"],
    format_func=lambda k: {"early": "Peringatan dini (s/d semester 1)", "full": "Model penuh (s/d semester 2)"}[k],
    help="Model peringatan dini hanya butuh data pendaftaran + semester 1, sehingga bisa dipakai lebih awal.",
)
bundle = models[mode]
st.sidebar.markdown(f"**Threshold:** `{bundle['threshold']:.2f}`")
st.sidebar.markdown("**Tingkat risiko**\n- 🟢 Rendah: skor < threshold\n- 🟠 Sedang: threshold - 0,50\n- 🔴 Tinggi: skor ≥ 0,50")

st.title("Deteksi Dini Mahasiswa Berisiko Dropout")
tab1, tab2, tab3 = st.tabs(["👤 Prediksi Individu", "📂 Prediksi Massal (CSV)", "ℹ️ Tentang Model"])

# ---------------------------------------------------------------- Tab 1: individu
with tab1:
    with st.form("form_mhs"):
        st.subheader("Profil & Pendaftaran")
        c1, c2, c3 = st.columns(3)
        course = c1.selectbox("Program studi", sorted(P.COURSES), index=sorted(P.COURSES).index(9500) if 9500 in P.COURSES else 0,
                              format_func=lambda k: f"{k} - {P.COURSES[k]}")
        age = c2.number_input("Usia saat mendaftar", 17, 70, 20)
        gender = c3.selectbox("Jenis kelamin", [0, 1], format_func=lambda v: {0: "Perempuan", 1: "Laki-laki"}[v])
        c1, c2, c3 = st.columns(3)
        attendance = c1.selectbox("Kelas", [1, 0], format_func=lambda v: {1: "Siang", 0: "Malam"}[v])
        displaced = c2.selectbox("Perantau (displaced)", [0, 1], format_func=lambda v: {0: "Tidak", 1: "Ya"}[v])
        admission = c3.number_input("Nilai masuk (Admission grade)", 0.0, 200.0, float(defaults["Admission_grade"]), 0.1)
        prev_grade = c1.number_input("Nilai kualifikasi sebelumnya", 0.0, 200.0, float(defaults["Previous_qualification_grade"]), 0.1)

        st.subheader("Finansial")
        c1, c2, c3 = st.columns(3)
        tuition = c1.selectbox("UKT", [1, 0], format_func=lambda v: {1: "Lunas / terkini", 0: "Menunggak"}[v])
        debtor = c2.selectbox("Status debitur", [0, 1], format_func=lambda v: {0: "Bukan debitur", 1: "Debitur"}[v])
        scholar = c3.selectbox("Beasiswa", [0, 1], format_func=lambda v: {0: "Tidak", 1: "Penerima beasiswa"}[v])

        st.subheader("Akademik Semester 1")
        c1, c2, c3 = st.columns(3)
        s1_cred = c1.number_input("MK diakui (credited)", 0, 30, 0, key="s1c")
        s1_enr = c2.number_input("MK diambil (enrolled)", 0, 30, 6, key="s1e")
        s1_eval = c3.number_input("Jumlah evaluasi", 0, 60, 8, key="s1v")
        s1_app = c1.number_input("MK lulus (approved)", 0, 30, 5, key="s1a")
        s1_grade = c2.number_input("Rata-rata nilai (0-20)", 0.0, 20.0, 12.0, 0.1, key="s1g")
        s1_noev = c3.number_input("MK tanpa evaluasi", 0, 20, 0, key="s1n")

        s2 = {}
        if mode == "full":
            st.subheader("Akademik Semester 2")
            c1, c2, c3 = st.columns(3)
            s2["cred"] = c1.number_input("MK diakui (credited)", 0, 30, 0, key="s2c")
            s2["enr"] = c2.number_input("MK diambil (enrolled)", 0, 30, 6, key="s2e")
            s2["eval"] = c3.number_input("Jumlah evaluasi", 0, 60, 8, key="s2v")
            s2["app"] = c1.number_input("MK lulus (approved)", 0, 30, 5, key="s2a")
            s2["grade"] = c2.number_input("Rata-rata nilai (0-20)", 0.0, 20.0, 12.0, 0.1, key="s2g")
            s2["noev"] = c3.number_input("MK tanpa evaluasi", 0, 20, 0, key="s2n")

        with st.expander("Pengaturan lanjutan (opsional, default = nilai umum pada data)"):
            a1, a2, a3 = st.columns(3)
            adv = {}
            for i, c in enumerate(["Marital_status", "Application_mode", "Application_order", "Previous_qualification",
                                   "Nacionality", "Mothers_qualification", "Fathers_qualification",
                                   "Mothers_occupation", "Fathers_occupation"]):
                adv[c] = [a1, a2, a3][i % 3].number_input(c, value=int(defaults[c]), step=1)
            m1, m2, m3 = st.columns(3)
            adv["Unemployment_rate"] = m1.number_input("Unemployment_rate", value=float(defaults["Unemployment_rate"]))
            adv["Inflation_rate"] = m2.number_input("Inflation_rate", value=float(defaults["Inflation_rate"]))
            adv["GDP"] = m3.number_input("GDP", value=float(defaults["GDP"]))
            adv["Educational_special_needs"] = m1.selectbox("Kebutuhan khusus", [0, 1], format_func=lambda v: {0: "Tidak", 1: "Ya"}[v])
            adv["International"] = m2.selectbox("Mahasiswa internasional", [0, 1], format_func=lambda v: {0: "Tidak", 1: "Ya"}[v])

        submitted = st.form_submit_button("🔍 Prediksi risiko dropout", type="primary")

    if submitted:
        row = dict(defaults)
        row.update(adv)
        row.update({
            "Course": course, "Age_at_enrollment": age, "Gender": gender, "Daytime_evening_attendance": attendance,
            "Displaced": displaced, "Admission_grade": admission, "Previous_qualification_grade": prev_grade,
            "Tuition_fees_up_to_date": tuition, "Debtor": debtor, "Scholarship_holder": scholar,
            "Curricular_units_1st_sem_credited": s1_cred, "Curricular_units_1st_sem_enrolled": s1_enr,
            "Curricular_units_1st_sem_evaluations": s1_eval, "Curricular_units_1st_sem_approved": s1_app,
            "Curricular_units_1st_sem_grade": s1_grade, "Curricular_units_1st_sem_without_evaluations": s1_noev,
        })
        if mode == "full":
            row.update({
                "Curricular_units_2nd_sem_credited": s2["cred"], "Curricular_units_2nd_sem_enrolled": s2["enr"],
                "Curricular_units_2nd_sem_evaluations": s2["eval"], "Curricular_units_2nd_sem_approved": s2["app"],
                "Curricular_units_2nd_sem_grade": s2["grade"], "Curricular_units_2nd_sem_without_evaluations": s2["noev"],
            })

        if s1_app > s1_enr:
            st.warning("MK lulus semester 1 lebih besar dari MK diambil. Periksa kembali input.")

        res = P.predict(pd.DataFrame([row])[columns], bundle).iloc[0]
        score, tier = float(res["risk_score"]), res["risk_tier"]

        st.divider()
        left, right = st.columns([1, 2])
        left.metric("Skor risiko dropout", f"{score * 100:.1f}%")
        left.progress(min(max(score, 0.0), 1.0))
        {"Tinggi": right.error, "Sedang": right.warning, "Rendah": right.success}[tier](
            f"**Tingkat risiko: {tier}**  \n" + {
                "Tinggi": "Prioritaskan untuk bimbingan khusus.",
                "Sedang": "Perlu dipantau; pertimbangkan pendampingan ringan.",
                "Rendah": "Tidak ditandai oleh model; pemantauan rutin.",
            }[tier])
        factors = P.risk_factors(row)
        if factors:
            right.markdown("**Faktor risiko yang terdeteksi (berdasarkan pola data historis):**")
            for f in factors:
                right.markdown(f"- {f}")
        st.caption("Skor adalah estimasi probabilitas dari model statistik, bukan keputusan final. "
                   "Gunakan sebagai dasar pendekatan suportif, bukan sanksi.")

# ---------------------------------------------------------------- Tab 2: massal
with tab2:
    st.markdown("Unggah CSV berisi **36 kolom fitur** seperti `data.csv` (kolom `Status` boleh ada atau tidak). "
                "Pemisah `;` atau `,` dikenali otomatis.")
    template = pd.DataFrame([defaults])[columns]
    st.download_button("⬇️ Unduh template CSV", template.to_csv(index=False, sep=";").encode(), "template_mahasiswa.csv", "text/csv")

    up = st.file_uploader("Pilih file CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
        missing = [c for c in columns if c not in raw.columns]
        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {missing}")
        else:
            out = P.predict(raw[columns], bundle)
            if "Status" in raw.columns:
                out.insert(0, "Status", raw["Status"].values)
            out = out.sort_values("risk_score", ascending=False)
            counts = out["risk_tier"].value_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total mahasiswa", len(out))
            m2.metric("🔴 Tinggi", int(counts.get("Tinggi", 0)))
            m3.metric("🟠 Sedang", int(counts.get("Sedang", 0)))
            m4.metric("🟢 Rendah", int(counts.get("Rendah", 0)))
            st.bar_chart(counts.reindex(["Rendah", "Sedang", "Tinggi"]).fillna(0))
            show = ["Status"] if "Status" in out.columns else []
            show += ["risk_score", "risk_tier", "predicted_dropout", "Course", "Age_at_enrollment",
                     "Tuition_fees_up_to_date", "Scholarship_holder", "Curricular_units_1st_sem_approved"]
            st.dataframe(out[show], use_container_width=True, height=400)
            buf = io.StringIO(); out.to_csv(buf, index=False)
            st.download_button("⬇️ Unduh hasil lengkap (CSV)", buf.getvalue().encode(), "hasil_prediksi_risiko.csv", "text/csv")

# ---------------------------------------------------------------- Tab 3: tentang
with tab3:
    st.markdown("""
**Model:** Logistic Regression (class_weight=balanced) dengan feature engineering performa akademik.

| Metrik (data test) | Peringatan dini | Model penuh |
|---|---|---|
| Recall Dropout | 89,1% | 91,6% |
| Precision Dropout | 60,5% | 60,2% |
| ROC-AUC | 0,913 | 0,932 |

**Cara membaca:** threshold sengaja rendah agar sebagian besar calon dropout tertangkap (recall tinggi). Akibatnya sekitar 4 dari 10 mahasiswa yang ditandai sebenarnya tidak dropout.

**Keterbatasan:** model menunjukkan korelasi, bukan sebab; dilatih dari satu institusi/periode sehingga perlu dipantau dan dilatih ulang berkala.
""")
