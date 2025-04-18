import sys
print("Python version:", sys.version)
print("Installed packages:")
import pkg_resources
print([p.project_name for p in pkg_resources.working_set])


import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model dan scaler
model = joblib.load('model_diabetes_rf.pkl')
scaler = joblib.load('scaler_diabetes.pkl')

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966483.png", width=100)
with col2:
    st.title("Sistem Prediksi Diabetes")
    st.markdown(
        "Aplikasi ini membantu memprediksi risiko diabetes berdasarkan data kesehatan dasar. "
        "Cocok digunakan oleh masyarakat umum maupun tenaga medis."
    )

st.markdown("---")

# Sidebar edukasi
with st.sidebar:
    st.header("Tentang Diabetes")
    st.info("""
**Diabetes** adalah kondisi kronis ketika kadar gula darah terlalu tinggi.

**Jenis utama:**
- Diabetes Tipe 1
- Diabetes Tipe 2
- Diabetes Gestasional

**Penyebab umum:** Pola makan tidak sehat, kurang aktivitas fisik, faktor keturunan.
""")
    st.success("Tips Pencegahan:\n\n- Gaya hidup sehat\n- Cek gula darah rutin\n- Kurangi konsumsi gula\n- Olahraga minimal 30 menit/hari")

    st.markdown("---")
    st.caption("Dibuat oleh Della Andini - Informatika Medis")

# Form input
st.subheader("Masukkan Data Kesehatan Anda")

with st.form("prediksi_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Kadar Glukosa", min_value=0, max_value=200)
        blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=140)
        skin_thickness = st.number_input("Ketebalan Lipatan Kulit", min_value=0, max_value=100)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, format="%.3f")
        age = st.number_input("Usia", min_value=1, max_value=120)

    submit = st.form_submit_button("Prediksi Sekarang")

# Fungsi prediksi
def prediksi_diabetes(data_input):
    data_array = np.array(data_input).reshape(1, -1)
    data_scaled = scaler.transform(data_array)
    hasil = model.predict(data_scaled)
    return "Positif Diabetes" if hasil[0] == 1 else "Negatif Diabetes"

# Tampilkan hasil prediksi
if submit:
    data_input = [pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]
    hasil = prediksi_diabetes(data_input)

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    if hasil == "Positif Diabetes":
        st.error("⚠️ Anda berisiko **Positif Diabetes**. Segera konsultasikan ke dokter.")
    else:
        st.success("✅ Anda **Tidak Terindikasi Diabetes**. Tetap jaga kesehatan!")

# Tambahan edukasi interaktif
st.markdown("---")
st.subheader("Informasi Tambahan")

with st.expander("Gejala Umum Diabetes"):
    st.write("""
    - Sering merasa haus
    - Sering buang air kecil
    - Mudah lelah
    - Berat badan turun drastis
    - Luka sulit sembuh
    """)

with st.expander("Kapan Harus Periksa?"):
    st.write("""
    - Memiliki riwayat keluarga diabetes
    - Merasa gejala di atas
    - Berat badan berlebih
    - Gaya hidup kurang sehat
    """)

with st.expander("Apa itu BMI?"):
    st.write("**BMI (Body Mass Index)** adalah ukuran lemak tubuh berdasarkan berat dan tinggi. Ideal: 18.5 - 24.9")

