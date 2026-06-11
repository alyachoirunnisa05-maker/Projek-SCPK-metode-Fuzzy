import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fuzzy import hitung_skor
from fuzzy import get_membership_data

st.set_page_config(page_title="SCPK", page_icon="💻", layout="wide")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame({
        "Alternatif": ["RAM", "SSD", "GPU", "CPU"],
        "Performa": [6500, 5000, 9000, 8000],
        "Biaya": [800000, 1000000, 4000000, 3500000],
        "Waktu Render": [40, 45, 25, 30],
        "Kompatibilitas": [95, 98, 85, 75],
        "Umur Pakai": [3, 4, 5, 4]
    })

st.title("💻 Sistem Pendukung Keputusan")
st.subheader("Pemilihan Upgrade Komponen PC dengan Metode Fuzzy Mamdani")

st.header("1. Data Awal")

st.dataframe(st.session_state.data, use_container_width=True)

st.header("Input Data")

nama = st.selectbox("Pilih Alternatif", ["RAM", "SSD", "GPU", "CPU"])

c1, c2 = st.columns(2)

with c1:
    performa = st.number_input("Performa", min_value=0)
    biaya = st.number_input("Biaya", min_value=0)
    render = st.number_input("Waktu Render", min_value=0)

with c2:
    kompatibilitas = st.number_input("Kompatibilitas", min_value=0)
    umur = st.number_input("Umur Pakai", min_value=0)

if st.button("Simpan Perubahan"):
    idx = st.session_state.data[
        st.session_state.data["Alternatif"] == nama
    ].index[0]

    st.session_state.data.loc[idx, "Performa"] = performa
    st.session_state.data.loc[idx, "Biaya"] = biaya
    st.session_state.data.loc[idx, "Waktu Render"] = render
    st.session_state.data.loc[idx, "Kompatibilitas"] = kompatibilitas
    st.session_state.data.loc[idx, "Umur Pakai"] = umur

    st.success(f"Data {nama} berhasil diperbarui")
    st.rerun()

st.divider()

st.header("2. Grafik Derajat Keanggotaan")

membership_data = get_membership_data()

col1, col2 = st.columns(2)

for i, (nama_variabel, data) in enumerate(membership_data.items()):

    fig, ax = plt.subplots(figsize=(5, 3))

    for nama_himpunan, mf in data["mf"].items():
        ax.plot(data["universe"], mf, label=nama_himpunan)

    ax.set_title(data["label"])
    ax.legend(fontsize=8)
    ax.grid(True)

    if i % 2 == 0:
        col1.pyplot(fig)
    else:
        col2.pyplot(fig)

st.divider()

st.header("3. Hasil Akhir")

hasil = []

for _, row in st.session_state.data.iterrows():

    skor, detail, _ = hitung_skor(
        row['Performa'],
        row['Biaya'],
        row['Waktu Render'],
        row['Kompatibilitas'],
        row['Umur Pakai']
    )

    hasil.append({
        "Alternatif": row["Alternatif"],
        "Nilai": round(skor, 2)
    })

hasil = pd.DataFrame(hasil)

hasil = hasil.sort_values(
    by="Nilai",
    ascending=False
).reset_index(drop=True)

st.dataframe(
    hasil,
    use_container_width=True
)

if not hasil.empty:
    terbaik = hasil.iloc[0]["Alternatif"]
    nilai = hasil.iloc[0]["Nilai"]

    st.success(f"Alternatif terbaik adalah {terbaik} dengan nilai akhir {nilai}")

fig, ax = plt.subplots(figsize=(6, 3))

ax.bar(
    hasil["Alternatif"],
    hasil["Nilai"]
)

ax.set_title("Ranking Alternatif")
ax.set_ylabel("Nilai")

st.pyplot(fig)