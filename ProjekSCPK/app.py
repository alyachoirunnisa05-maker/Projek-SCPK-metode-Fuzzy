import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fuzzy import bangun_sistem_fuzzy, hitung_skor, get_rule_base_table

st.set_page_config(page_title="SCPK", page_icon="💻", layout="wide")


# Bangun sistem fuzzy sekali saja (243 rule), lalu disimpan di cache resource
@st.cache_resource
def get_sistem():
    return bangun_sistem_fuzzy()

sistem_ctrl, performa, biaya, waktu_render, kompatibilitas, umur_pakai, skor = get_sistem()

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
    performa_input = st.number_input("Performa", min_value=0)
    biaya_input = st.number_input("Biaya", min_value=0)
    render_input = st.number_input("Waktu Render", min_value=0)
with c2:
    kompatibilitas_input = st.number_input("Kompatibilitas", min_value=0)
    umur_input = st.number_input("Umur Pakai", min_value=0)
if st.button("Simpan Perubahan"):
    idx = st.session_state.data[
        st.session_state.data["Alternatif"] == nama
    ].index[0]
    st.session_state.data.loc[idx, "Performa"] = performa_input
    st.session_state.data.loc[idx, "Biaya"] = biaya_input
    st.session_state.data.loc[idx, "Waktu Render"] = render_input
    st.session_state.data.loc[idx, "Kompatibilitas"] = kompatibilitas_input
    st.session_state.data.loc[idx, "Umur Pakai"] = umur_input
    st.success(f"Data {nama} berhasil diperbarui")
    st.rerun()
st.divider()
st.header("2. Grafik Derajat Keanggotaan")

# Daftar variabel untuk diplot: (label, objek fuzzy, nama-nama himpunan)
variabel_plot = {
    "Performa": (performa, ["rendah", "sedang", "tinggi"]),
    "Biaya": (biaya, ["murah", "sedang", "mahal"]),
    "Waktu Render": (waktu_render, ["cepat", "sedang", "lambat"]),
    "Kompatibilitas": (kompatibilitas, ["rendah", "sedang", "tinggi"]),
    "Umur Pakai": (umur_pakai, ["pendek", "sedang", "panjang"]),
    "Skor Rekomendasi": (skor, ["buruk", "cukup", "baik"]),
}

col1, col2 = st.columns(2)
for i, (label, (var, set_names)) in enumerate(variabel_plot.items()):
    fig, ax = plt.subplots(figsize=(5, 3))
    for nama_himpunan in set_names:
        ax.plot(var.universe, var[nama_himpunan].mf, label=nama_himpunan)
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True)
    if i % 2 == 0:
        col1.pyplot(fig)
    else:
        col2.pyplot(fig)
st.divider()
st.header("Rule Base Fuzzy")
st.write(
    f"Sistem ini memiliki total **{len(get_rule_base_table())} rule** "
    "yang di-generate otomatis dari seluruh kombinasi himpunan fuzzy "
    "(3 himpunan x 5 variabel input = 3^5 kombinasi)."
)
with st.expander("Lihat semua rule"):
    st.dataframe(get_rule_base_table(), use_container_width=True)
st.divider()
st.header("3. Hasil Akhir")
hasil = []
for _, row in st.session_state.data.iterrows():
    nilai_skor = hitung_skor(
        sistem_ctrl,
        row['Performa'],
        row['Biaya'],
        row['Waktu Render'],
        row['Kompatibilitas'],
        row['Umur Pakai']
    )
    hasil.append({
        "Alternatif": row["Alternatif"],
        "Nilai": round(nilai_skor, 2)
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