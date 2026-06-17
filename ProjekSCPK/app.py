import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz 
from fuzzy import bangun_sistem_fuzzy, hitung_skor, get_rule_base_table

st.set_page_config(page_title="SCPK", page_icon="💻", layout="wide")


@st.cache_resource
def get_sistem():
    return bangun_sistem_fuzzy()

sistem_ctrl, performa, biaya, waktu_render, kompatibilitas, umur_pakai, skor = get_sistem()


# OPTIMASI: get_rule_base_table() di-cache, dan dipanggil SEKALI saja,
# lalu dipakai ulang untuk hitung len() maupun ditampilkan di expander.
@st.cache_data
def get_rule_table_cached():
    return get_rule_base_table()

rule_table = get_rule_table_cached()


# OPTIMASI: hasil hitung_skor() di-cache berdasarkan nilai input saja.
@st.cache_data
def hitung_skor_cached(_sistem_ctrl, performa_val, biaya_val, waktu_render_val,
                        kompatibilitas_val, umur_pakai_val):
    return hitung_skor(_sistem_ctrl, performa_val, biaya_val, waktu_render_val,
                        kompatibilitas_val, umur_pakai_val)


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
data_terpilih = st.session_state.data[st.session_state.data["Alternatif"] == nama].iloc[0]
with c1:
    performa_input = st.number_input("Performa", min_value=0, value=int(data_terpilih["Performa"]))
    biaya_input = st.number_input("Biaya", min_value=0, value=int(data_terpilih["Biaya"]))
    render_input = st.number_input("Waktu Render", min_value=0, value=int(data_terpilih["Waktu Render"]))
with c2:
    kompatibilitas_input = st.number_input("Kompatibilitas", min_value=0, value=int(data_terpilih["Kompatibilitas"]))
    umur_input = st.number_input("Umur Pakai", min_value=0, value=int(data_terpilih["Umur Pakai"]))
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

st.header("2. Grafik Derajat Keanggotaan & Wiring Fuzzifikasi")
st.caption("Garis merah putus-putus menunjukkan nilai input saat ini beserta posisi derajat keanggotaannya (μ).")

# Definisikan mapping variabel ke input yang ada di form
variabel_plot = {
    "Performa": (performa, ["rendah", "sedang", "tinggi"], performa_input),
    "Biaya": (biaya, ["murah", "sedang", "mahal"], biaya_input),
    "Waktu Render": (waktu_render, ["cepat", "sedang", "lambat"], render_input),
    "Kompatibilitas": (kompatibilitas, ["rendah", "sedang", "tinggi"], kompatibilitas_input),
    "Umur Pakai": (umur_pakai, ["pendek", "sedang", "panjang"], umur_input),
    
    "Skor Rekomendasi": (skor, ["buruk", "cukup", "baik", "sangat baik"], None), 
}


col1, col2 = st.columns(2)
for i, (label, (var, set_names, current_input)) in enumerate(variabel_plot.items()):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    # Plot kurva keanggotaan dasar
    for nama_himpunan in set_names:
        ax.plot(var.universe, var[nama_himpunan].mf, label=nama_himpunan)
    
    # WIRING FUZZIFIKASI: Gambar nilai input jika variabel tersebut memiliki input
    if current_input is not None:
        # Batasi nilai input agar tidak keluar dari range universe (mencegah error plotting)
        input_clamped = max(min(current_input, var.universe[-1]), var.universe[0])
        
        # Gambar garis vertikal penanda nilai input
        ax.axvline(x=input_clamped, color='red', linestyle='--', alpha=0.7, label=f'Input: {current_input}')
        
        # Hitung dan plot titik potong (wiring) untuk masing-masing himpunan fuzzy
        # Ditambahkan tracker offset agar teks tidak saling tumpang tindih (overlap)
        label_count = 0 
        
        for nama_himpunan in set_names:
            # Menggunakan interp_membership untuk mencari nilai mu (y) dari nilai x (input)
            mu_val = fuzz.interp_membership(var.universe, var[nama_himpunan].mf, input_clamped)
            
            if mu_val > 0: # Hanya tampilkan jika nilai keanggotaannya lebih dari 0
                # Gambar titik potong di kurva
                ax.plot(input_clamped, mu_val, 'ro', markersize=5, zorder=5) 
                
                # menaikkan label yang aktif biar ga tumpang tindih
                y_text_position = mu_val + 0.05 + (0.12 * label_count)
                
                # Beri teks keterangan nilai mu di samping titik
                ax.text(
                    input_clamped, 
                    y_text_position, 
                    f"μ {nama_himpunan}: {mu_val:.2f}", 
                    fontsize=8, 
                    weight='bold', 
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2', edgecolor='red', lw=0.5),
                    zorder=6
                )
                
                # Naikkan counter jika label sudah digambar
                label_count += 1

    ax.set_title(label)
    ax.set_ylim(-0.1, 1.2) # Beri ruang vertikal agar teks tidak terpotong
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Bagi visualisasi ke dalam 2 kolom Streamlit
    if i % 2 == 0:
        col1.pyplot(fig)
    else:
        col2.pyplot(fig)

st.divider()
st.header("Rule Base Fuzzy")
st.write(
    f"Sistem ini memiliki total **{len(rule_table)} rule** "
    "yang di-generate otomatis dari seluruh kombinasi himpunan fuzzy "
    "(3 himpunan x 5 variabel input = 3^5 kombinasi)."
)
with st.expander("Lihat semua rule"):
    st.dataframe(rule_table, use_container_width=True)
st.divider()
st.header("3. Hasil Akhir")
hasil = []
for _, row in st.session_state.data.iterrows():
    nilai_skor = hitung_skor_cached(
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