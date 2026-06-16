import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Bobot tiap himpunan fuzzy, dipakai untuk generate rule base otomatis.
# 2 = paling menguntungkan, 1 = sedang, 0 = paling tidak menguntungkan
BOBOT = {
    'performa':       {'rendah': 0, 'sedang': 1, 'tinggi': 2},
    'biaya':          {'murah': 2, 'sedang': 1, 'mahal': 0},   # murah = paling baik
    'waktu_render':   {'cepat': 2, 'sedang': 1, 'lambat': 0},  # cepat = paling baik
    'kompatibilitas': {'rendah': 0, 'sedang': 1, 'tinggi': 2},
    'umur_pakai':     {'pendek': 0, 'sedang': 1, 'panjang': 2},
}

# Batas total bobot (rentang 0-10) untuk menentukan kategori output skor
BATAS_BURUK_MAX = 3   # total bobot 0-3 -> 'buruk'
BATAS_CUKUP_MAX = 6   # total bobot 4-6 -> 'cukup'
# total bobot 7-10 -> 'baik'


def _generate_rule_definitions():
    """Generate semua kombinasi rule (3^5 = 243) sekali, dipakai 2 tempat:
    bangun_sistem_fuzzy() (untuk ctrl.Rule) dan get_rule_base_table()."""
    rule_base = []
    for p in BOBOT['performa']:
        for b in BOBOT['biaya']:
            for w in BOBOT['waktu_render']:
                for k in BOBOT['kompatibilitas']:
                    for u in BOBOT['umur_pakai']:
                        total = (BOBOT['performa'][p] + BOBOT['biaya'][b]
                                 + BOBOT['waktu_render'][w] + BOBOT['kompatibilitas'][k]
                                 + BOBOT['umur_pakai'][u])
                        if total <= BATAS_BURUK_MAX:
                            hasil = 'buruk'
                        elif total <= BATAS_CUKUP_MAX:
                            hasil = 'cukup'
                        else:
                            hasil = 'baik'
                        rule_base.append({
                            'performa': p, 'biaya': b, 'waktu_render': w,
                            'kompatibilitas': k, 'umur_pakai': u,
                            'total_bobot': total, 'hasil': hasil,
                        })
    return rule_base


def bangun_sistem_fuzzy():
    """
    Membangun seluruh komponen fuzzy Mamdani dalam satu fungsi:
      1. Universe (semesta pembicaraan)
      2. Membership function tiap himpunan
      3. Rule base -> ControlSystem

    Panggil fungsi ini SEKALI saja (cache di Streamlit), karena membangun
    243 rule cukup berat kalau diulang tiap kali ada input baru.
    """

    # 1. UNIVERSE
    performa       = ctrl.Antecedent(np.arange(0, 10001, 1),   'performa')
    biaya          = ctrl.Antecedent(np.arange(0, 5000001, 1), 'biaya')
    waktu_render   = ctrl.Antecedent(np.arange(0, 61, 1),      'waktu_render')
    kompatibilitas = ctrl.Antecedent(np.arange(0, 101, 1),     'kompatibilitas')
    umur_pakai     = ctrl.Antecedent(np.arange(0, 11, 1),      'umur_pakai')
    skor           = ctrl.Consequent(np.arange(0, 101, 1),     'skor')

    # 2. MEMBERSHIP FUNCTION
    # Himpunan ekstrem (rendah/tinggi, murah/mahal, dst) pakai trapmf
    # supaya ada area datar (membership = 1). Himpunan "sedang" pakai trimf.
    performa['rendah']  = fuzz.trapmf(performa.universe, [0, 0, 2000, 5000])
    performa['sedang']  = fuzz.trimf(performa.universe, [2000, 5000, 8000])
    performa['tinggi']  = fuzz.trapmf(performa.universe, [5000, 8000, 10000, 10000])

    biaya['murah']  = fuzz.trapmf(biaya.universe, [0, 0, 500000, 2000000])
    biaya['sedang'] = fuzz.trimf(biaya.universe, [500000, 2000000, 3500000])
    biaya['mahal']  = fuzz.trapmf(biaya.universe, [2000000, 3500000, 5000000, 5000000])

    waktu_render['cepat']  = fuzz.trapmf(waktu_render.universe, [0, 0, 10, 25])
    waktu_render['sedang'] = fuzz.trimf(waktu_render.universe, [15, 30, 45])
    waktu_render['lambat'] = fuzz.trapmf(waktu_render.universe, [30, 45, 60, 60])

    kompatibilitas['rendah'] = fuzz.trapmf(kompatibilitas.universe, [0, 0, 20, 50])
    kompatibilitas['sedang'] = fuzz.trimf(kompatibilitas.universe, [30, 55, 80])
    kompatibilitas['tinggi'] = fuzz.trapmf(kompatibilitas.universe, [60, 80, 100, 100])

    umur_pakai['pendek']  = fuzz.trapmf(umur_pakai.universe, [0, 0, 1, 4])
    umur_pakai['sedang']  = fuzz.trimf(umur_pakai.universe, [2, 5, 8])
    umur_pakai['panjang'] = fuzz.trapmf(umur_pakai.universe, [5, 8, 10, 10])

    skor['buruk'] = fuzz.trapmf(skor.universe, [0, 0, 20, 50])
    skor['cukup'] = fuzz.trimf(skor.universe, [20, 50, 80])
    skor['baik']  = fuzz.trapmf(skor.universe, [50, 80, 100, 100])

    # 3. RULE BASE -> ControlSystem
    rule_base = [
        ctrl.Rule(
            performa[d['performa']] & biaya[d['biaya']] & waktu_render[d['waktu_render']]
            & kompatibilitas[d['kompatibilitas']] & umur_pakai[d['umur_pakai']],
            skor[d['hasil']]
        )
        for d in _generate_rule_definitions()
    ]

    sistem_ctrl = ctrl.ControlSystem(rule_base)

    return sistem_ctrl, performa, biaya, waktu_render, kompatibilitas, umur_pakai, skor


def get_rule_base_table():
    """Rule base dalam bentuk DataFrame, untuk ditampilkan di Streamlit."""
    df = pd.DataFrame(_generate_rule_definitions())
    return df.rename(columns={
        'performa': 'Performa', 'biaya': 'Biaya', 'waktu_render': 'Waktu Render',
        'kompatibilitas': 'Kompatibilitas', 'umur_pakai': 'Umur Pakai',
        'total_bobot': 'Total Bobot', 'hasil': 'Hasil Skor',
    })


def hitung_skor(sistem_ctrl, performa_val, biaya_val, waktu_render_val,
                kompatibilitas_val, umur_pakai_val):
    """
    Fuzzifikasi -> Inferensi (rule base) -> Defuzzifikasi.
    Mengembalikan skor akhir (0-100).
    """
    sim = ctrl.ControlSystemSimulation(sistem_ctrl)
    sim.input['performa']       = float(performa_val)
    sim.input['biaya']          = float(biaya_val)
    sim.input['waktu_render']   = float(waktu_render_val)
    sim.input['kompatibilitas'] = float(kompatibilitas_val)
    sim.input['umur_pakai']     = float(umur_pakai_val)
    sim.compute()
    return sim.output['skor']