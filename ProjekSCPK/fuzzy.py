import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


BOBOT = {
    'performa':       {'rendah': 0, 'sedang': 1, 'tinggi': 2},
    'biaya':          {'murah': 2, 'sedang': 1, 'mahal': 0},
    'waktu_render':   {'cepat': 2, 'sedang': 1, 'lambat': 0},
    'kompatibilitas': {'rendah': 0, 'sedang': 1, 'tinggi': 2},
    'umur_pakai':     {'pendek': 0, 'sedang': 1, 'panjang': 2},
}

BATAS_BURUK_MAX = 3
BATAS_CUKUP_MAX = 6

# Cache rule definitions di level modul, supaya _generate_rule_definitions()
# tidak dihitung ulang setiap kali get_rule_base_table() dipanggil.
_RULE_DEFINITIONS_CACHE = None


def _generate_rule_definitions():
    """Generate semua kombinasi rule (3^5 = 243) sekali, dipakai 2 tempat:
    bangun_sistem_fuzzy() (untuk ctrl.Rule) dan get_rule_base_table().
    Hasilnya disimpan di cache modul karena isinya selalu sama (BOBOT tidak
    berubah saat runtime), jadi tidak perlu dihitung ulang berkali-kali."""
    global _RULE_DEFINITIONS_CACHE
    if _RULE_DEFINITIONS_CACHE is not None:
        return _RULE_DEFINITIONS_CACHE

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
    _RULE_DEFINITIONS_CACHE = rule_base
    return rule_base


def bangun_sistem_fuzzy():
    """
    Membangun seluruh komponen fuzzy Mamdani dalam satu fungsi.

    CATATAN OPTIMASI:
    Resolusi universe 'performa' dan 'biaya' diperkecil (step dinaikkan)
    dari step=1 menjadi step=10 dan step=5000. Range (batas bawah/atas)
    TIDAK berubah, dan semua breakpoint membership function (trapmf/trimf)
    tetap kelipatan dari step baru, sehingga bentuk kurva fuzzy-nya
    100% identik dengan sebelumnya — hanya jumlah titik di array yang
    berkurang drastis (10.001 -> 1.001 dan 5.000.001 -> 1.001).
    Ini membuat pembuatan membership function, perhitungan inferensi,
    dan terutama proses plotting di app.py jadi jauh lebih ringan,
    tanpa mengubah hasil skor sama sekali.
    """

    # 1. UNIVERSE (resolusi performa & biaya diperkecil, range tetap sama)
    performa       = ctrl.Antecedent(np.arange(0, 10001, 10),   'performa')
    biaya          = ctrl.Antecedent(np.arange(0, 5000001, 5000), 'biaya')
    waktu_render   = ctrl.Antecedent(np.arange(0, 61, 1),      'waktu_render')
    kompatibilitas = ctrl.Antecedent(np.arange(0, 101, 1),     'kompatibilitas')
    umur_pakai     = ctrl.Antecedent(np.arange(0, 11, 1),      'umur_pakai')
    skor           = ctrl.Consequent(np.arange(0, 101, 1),     'skor')

    # 2. MEMBERSHIP FUNCTION (parameter SAMA PERSIS seperti sebelumnya)
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

    # 3. RULE BASE -> ControlSystem (logic tidak berubah)
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
    """Fuzzifikasi -> Inferensi (rule base) -> Defuzzifikasi. Tidak diubah."""
    sim = ctrl.ControlSystemSimulation(sistem_ctrl)
    sim.input['performa']       = float(performa_val)
    sim.input['biaya']          = float(biaya_val)
    sim.input['waktu_render']   = float(waktu_render_val)
    sim.input['kompatibilitas'] = float(kompatibilitas_val)
    sim.input['umur_pakai']     = float(umur_pakai_val)
    sim.compute()
    return sim.output['skor']