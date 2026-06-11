import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =============================================================================
# FUZZY ENGINE - Sistem Fuzzy Mamdani untuk Pemilihan Komponen PC
# Developer mendefinisikan semua universe, himpunan fuzzy, dan rule base di sini
# =============================================================================

# -----------------------------------------------------------------------------
# 1. DEFINISI UNIVERSE (Semesta Pembicaraan)
# -----------------------------------------------------------------------------

def build_fuzzy_system():
    """
    Membangun dan mengembalikan objek ControlSystem fuzzy Mamdani
    beserta semua Antecedent dan Consequent-nya.
    """

    # --- Antecedent (Input) ---
    performa      = ctrl.Antecedent(np.arange(0, 10001, 1),    'performa')
    biaya         = ctrl.Antecedent(np.arange(0, 5000001, 1),  'biaya')
    waktu_render  = ctrl.Antecedent(np.arange(0, 61, 1),       'waktu_render')
    kompatibilitas= ctrl.Antecedent(np.arange(0, 101, 1),      'kompatibilitas')
    umur_pakai    = ctrl.Antecedent(np.arange(0, 11, 1),       'umur_pakai')

    # --- Consequent (Output) ---
    skor = ctrl.Consequent(np.arange(0, 101, 1), 'skor')

    # -------------------------------------------------------------------------
    # 2. DEFINISI FUNGSI KEANGGOTAAN (Membership Functions)
    #    Setiap variabel dibagi menjadi 3 himpunan fuzzy
    # -------------------------------------------------------------------------

    # --- Performa: rendah, sedang, tinggi ---
    performa['rendah']  = fuzz.trimf(performa.universe,  [0,    0,    5000])
    performa['sedang']  = fuzz.trimf(performa.universe,  [2000, 5000, 8000])
    performa['tinggi']  = fuzz.trimf(performa.universe,  [5000, 10000, 10000])

    # --- Biaya: murah, sedang, mahal ---
    # Catatan: biaya MURAH = lebih disukai, MAHAL = kurang disukai
    biaya['murah']  = fuzz.trimf(biaya.universe, [0,       0,       2000000])
    biaya['sedang'] = fuzz.trimf(biaya.universe, [500000,  2000000, 3500000])
    biaya['mahal']  = fuzz.trimf(biaya.universe, [2000000, 5000000, 5000000])

    # --- Waktu Render: cepat, sedang, lambat ---
    # Catatan: waktu CEPAT = lebih disukai
    waktu_render['cepat']   = fuzz.trimf(waktu_render.universe, [0,  0,  25])
    waktu_render['sedang']  = fuzz.trimf(waktu_render.universe, [15, 30, 45])
    waktu_render['lambat']  = fuzz.trimf(waktu_render.universe, [30, 60, 60])

    # --- Kompatibilitas: rendah, sedang, tinggi ---
    kompatibilitas['rendah']  = fuzz.trimf(kompatibilitas.universe, [0,   0,   50])
    kompatibilitas['sedang']  = fuzz.trimf(kompatibilitas.universe, [30,  55,  80])
    kompatibilitas['tinggi']  = fuzz.trimf(kompatibilitas.universe, [60, 100, 100])

    # --- Umur Pakai: pendek, sedang, panjang ---
    umur_pakai['pendek']  = fuzz.trimf(umur_pakai.universe, [0, 0, 4])
    umur_pakai['sedang']  = fuzz.trimf(umur_pakai.universe, [2, 5, 8])
    umur_pakai['panjang'] = fuzz.trimf(umur_pakai.universe, [5, 10, 10])

    # --- Skor Output: buruk, cukup, baik ---
    skor['buruk']  = fuzz.trimf(skor.universe, [0,   0,   50])
    skor['cukup']  = fuzz.trimf(skor.universe, [20,  50,  80])
    skor['baik']   = fuzz.trimf(skor.universe, [50, 100, 100])

    # -------------------------------------------------------------------------
    # 3. RULE BASE (Aturan Fuzzy IF-THEN)
    #    Developer mendefinisikan semua aturan di sini
    # -------------------------------------------------------------------------

    rules = [
    # ── SKOR BURUK ──────────────────────────────────────────────────────────
    # Performa rendah selalu jadi masalah utama
    ctrl.Rule(performa['rendah'] & biaya['mahal'],            skor['buruk']),
    ctrl.Rule(performa['rendah'] & waktu_render['lambat'],    skor['buruk']),
    ctrl.Rule(performa['rendah'] & kompatibilitas['rendah'],  skor['buruk']),
    # Biaya mahal tapi tidak sepadan
    ctrl.Rule(biaya['mahal'] & kompatibilitas['rendah'],      skor['buruk']),
    ctrl.Rule(biaya['mahal'] & umur_pakai['pendek'],          skor['buruk']),

    # ── SKOR CUKUP ──────────────────────────────────────────────────────────
    # Performa oke tapi ada kelemahan
    ctrl.Rule(performa['tinggi'] & biaya['mahal'],            skor['cukup']),
    ctrl.Rule(performa['sedang'] & biaya['sedang'],           skor['cukup']),
    ctrl.Rule(performa['sedang'] & waktu_render['sedang'],    skor['cukup']),
    ctrl.Rule(performa['sedang'] & kompatibilitas['sedang'],  skor['cukup']),
    # Umur pakai jadi penyelamat komponen biasa
    ctrl.Rule(performa['rendah'] & umur_pakai['panjang'],     skor['cukup']),

    # ── SKOR BAIK ───────────────────────────────────────────────────────────
    # Kombinasi ideal
    ctrl.Rule(performa['tinggi'] & biaya['murah'],                          skor['baik']),
    ctrl.Rule(performa['tinggi'] & waktu_render['cepat'],                   skor['baik']),
    ctrl.Rule(performa['tinggi'] & kompatibilitas['tinggi'],                skor['baik']),
    ctrl.Rule(performa['tinggi'] & umur_pakai['panjang'],                   skor['baik']),
    ctrl.Rule(kompatibilitas['tinggi'] & umur_pakai['panjang'] & biaya['murah'], skor['baik']),
]

    sistem_ctrl = ctrl.ControlSystem(rules)

    return sistem_ctrl, performa, biaya, waktu_render, kompatibilitas, umur_pakai, skor


def hitung_skor(performa_val, biaya_val, waktu_render_val,
                kompatibilitas_val, umur_pakai_val):
    """
    Menjalankan simulasi fuzzy Mamdani dan mengembalikan:
      - skor_output   : float, hasil defuzzifikasi (0–100)
      - detail        : dict berisi nilai membership tiap himpunan per variabel
    """
    sistem_ctrl, performa, biaya, waktu_render, kompatibilitas, umur_pakai, skor = \
        build_fuzzy_system()

    sim = ctrl.ControlSystemSimulation(sistem_ctrl)

    # Masukkan nilai input
    sim.input['performa']       = float(performa_val)
    sim.input['biaya']          = float(biaya_val)
    sim.input['waktu_render']   = float(waktu_render_val)
    sim.input['kompatibilitas'] = float(kompatibilitas_val)
    sim.input['umur_pakai']     = float(umur_pakai_val)

    sim.compute()
    skor_output = sim.output['skor']

    # -------------------------------------------------------------------------
    # Hitung derajat keanggotaan (fuzzifikasi) untuk setiap variabel
    # -------------------------------------------------------------------------
    detail = {
        'performa': {
            'rendah': float(fuzz.interp_membership(
                performa.universe, performa['rendah'].mf, performa_val)),
            'sedang': float(fuzz.interp_membership(
                performa.universe, performa['sedang'].mf, performa_val)),
            'tinggi': float(fuzz.interp_membership(
                performa.universe, performa['tinggi'].mf, performa_val)),
        },
        'biaya': {
            'murah': float(fuzz.interp_membership(
                biaya.universe, biaya['murah'].mf, biaya_val)),
            'sedang': float(fuzz.interp_membership(
                biaya.universe, biaya['sedang'].mf, biaya_val)),
            'mahal': float(fuzz.interp_membership(
                biaya.universe, biaya['mahal'].mf, biaya_val)),
        },
        'waktu_render': {
            'cepat': float(fuzz.interp_membership(
                waktu_render.universe, waktu_render['cepat'].mf, waktu_render_val)),
            'sedang': float(fuzz.interp_membership(
                waktu_render.universe, waktu_render['sedang'].mf, waktu_render_val)),
            'lambat': float(fuzz.interp_membership(
                waktu_render.universe, waktu_render['lambat'].mf, waktu_render_val)),
        },
        'kompatibilitas': {
            'rendah': float(fuzz.interp_membership(
                kompatibilitas.universe, kompatibilitas['rendah'].mf, kompatibilitas_val)),
            'sedang': float(fuzz.interp_membership(
                kompatibilitas.universe, kompatibilitas['sedang'].mf, kompatibilitas_val)),
            'tinggi': float(fuzz.interp_membership(
                kompatibilitas.universe, kompatibilitas['tinggi'].mf, kompatibilitas_val)),
        },
        'umur_pakai': {
            'pendek': float(fuzz.interp_membership(
                umur_pakai.universe, umur_pakai['pendek'].mf, umur_pakai_val)),
            'sedang': float(fuzz.interp_membership(
                umur_pakai.universe, umur_pakai['sedang'].mf, umur_pakai_val)),
            'panjang': float(fuzz.interp_membership(
                umur_pakai.universe, umur_pakai['panjang'].mf, umur_pakai_val)),
        },
    }

    return skor_output, detail, (performa, biaya, waktu_render, kompatibilitas, umur_pakai, skor)


def get_membership_data():
    """
    Mengembalikan data universe dan MF untuk keperluan plotting di Streamlit.
    Dipanggil sekali saat tab grafik dibuka.
    """
    _, performa, biaya, waktu_render, kompatibilitas, umur_pakai, skor = \
        build_fuzzy_system()

    return {
        'performa': {
            'universe': performa.universe,
            'mf': {
                'rendah': performa['rendah'].mf,
                'sedang': performa['sedang'].mf,
                'tinggi': performa['tinggi'].mf,
            },
            'label': 'Performa',
            'satuan': 'benchmark score'
        },
        'biaya': {
            'universe': biaya.universe,
            'mf': {
                'murah':  biaya['murah'].mf,
                'sedang': biaya['sedang'].mf,
                'mahal':  biaya['mahal'].mf,
            },
            'label': 'Biaya',
            'satuan': 'Rp'
        },
        'waktu_render': {
            'universe': waktu_render.universe,
            'mf': {
                'cepat':  waktu_render['cepat'].mf,
                'sedang': waktu_render['sedang'].mf,
                'lambat': waktu_render['lambat'].mf,
            },
            'label': 'Waktu Render',
            'satuan': 'detik'
        },
        'kompatibilitas': {
            'universe': kompatibilitas.universe,
            'mf': {
                'rendah': kompatibilitas['rendah'].mf,
                'sedang': kompatibilitas['sedang'].mf,
                'tinggi': kompatibilitas['tinggi'].mf,
            },
            'label': 'Kompatibilitas',
            'satuan': '%'
        },
        'umur_pakai': {
            'universe': umur_pakai.universe,
            'mf': {
                'pendek':  umur_pakai['pendek'].mf,
                'sedang':  umur_pakai['sedang'].mf,
                'panjang': umur_pakai['panjang'].mf,
            },
            'label': 'Umur Pakai',
            'satuan': 'tahun'
        },
        'skor': {
            'universe': skor.universe,
            'mf': {
                'buruk': skor['buruk'].mf,
                'cukup': skor['cukup'].mf,
                'baik':  skor['baik'].mf,
            },
            'label': 'Skor Rekomendasi',
            'satuan': 'poin'
        },
    }