"""Generator data SINTETIS yang realistis untuk industri tambang.

Menghasilkan:
  data/incidents.csv          — laporan insiden berlabel (kategori hazard + tingkat risiko)
  data/shift_reports.jsonl    — teks laporan shift + label JSON (gold) untuk ekstraksi
  data/work_orders.csv        — histori work order maintenance
  data/fault_codes.json       — tabel kode gangguan (konsisten dengan manual di data/corpus)
  data/eval/rag_qa.jsonl      — set evaluasi RAG (pertanyaan, dokumen acuan, kata kunci jawaban)

Semua data deterministik (seed) agar eksperimen reproducible.
Ganti file-file ini dengan data asli perusahaan dengan skema kolom yang sama.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from minesense.config import get_settings

# ----------------------------------------------------------------------------
# Kosakata umum
# ----------------------------------------------------------------------------
PITS = ["Pit Barat", "Pit Timur", "Pit Selatan", "Pit Utara", "Pit Kenari", "Pit Meranti"]
LOCS = ["haul road KM 3", "haul road KM 7", "area loading", "disposal Selatan", "simpang 4 ROM",
        "ramp B2", "jenjang 45", "workshop utama", "fuel station 2", "crusher", "dumping point D5",
        "kolam pengendapan KP-3", "jalan tambang segmen 12"]
TIMES = ["pukul 02.15", "pukul 03.40", "pukul 07.20", "pukul 10.05", "pukul 13.30", "pukul 15.45",
         "pukul 19.10", "pukul 22.50", "menjelang pergantian shift", "saat jam istirahat malam"]
WEATHER = ["", "", "", "kondisi hujan ringan,", "jalan basah setelah hujan,", "jarak pandang terbatas karena debu,",
           "kabut tebal,", "cuaca cerah,"]


def _unit(prefix: str, rng: random.Random) -> str:
    return f"{prefix}-{rng.randint(1001, 3099)}"


# Template per kategori. {u}=unit, {loc}=lokasi, {t}=waktu, {w}=cuaca
EVENTS: dict[str, list[str]] = {
    "kendaraan_alat_berat": [
        "Dump truck {u} menabrak bagian belakang dump truck lain di {loc} {t}, {w} jarak beriring diduga kurang dari 50 meter.",
        "Unit {u} tergelincir dan keluar jalur ke arah tanggul di {loc} {t}, {w} kecepatan diperkirakan melebihi batas.",
        "Light vehicle bersenggolan dengan {u} di {loc} {t} karena LV masuk tanpa izin radio.",
        "Excavator {u} menyenggol kabin dump truck saat proses loading di {loc} {t}.",
        "Unit {u} mundur dan menabrak LV yang parkir di blind spot di {loc} {t}, alarm mundur tidak berbunyi.",
        "Dump truck {u} terguling di tepi dumping {loc} {t}, tinggi tanggul tidak memenuhi standar.",
        "Dozer {u} hampir menabrak pekerja yang berjalan di area {loc} {t}.",
        "Rem dump truck {u} blong saat menuruni ramp di {loc} {t}, operator mengarahkan unit ke tanggul penahan.",
    ],
    "longsor_geoteknik": [
        "Terjadi longsoran material dari lereng di {loc} {t}, {w} material menutup sebagian jalan.",
        "Ditemukan retakan tarik baru sepanjang kurang lebih 15 meter di crest lereng {loc} {t}.",
        "Batu jatuh (rockfall) dari muka jenjang mengenai sisi unit {u} di {loc} {t}.",
        "Radar pemantau lereng menunjukkan pergerakan 12 mm per hari di {loc}, status naik ke siaga.",
        "Amblesan permukaan di area dumping {loc} {t} saat unit {u} akan dumping.",
        "Rembesan air baru dan penggembungan pada toe lereng teramati di {loc} {t}.",
    ],
    "peledakan": [
        "Ditemukan 2 lubang misfire setelah peledakan di {loc} {t}.",
        "Flyrock terlempar keluar dari radius aman dan jatuh dekat {loc} {t}.",
        "Seorang pekerja masih berada di dalam radius 500 meter saat sirine peledakan dibunyikan di {loc}.",
        "Getaran peledakan dikeluhkan warga, terukur melebihi batas internal di {loc}.",
        "Aktivitas charging lubang ledak tetap berjalan saat petir terdeteksi dalam radius 8 km di {loc} {t}.",
        "Mata bor unit drill {u} mengenai sisa bahan peledak dari lubang lama di {loc} {t}.",
    ],
    "listrik": [
        "Mekanik tersengat listrik saat memperbaiki panel pompa dewatering di {loc} {t}, LOTO tidak diterapkan.",
        "Terjadi percikan api (arc flash) di panel MCC {loc} {t} saat breaker dioperasikan.",
        "Kabel trailing pompa di {loc} terkelupas dan terlindas unit {u} {t}.",
        "Battery isolator unit {u} tidak dimatikan saat penggantian alternator di {loc}.",
        "Tiang lampu penerangan tower roboh dan kabel bertegangan menyentuh tanah di {loc} {t}.",
    ],
    "bekerja_di_ketinggian": [
        "Mekanik terpeleset dari tangga excavator {u} setinggi 3 meter di {loc} {t}.",
        "Pekerja naik ke atas vessel dump truck {u} tanpa full body harness di {loc} {t}.",
        "Kunci pas terjatuh dari deck excavator {u} dan hampir mengenai helper di bawahnya di {loc}.",
        "Pekerja terjatuh dari scaffolding saat pemasangan conveyor di {loc} {t}.",
        "Operator turun dari kabin {u} tanpa tiga titik kontak dan terjatuh di {loc} {t}.",
    ],
    "lingkungan": [
        "Selang hidrolik unit {u} pecah dan oli tumpah sekitar {vol} liter di {loc} {t}.",
        "Tumpahan solar sekitar {vol} liter dari fuel truck di {loc} {t}.",
        "pH outlet kolam pengendapan di {loc} terukur 4,8 di bawah target internal.",
        "TSS outlet {loc} terukur 340 mg/L setelah hujan deras {t}.",
        "Coolant bocor dari radiator unit {u} sekitar {vol} liter dan mengalir ke saluran drainase di {loc}.",
        "Drum limbah B3 ditemukan tanpa label di {loc}.",
    ],
    "kesehatan_kerja": [
        "Alarm microsleep operator dump truck {u} berbunyi dua kali dalam satu shift di {loc} {t}.",
        "Operator {u} mengalami pusing dan tekanan darah 170/105 saat fit to work check.",
        "Hasil pengukuran debu di {loc} melebihi nilai ambang batas internal.",
        "Operator crusher mengeluh gangguan pendengaran, kebisingan terukur 95 dB di {loc}.",
        "Pekerja mengalami dehidrasi dan pingsan di {loc} {t} saat cuaca panas.",
        "Operator {u} mengaku hanya tidur 4 jam sebelum shift malam dan tetap mengoperasikan unit.",
    ],
}

# Konsekuensi -> tingkat risiko (selaras matriks SOP-K3-008)
OUTCOMES: dict[str, list[str]] = {
    "rendah": [
        "Tidak ada korban maupun kerusakan.", "Nyaris celaka, tidak ada cedera.",
        "Kerusakan ringan pada spion, estimasi biaya Rp2 juta.", "Korban hanya memerlukan P3K di lokasi.",
        "Tidak ada cedera, aktivitas dilanjutkan setelah area diamankan.",
    ],
    "sedang": [
        "Operator mendapat perawatan medis di klinik dan kembali bekerja shift berikutnya.",
        "Kerusakan unit diperkirakan Rp45 juta.", "Satu pekerja mengalami luka robek dan dijahit di klinik.",
        "Unit downtime 2 shift untuk perbaikan.", "Area ditutup sementara selama 6 jam.",
    ],
    "tinggi": [
        "Korban mengalami patah tulang dan tidak dapat bekerja (LTI).",
        "Kerusakan unit diperkirakan Rp350 juta.", "Kejadian berpotensi fatal (high potential incident).",
        "Dua pekerja dirujuk ke rumah sakit dan tidak masuk kerja 5 hari.",
        "Operasi pit dihentikan satu shift penuh.",
    ],
    "kritis": [
        "Satu pekerja meninggal dunia.", "Korban mengalami cedera berat dan cacat permanen.",
        "Longsoran menimbun dua unit dan operasi pit dihentikan.",
        "Korban kritis dievakuasi ke rumah sakit rujukan.", "Material menimpa kabin dan operator terjepit.",
    ],
}
# Kategori tertentu condong ke risiko tertentu (lebih realistis)
RISK_PRIOR = {
    "kendaraan_alat_berat": [0.35, 0.30, 0.25, 0.10],
    "longsor_geoteknik": [0.20, 0.30, 0.30, 0.20],
    "peledakan": [0.10, 0.25, 0.45, 0.20],
    "listrik": [0.20, 0.30, 0.30, 0.20],
    "bekerja_di_ketinggian": [0.30, 0.30, 0.30, 0.10],
    "lingkungan": [0.40, 0.40, 0.15, 0.05],
    "kesehatan_kerja": [0.45, 0.35, 0.15, 0.05],
}
FOLLOWUPS = ["Area telah dibarikade.", "Pengawas sudah diinformasikan.", "Investigasi sedang berjalan.",
             "Unit dihentikan sementara.", "Laporan dibuat oleh pengawas shift.", ""]


def gen_incidents(n: int, rng: random.Random) -> list[dict]:
    from minesense.schemas import RISK_LEVELS

    rows = []
    cats = list(EVENTS)
    for i in range(n):
        cat = rng.choice(cats)
        risk = rng.choices(RISK_LEVELS, weights=RISK_PRIOR[cat])[0]
        t_idx = rng.randrange(len(EVENTS[cat]))
        tpl = EVENTS[cat][t_idx]
        unit_prefix = rng.choice(["DT", "DT", "EX", "BD", "LV", "GR"])
        text = tpl.format(u=_unit(unit_prefix, rng), loc=f"{rng.choice(LOCS)} {rng.choice(PITS)}",
                          t=rng.choice(TIMES), w=rng.choice(WEATHER), vol=rng.choice([5, 15, 40, 120, 300]))
        text = " ".join(text.split())
        text = f"{text} {rng.choice(OUTCOMES[risk])} {rng.choice(FOLLOWUPS)}".strip()
        # noise label 3% — meniru ketidakkonsistenan pelabelan manusia
        if rng.random() < 0.03:
            risk = rng.choice(RISK_LEVELS)
        rows.append({"incident_id": f"INC-{2026_0000 + i}", "description": text, "category": cat, "risk_level": risk,
                     "template_id": f"{cat}:{t_idx}"})
    return rows


# ----------------------------------------------------------------------------
# Laporan shift
# ----------------------------------------------------------------------------
def _fmt_id(x: float, dec: int = 0) -> str:
    """Format angka gaya Indonesia: 25.000 / 5,2"""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def gen_shift_reports(n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        day = rng.randint(1, 28)
        month = rng.randint(1, 12)
        tanggal = f"2026-{month:02d}-{day:02d}"
        shift = rng.choice(["pagi", "malam"])
        pit = rng.choice(PITS)
        ob = float(rng.randrange(8000, 60000, 50))
        coal = float(rng.randrange(1500, 15000, 10))
        sr = round(ob / coal, 2)
        rain = rng.choice([0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.25, 4.0])
        units = sorted({_unit(rng.choice(["DT", "EX", "BD", "GR"]), rng) for _ in range(rng.randint(0, 3))})
        inc = rng.choice([0, 0, 0, 1, 1, 2])
        catatan = rng.choice([
            "Jalan KM 5 bergelombang, perlu grading.", "Antrian di loading point EX karena kekurangan DT.",
            "Produksi di bawah plan karena hujan.", "Fuel truck terlambat 30 menit.", "Tidak ada catatan khusus.",
        ])
        gold = {"tanggal": tanggal, "shift": shift, "pit": pit, "ob_removal_bcm": ob, "coal_getting_ton": coal,
                "stripping_ratio": sr, "hujan_jam": rain, "unit_breakdown": units, "jumlah_insiden": inc,
                "catatan": catatan}
        d_id = f"{day:02d}/{month:02d}/2026"
        style = rng.randint(0, 2)
        bd = ", ".join(units) if units else "nihil"
        if style == 0:
            text = (f"LAPORAN PRODUKSI SHIFT\nTanggal: {d_id}\nShift: {shift.title()}\nLokasi: {pit}\n"
                    f"OB Removal: {_fmt_id(ob)} BCM\nCoal Getting: {_fmt_id(coal)} ton\n"
                    f"Stripping Ratio: {_fmt_id(sr, 2)}\nRain & Slippery: {_fmt_id(rain, 2)} jam\n"
                    f"Unit Breakdown: {bd}\nInsiden/Near miss: {inc}\nCatatan: {catatan}")
        elif style == 1:
            text = (f"Selamat pagi pak, izin lapor shift {shift} tgl {d_id} {pit}. "
                    f"OB tercapai {_fmt_id(ob)} bcm, coal {_fmt_id(coal)} ton (SR {_fmt_id(sr, 2)}). "
                    f"Hujan + slippery {_fmt_id(rain, 2)} jam. Unit BD: {bd}. "
                    f"Insiden {inc} kejadian. {catatan} Terima kasih.")
        else:
            text = (f"[{pit}] Shift {shift.upper()} — {d_id}\n| Item | Aktual |\n|---|---|\n"
                    f"| Overburden (BCM) | {_fmt_id(ob)} |\n| Batubara (ton) | {_fmt_id(coal)} |\n"
                    f"| SR | {_fmt_id(sr, 2)} |\n| Jam hujan/licin | {_fmt_id(rain, 2)} |\n"
                    f"| Breakdown | {bd} |\n| Jumlah insiden | {inc} |\nKeterangan: {catatan}")
        out.append({"report_id": f"SR-{i:05d}", "text": text, "gold": gold, "style": style})
    return out


# ----------------------------------------------------------------------------
# Kode gangguan & work order
# ----------------------------------------------------------------------------
@dataclass
class Fault:
    code: str
    model: str
    title: str
    causes: list[str]
    actions: list[str]
    symptoms: list[str]
    critical: bool = False


FAULTS = [
    Fault("E-ENG-101", "DT-100", "Temperatur coolant tinggi (> 105 °C)",
          ["Radiator tersumbat debu/lumpur", "Coolant kurang/bocor", "Fan belt slip/putus", "Thermostat macet", "Water pump rusak"],
          ["Idle-kan mesin 5 menit sebelum dimatikan", "Cek level coolant setelah suhu < 50 °C", "Bersihkan radiator", "Cek fan belt", "Uji thermostat"],
          ["mesin overheat", "suhu mesin naik terus saat tanjakan", "indikator temperatur merah", "coolant cepat habis"]),
    Fault("E-ENG-115", "DT-100", "Tekanan oli mesin rendah",
          ["Level oli rendah", "Filter oli tersumbat", "Pengenceran oli oleh bahan bakar", "Pompa oli aus", "Sensor tekanan rusak"],
          ["Matikan mesin segera", "Cek level oli", "Ambil sampel oli (fuel dilution)", "Verifikasi dengan manometer mekanik"],
          ["lampu tekanan oli menyala", "tekanan oli drop saat idle", "bunyi ketukan dari mesin"], True),
    Fault("E-TRM-201", "DT-100", "Temperatur oli transmisi tinggi (> 120 °C)",
          ["Level oli transmisi kurang", "Oil cooler tersumbat", "Kopling slip", "Pola operasi rpm tinggi gigi rendah"],
          ["Kurangi beban", "Cek level oli transmisi", "Bersihkan cooler", "Tes stall"],
          ["transmisi panas", "perpindahan gigi kasar dan suhu transmisi tinggi", "unit kehilangan tenaga di tanjakan"]),
    Fault("E-BRK-301", "DT-100", "Tekanan sistem rem rendah",
          ["Kebocoran selang rem", "Nitrogen accumulator rendah", "Pompa rem aus"],
          ["Unit dilarang beroperasi, pasang tag DO NOT OPERATE", "Cek kebocoran", "Ukur tekanan nitrogen accumulator", "Tes rem statis"],
          ["rem terasa dalam", "alarm tekanan rem", "pengereman tidak pakem saat turunan"], True),
    Fault("E-SUS-401", "DT-100", "Tekanan silinder suspensi tidak seimbang",
          ["Kebocoran gas nitrogen suspensi", "Seal silinder suspensi rusak"],
          ["Ukur ride height", "Isi ulang nitrogen", "Ganti seal bila bocor"],
          ["unit miring ke satu sisi", "guncangan berlebih", "suspensi bocor oli"]),
    Fault("E-ELC-501", "DT-100", "Tegangan sistem rendah (< 24 V)",
          ["Alternator rusak", "Belt alternator slip", "Terminal baterai korosi", "Kabel ground rusak"],
          ["Ukur tegangan pengisian (27–28,5 V)", "Bersihkan terminal", "Cek output alternator"],
          ["unit susah starter", "lampu redup", "monitor panel mati hidup"]),
    Fault("E-HYD-601", "DT-100", "Vessel (hoist) lambat naik",
          ["Level oli hidrolik rendah", "Pompa hoist aus", "Setting relief valve turun", "Filter suction tersumbat"],
          ["Cek level oli hidrolik", "Ukur tekanan relief hoist", "Cek waktu angkat (± 20 detik)"],
          ["vessel lambat naik saat dumping", "hoist tidak kuat angkat penuh"]),
    Fault("X-HYD-110", "EX-2000", "Temperatur oli hidrolik tinggi (> 95 °C)",
          ["Oil cooler tersumbat", "Level oli hidrolik rendah", "Relief valve bocor internal", "Fan cooler tidak optimal"],
          ["Kurangi beban kerja", "Cek level oli", "Bersihkan oil cooler", "Ukur putaran fan", "Thermo-gun pada relief valve"],
          ["oli hidrolik panas", "gerakan melambat setelah 3 jam operasi", "alarm hydraulic temperature"]),
    Fault("X-HYD-120", "EX-2000", "Tekanan pilot rendah (< 30 bar)",
          ["Filter pilot tersumbat", "Pompa pilot aus", "Relief pilot valve rusak"],
          ["Ganti filter pilot", "Ukur tekanan pilot di test port", "Setel relief pilot"],
          ["semua gerakan lambat", "joystick tidak responsif", "boom tidak mau naik"]),
    Fault("X-SWG-210", "EX-2000", "Swing lambat atau berbunyi abnormal",
          ["Oli swing machinery kurang/terkontaminasi", "Gear swing aus", "Swing brake tidak release", "Swing motor bocor internal"],
          ["Cek oli swing & magnetic plug", "Uji drain rate swing motor", "Cek swing brake release pressure"],
          ["swing lambat", "bunyi gemeretak saat swing", "swing tersendat"]),
    Fault("X-ENG-310", "EX-2000", "Mesin overheat",
          ["Radiator tersumbat", "Coolant kurang", "Fan belt", "Thermostat"],
          ["Idle-kan mesin sebelum dimatikan", "Bersihkan radiator", "Cek coolant"],
          ["mesin excavator overheat", "suhu engine tinggi"]),
    Fault("X-UND-410", "EX-2000", "Track kendur atau lepas",
          ["Tekanan grease track adjuster turun", "Seal adjuster bocor", "Idler/roller aus"],
          ["Ukur sag track (30–50 mm)", "Tambah grease adjuster", "Cek seal adjuster"],
          ["track kendur", "track lepas saat travel", "rantai track melorot"]),
    Fault("X-ELC-510", "EX-2000", "Error komunikasi kontroler (CAN bus)",
          ["Konektor longgar/korosi", "Harness terkelupas", "Resistor terminasi rusak"],
          ["Ukur resistansi CAN (± 60 ohm)", "Periksa konektor boom foot"],
          ["monitor error komunikasi", "fungsi kontrol hilang sesaat", "kode CAN muncul hilang"]),
    Fault("D-ENG-101", "BD-375", "Mesin overheat",
          ["Radiator tersumbat material", "Fan reversible tidak bekerja", "Coolant kurang"],
          ["Aktifkan fan reverse", "Cek coolant"], ["dozer overheat", "suhu mesin dozer tinggi"]),
    Fault("D-PT-201", "BD-375", "Temperatur oli power train tinggi (> 120 °C)",
          ["Torque converter stall terlalu lama", "Oil cooler tersumbat"],
          ["Kurangi beban dozing", "Hindari stall > 10 detik", "Bersihkan cooler"],
          ["power train panas", "dozer kehilangan tenaga saat dozing berat"]),
    Fault("D-UND-301", "BD-375", "Track shoe bolt longgar / shoe patah",
          ["Torque baut tidak sesuai", "Operasi di batuan keras dengan kecepatan tinggi"],
          ["Torque ulang baut", "Inspeksi retakan shoe"], ["track shoe lepas", "baut shoe longgar", "bunyi ketukan dari track"]),
]


def gen_work_orders(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(n):
        f = rng.choice(FAULTS)
        cause = rng.choice(f.causes)
        action = "; ".join(rng.sample(f.actions, k=min(len(f.actions), rng.randint(1, 3))))
        sym = rng.choice(f.symptoms)
        extra = rng.choice(["", " saat shift malam", " setelah hujan", " di tanjakan ramp", " setelah PM 250"])
        rows.append({
            "wo_id": f"WO-{26000 + i}", "unit": f"{f.model.split('-')[0]}-{rng.randint(1001, 3099)}", "model": f.model,
            "fault_code": f.code if rng.random() > 0.15 else "",  # kadang operator tidak mencatat kode
            "symptom": f"{sym}{extra}", "root_cause": cause, "action": action,
            "downtime_hours": round(rng.uniform(0.5, 36), 1), "hour_meter": rng.randint(1200, 48000),
        })
    return rows


# ----------------------------------------------------------------------------
# Set evaluasi RAG (ditulis manual, mengacu fakta di data/corpus)
# ----------------------------------------------------------------------------
RAG_QA = [
    ("Berapa batas kecepatan dump truck di jalan angkut saat basah atau licin?", "SOP-K3-001", ["25 km/jam"]),
    ("Berapa jarak beriring minimum antar dump truck saat jalan basah?", "SOP-K3-001", ["100 meter"]),
    ("Berapa tinggi minimum tanggul pengaman di tepi dumping?", "SOP-K3-001", ["diameter roda"]),
    ("Apakah operator dump truck boleh turun dari kabin saat proses loading?", "SOP-K3-001", ["dilarang turun"]),
    ("Apa langkah verifikasi energi nol pada prosedur LOTO?", "SOP-K3-002", ["try-out", "zero energy"]),
    ("Siapa yang boleh melepas gembok LOTO jika pemiliknya tidak ada di lokasi?", "SOP-K3-002", ["KTT"]),
    ("Apa yang harus dipasang sebelum bekerja di bawah vessel dump truck?", "SOP-K3-002", ["body prop"]),
    ("Pada ketinggian berapa pekerjaan dianggap bekerja di ketinggian?", "SOP-K3-003", ["1,8 meter"]),
    ("Berapa kekuatan minimum anchor point untuk harness?", "SOP-K3-003", ["22 kN"]),
    ("Berapa lama maksimal waktu evakuasi pekerja yang tergantung harness?", "SOP-K3-003", ["15 menit"]),
    ("Berapa radius aman peledakan untuk manusia?", "SOP-K3-004", ["500 meter"]),
    ("Berapa lama harus menunggu sebelum mendekati lokasi misfire?", "SOP-K3-004", ["30 menit"]),
    ("Bolehkah mengebor dekat lubang misfire?", "SOP-K3-004", ["1 meter"]),
    ("Apa tindakan jika kecepatan pergerakan lereng 10 sampai 50 mm per hari?", "SOP-K3-005", ["evakuasi", "Siaga"]),
    ("Berapa faktor keamanan minimum lereng keseluruhan?", "SOP-K3-005", ["1,3"]),
    ("Kapan inspeksi geoteknik tambahan harus dilakukan setelah hujan?", "SOP-K3-005", ["50 mm"]),
    ("Berapa jam tidur minimum sebelum shift untuk operator?", "SOP-K3-006", ["6 jam"]),
    ("Apa tindakan jika alarm microsleep berbunyi dua kali dalam satu shift?", "SOP-K3-006", ["dibebastugaskan"]),
    ("Berapa batas tekanan darah operator agar boleh bekerja?", "SOP-K3-006", ["160"]),
    ("Berapa target waktu respons ERT di dalam pit?", "SOP-K3-007", ["10 menit"]),
    ("Kapan aktivitas boleh dilanjutkan setelah petir?", "SOP-K3-007", ["30 menit"]),
    ("Seberapa jauh operator harus menjauh dari unit yang terbakar?", "SOP-K3-007", ["30 meter"]),
    ("Bagaimana penanganan tumpahan oli lebih dari 200 liter?", "SOP-ENV-001", ["KTT", "Besar"]),
    ("Berapa lama limbah B3 tumpahan harus dikirim ke TPS?", "SOP-ENV-001", ["24 jam"]),
    ("Apa yang dilakukan jika pH outlet kolam pengendapan di bawah 6?", "SOP-ENV-002", ["kapur"]),
    ("Berapa target internal TSS di outlet kolam pengendapan?", "SOP-ENV-002", ["200 mg/L"]),
    ("Apa saja item kritis P2H yang membuat unit tidak boleh dioperasikan?", "SOP-OPS-001", ["rem", "steering"]),
    ("Berapa batas waktu laporan awal untuk insiden tingkat tinggi?", "SOP-K3-008", ["4 jam"]),
    ("Metode investigasi apa yang digunakan untuk insiden kritis?", "SOP-K3-008", ["ICAM"]),
    ("Apa penyebab kode E-ENG-101 pada dump truck?", "MAN-DT100", ["radiator", "thermostat"]),
    ("Apa yang harus dilakukan saat muncul kode E-BRK-301?", "MAN-DT100", ["dilarang beroperasi"]),
    ("Berapa interval penggantian oli hidrolik dump truck DT-100?", "MAN-DT100", ["2000 jam"]),
    ("Berapa tekanan pilot normal pada excavator EX-2000?", "MAN-EX2000", ["30–35 bar"]),
    ("Kenapa swing excavator lambat dan berbunyi?", "MAN-EX2000", ["swing machinery"]),
    ("Berapa resistansi normal CAN bus pada excavator?", "MAN-EX2000", ["60 ohm"]),
    ("Apa penyebab temperatur oli power train dozer tinggi?", "MAN-BD375", ["stall"]),
    ("Apa itu Kepmen ESDM 1827 K/30/MEM/2018?", "REG-001", ["Kaidah Teknik Pertambangan"]),
    ("Siapa yang bertanggung jawab atas keselamatan operasi di wilayah tambang?", "REG-001", ["Kepala Teknik Tambang"]),
    ("Bagaimana cara menghitung stripping ratio?", "OPS-002", ["BCM", "ton"]),
    ("Apa rumus Physical Availability?", "OPS-002", ["jam breakdown"]),
    ("Apa perbedaan BCM dan LCM?", "GLOS-001", ["swell"]),
]


def write_all(seed: int = 42, n_incidents: int = 1400, n_reports: int = 300, n_wo: int = 800) -> dict[str, int]:
    s = get_settings()
    d: Path = s.data_dir
    (d / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    inc = gen_incidents(n_incidents, rng)
    with open(d / "incidents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(inc[0]))
        w.writeheader()
        w.writerows(inc)

    reps = gen_shift_reports(n_reports, rng)
    with open(d / "shift_reports.jsonl", "w", encoding="utf-8") as f:
        for r in reps:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    wos = gen_work_orders(n_wo, rng)
    with open(d / "work_orders.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(wos[0]))
        w.writeheader()
        w.writerows(wos)

    with open(d / "fault_codes.json", "w", encoding="utf-8") as f:
        json.dump({x.code: x.__dict__ for x in FAULTS}, f, ensure_ascii=False, indent=2)

    with open(d / "eval" / "rag_qa.jsonl", "w", encoding="utf-8") as f:
        for q, doc, kws in RAG_QA:
            f.write(json.dumps({"question": q, "doc_id": doc, "keywords": kws}, ensure_ascii=False) + "\n")

    return {"incidents": len(inc), "shift_reports": len(reps), "work_orders": len(wos),
            "fault_codes": len(FAULTS), "rag_qa": len(RAG_QA)}


if __name__ == "__main__":
    print(write_all())
