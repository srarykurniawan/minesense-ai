"""Generator set evaluasi RAG v2 (ditulis manual) → data/eval/rag_qa_v2.jsonl

Prinsip (agar angka TIDAK optimistis seperti v1):
  - paraphrase : makna sama, kosakata BERBEDA dari dokumen (sinonim, susunan kalimat lain)
  - colloquial : gaya bahasa lapangan/WhatsApp, singkatan, salah ketik ringan
  - scenario   : situasi konkret yang harus dipetakan ke aturan (butuh penalaran ringan)
  - unanswerable: masih domain tambang tetapi jawabannya TIDAK ada di dokumen → sistem harus menolak
`keywords` = fakta yang WAJIB muncul di jawaban/konteks (semua harus ada). Ditulis tanpa melihat skor sistem.
Jalankan: python data/eval/build_rag_qa_v2.py
"""
import json
from pathlib import Path

P, C, S, U = "paraphrase", "colloquial", "scenario", "unanswerable"

QA = [
    # ---- SOP-K3-001 Dump truck ----
    (P, "Saat permukaan jalan angkut licin sehabis hujan, laju maksimal truk jungkit dibatasi berapa?", "SOP-K3-001", ["25 km"]),
    (C, "bro kalo jalan hauling lagi kering & lurus DT boleh lari brp km/jam?", "SOP-K3-001", ["40 km"]),
    (P, "Ketika membawa muatan di tanjakan curam di atas 8 persen, berapa kecepatan tertinggi yang diizinkan?", "SOP-K3-001", ["20 km"]),
    (P, "Seberapa jauh truk harus berjarak dengan truk di depannya kalau debu tebal membuat pandangan kurang dari 100 meter?", "SOP-K3-001", ["100 meter"]),
    (C, "jarak antar DT normalnya brp meter sih?", "SOP-K3-001", ["50 meter"]),
    (S, "Operator truk ingin mengambil HP yang jatuh ke lantai luar kabin sementara excavator sedang mengisi muatan. Apakah boleh?", "SOP-K3-001", ["dilarang turun"]),
    (P, "Tanggul pengaman di bibir area buangan harus setinggi apa?", "SOP-K3-001", ["diameter roda"]),
    (S, "Pengawas melihat ada rekahan di permukaan area dumping. Apa yang harus dilakukan operator truk?", "SOP-K3-001", ["crack"]),
    (P, "Di posisi mana truk harus menunggu giliran sebelum masuk titik muat dan berapa jaraknya dari alat gali?", "SOP-K3-001", ["30 m"]),
    (C, "LV mau masuk loading point, prosedurnya apa?", "SOP-K3-001", ["radio"]),
    (P, "Siapa yang berwenang menetapkan kondisi jalan licin dan menghentikan pengangkutan?", "SOP-K3-001", ["pengawas"]),
    (P, "Muatan berlebih berapa persen yang wajib dilaporkan ke atasan?", "SOP-K3-001", ["10%"]),
    # ---- SOP-K3-002 LOTO ----
    (P, "Apa prinsip dasar penguncian personal dalam prosedur isolasi energi?", "SOP-K3-002", ["satu gembok"]),
    (S, "Mekanik A lupa melepas gemboknya lalu pulang. Unit harus segera dipakai. Bagaimana gembok itu boleh dibuka?", "SOP-K3-002", ["KTT"]),
    (P, "Sebelum bekerja di kolong bak truk yang terangkat, alat penopang apa yang wajib dipasang?", "SOP-K3-002", ["body prop"]),
    (P, "Bagaimana membuang sisa tekanan pada sistem hidrolik setelah mesin dimatikan?", "SOP-K3-002", ["3–5 kali"]),
    (C, "warna gembok pribadi buat LOTO apa ya?", "SOP-K3-002", ["merah"]),
    (P, "Tahap mana dalam penguncian energi yang sama sekali tidak boleh dilewatkan?", "SOP-K3-002", ["Verifikasi"]),
    (P, "Pada suhu berapa oli/radiator dianggap aman untuk mulai dikerjakan?", "SOP-K3-002", ["50 °C"]),
    (P, "Apa konsekuensi bagi pekerja yang melanggar prosedur penguncian energi?", "SOP-K3-002", ["Life Saving Rules"]),
    # ---- SOP-K3-003 Ketinggian ----
    (P, "Mulai elevasi berapa sebuah pekerjaan digolongkan kerja di tempat tinggi?", "SOP-K3-003", ["1,8"]),
    (C, "titik cantolan harness minimal kuat nahan brp?", "SOP-K3-003", ["22 kN"]),
    (P, "Dalam berapa menit korban yang tergantung di sabuk pengaman harus diturunkan?", "SOP-K3-003", ["15 menit"]),
    (S, "Angin bertiup 45 km/jam saat tim akan memperbaiki conveyor di atas. Lanjut atau stop?", "SOP-K3-003", ["40 km"]),
    (P, "Apa yang harus dilakukan pada sabuk pengaman tubuh yang pernah menahan orang jatuh?", "SOP-K3-003", ["dimusnahkan"]),
    (P, "Berapa jumlah minimal orang dalam tim kerja di ketinggian?", "SOP-K3-003", ["2 orang"]),
    (C, "naik tangga alat berat aturannya gimana?", "SOP-K3-003", ["tiga titik kontak"]),
    # ---- SOP-K3-004 Peledakan ----
    (P, "Seberapa jauh semua orang harus menjauh dari lokasi blasting?", "SOP-K3-004", ["500 meter"]),
    (P, "Unit alat berat harus diparkir minimal berapa meter dari titik ledakan?", "SOP-K3-004", ["300 meter"]),
    (S, "Setelah ledakan ada satu lubang yang tidak meletus. Kapan orang boleh mendekat?", "SOP-K3-004", ["30 menit"]),
    (C, "boleh ga ngebor deket lubang yg gagal ledak?", "SOP-K3-004", ["1 meter"]),
    (P, "Sinyal sirine apa yang menandakan area sudah aman pasca peledakan?", "SOP-K3-004", ["2 kali"]),
    (S, "Batu hasil ledakan terlempar sampai ke luar zona aman. Masuk kategori risiko apa?", "SOP-K3-004", ["kritis"]),
    (P, "Dokumen izin apa yang wajib dimiliki orang yang melakukan peledakan?", "SOP-K3-004", ["KIM"]),
    (S, "Detektor menangkap sambaran petir 8 km dari lokasi saat pengisian bahan peledak. Apa tindakannya?", "SOP-K3-004", ["dihentikan"]),
    # ---- SOP-K3-005 Lereng ----
    (P, "Berapa angka faktor keamanan paling rendah untuk lereng keseluruhan?", "SOP-K3-005", ["1,3"]),
    (S, "Radar menunjukkan dinding pit bergeser 20 mm sehari. Status apa dan apa yang dilakukan?", "SOP-K3-005", ["Siaga"]),
    (S, "Pergerakan lereng terbaca 5 mm per hari. Apa respons yang tepat?", "SOP-K3-005", ["4 jam"]),
    (P, "Tinggi maksimum satu jenjang tambang menurut desain?", "SOP-K3-005", ["10 meter"]),
    (C, "abis ujan deras >50mm geotek wajib ngecek ga?", "SOP-K3-005", ["50 mm"]),
    (P, "Retakan sebesar apa yang wajib dilaporkan sebagai hazard?", "SOP-K3-005", ["5 cm"]),
    (P, "Data radar pemantau kemiringan ditinjau setiap berapa jam?", "SOP-K3-005", ["12 jam"]),
    # ---- SOP-K3-006 Kelelahan ----
    (P, "Berapa lama operator wajib beristirahat tidur dalam sehari sebelum mulai bekerja?", "SOP-K3-006", ["6 jam"]),
    (S, "Tensi operator terukur 165/95 saat cek awal shift. Boleh mengemudi?", "SOP-K3-006", ["160"]),
    (S, "Kamera kabin dua kali mendeteksi operator tertidur sesaat di shift yang sama. Apa keputusannya?", "SOP-K3-006", ["dibebastugaskan"]),
    (C, "shift malem maks brp hari berturut2?", "SOP-K3-006", ["7 hari"]),
    (P, "Berapa durasi jeda pendek melawan kantuk pada shift malam dan seberapa sering?", "SOP-K3-006", ["10 menit"]),
    (P, "Berapa kadar alkohol yang diperbolehkan saat tes awal shift?", "SOP-K3-006", ["0,00"]),
    # ---- SOP-K3-007 Darurat ----
    (P, "Tim tanggap darurat harus tiba di lokasi dalam pit maksimal berapa menit?", "SOP-K3-007", ["10 menit"]),
    (S, "Truk terbakar di tengah jalan angkut. Operator harus menjauh berapa meter dan ke arah mana?", "SOP-K3-007", ["30 meter"]),
    (P, "Kapan pekerjaan di lapangan terbuka boleh dimulai kembali setelah badai petir?", "SOP-K3-007", ["30 menit"]),
    (P, "Seberapa sering latihan simulasi keadaan darurat diadakan?", "SOP-K3-007", ["6 bulan"]),
    (C, "kalo ada kecelakaan, manggil di radio ngomong apa?", "SOP-K3-007", ["EMERGENCY"]),
    (P, "Dalam berapa lama Kepala Teknik Tambang harus diberi tahu bila ada korban?", "SOP-K3-007", ["1 jam"]),
    # ---- SOP-ENV-001 B3 ----
    (S, "Selang hidrolik pecah dan 300 liter oli tumpah ke sungai kecil. Masuk kategori apa?", "SOP-ENV-001", ["Besar"]),
    (P, "Bolehkah ceceran solar dibersihkan dengan menyemprotkan air?", "SOP-ENV-001", ["dilarang"]),
    (P, "Tanah yang tercemar oli harus dibawa ke tempat penyimpanan limbah paling lambat kapan?", "SOP-ENV-001", ["24 jam"]),
    (C, "isi spill kit minimal apa aja?", "SOP-ENV-001", ["absorbent pad"]),
    # ---- SOP-ENV-002 AAT ----
    (P, "Air keluaran kolam sedimen memiliki keasaman di bawah 6. Bahan apa yang ditambahkan?", "SOP-ENV-002", ["kapur"]),
    (P, "Batas internal kandungan besi total pada air buangan?", "SOP-ENV-002", ["7 mg/L"]),
    (P, "Kapan lumpur di kolam pengendapan harus dikeruk?", "SOP-ENV-002", ["50%"]),
    (C, "ph outlet KPL dicek tiap kapan?", "SOP-ENV-002", ["harian"]),
    # ---- SOP-OPS-001 P2H ----
    (P, "Komponen apa saja yang bila rusak membuat unit tidak boleh jalan sama sekali?", "SOP-OPS-001", ["Rem"]),
    (S, "Wiper truk rusak tapi rem dan kemudi normal. Kapan wiper harus diperbaiki?", "SOP-OPS-001", ["3 hari"]),
    (P, "Apa arti jika operator terus-menerus menambah oli mesin setiap shift?", "SOP-OPS-001", ["kebocoran"]),
    (C, "form P2H dikumpul paling telat kapan?", "SOP-OPS-001", ["1 jam"]),
    # ---- SOP-K3-008 Investigasi ----
    (S, "Seorang pekerja patah kaki dan tidak bisa masuk kerja besok. Dalam berapa jam laporan awal harus dibuat?", "SOP-K3-008", ["4 jam"]),
    (P, "Metode analisis akar masalah apa untuk kejadian berisiko rendah dan sedang?", "SOP-K3-008", ["5 Why"]),
    (P, "Berapa hari batas penyelesaian investigasi kejadian kritis?", "SOP-K3-008", ["14 hari"]),
    (P, "Urutan prioritas tindakan perbaikan dimulai dari apa?", "SOP-K3-008", ["eliminasi"]),
    (P, "Dalam berapa jam peringatan keselamatan harus disebarkan setelah insiden tinggi?", "SOP-K3-008", ["48 jam"]),
    # ---- MAN-DT100 ----
    (S, "Indikator suhu air radiator truk 100 ton naik melewati 105 derajat. Apa kemungkinan penyebabnya?", "MAN-DT100", ["thermostat"]),
    (C, "DT-100 lampu tekanan oli nyala pas idle, harus ngapain?", "MAN-DT100", ["matikan mesin"]),
    (P, "Berapa tekanan angin ban truk DT-100 dalam keadaan dingin?", "MAN-DT100", ["7,0 bar"]),
    (P, "Tegangan pengisian normal ketika mesin truk menyala berapa volt?", "MAN-DT100", ["27"]),
    (S, "Bak truk naiknya pelan banget saat membuang muatan. Apa yang perlu diperiksa?", "MAN-DT100", ["relief"]),
    (P, "Pada servis berkala berapa jam oli hidrolik dan oli transmisi DT-100 diganti?", "MAN-DT100", ["2000"]),
    (S, "Truk miring ke satu sisi dan guncangannya keras. Komponen apa yang dicurigai?", "MAN-DT100", ["suspensi"]),
    # ---- MAN-EX2000 ----
    (P, "Kisaran tekanan pilot yang normal pada excavator kelas 200 ton?", "MAN-EX2000", ["30–35 bar"]),
    (S, "Excavator bunyi gemeretak saat berputar dan gerakan putarnya tersendat. Apa yang diperiksa?", "MAN-EX2000", ["swing"]),
    (C, "track EX kendur, sag normalnya brp mm?", "MAN-EX2000", ["30–50 mm"]),
    (P, "Berapa nilai resistansi yang benar antara jalur CAN-H dan CAN-L?", "MAN-EX2000", ["60 ohm"]),
    (P, "Suhu kerja normal oli hidrolik excavator berapa derajat?", "MAN-EX2000", ["50–80"]),
    # ---- MAN-BD375 ----
    (S, "Dozer kehilangan tenaga dan oli power train panas saat mendorong material berat terus-menerus. Saran?", "MAN-BD375", ["stall"]),
    (P, "Bagaimana membersihkan radiator dozer yang tersumbat tanpa membongkar?", "MAN-BD375", ["fan reverse"]),
    # ---- REG-001 ----
    (P, "Peraturan menteri ESDM mana yang mengatur kaidah pertambangan yang baik dan pengawasannya?", "REG-001", ["26 Tahun 2018"]),
    (P, "Siapa pejabat pemerintah yang mengawasi pelaksanaan kaidah teknik pertambangan?", "REG-001", ["Inspektur Tambang"]),
    (C, "SMKP minerba itu isinya elemen apa aja?", "REG-001", ["kebijakan"]),
    # ---- OPS-002 ----
    (S, "Shift ini OB 30.000 BCM dan batubara 6.000 ton. Berapa stripping ratio-nya dan bagaimana rumusnya?", "OPS-002", ["BCM"]),
    (P, "Target ketersediaan fisik dump truck berapa persen?", "OPS-002", ["90%"]),
    (P, "Jam berapa shift malam dimulai dan berakhir?", "OPS-002", ["18.00"]),
    # ---- GLOS-001 ----
    (P, "Apa beda volume material in-situ dengan volume setelah digali?", "GLOS-001", ["LCM"]),
    (C, "LTI tuh maksudnya apa?", "GLOS-001", ["shift berikutnya"]),
    # ---- UNANSWERABLE (domain tambang, jawaban tidak ada di dokumen) ----
    (U, "Berapa harga satu ban baru untuk dump truck DT-100?", None, []),
    (U, "Berapa kapasitas tangki bahan bakar DT-100?", None, []),
    (U, "Siapa nama Kepala Teknik Tambang di PT Nusantara Mineral Sejahtera saat ini?", None, []),
    (U, "Berapa gaji pokok operator excavator?", None, []),
    (U, "Berapa target produksi batubara tahunan perusahaan?", None, []),
    (U, "Berapa interval penggantian filter udara kabin AC excavator EX-2000?", None, []),
    (U, "Merek bahan peledak apa yang dipakai di site?", None, []),
    (U, "Berapa denda dari pemerintah jika terjadi kecelakaan fatal?", None, []),
    (U, "Berapa kedalaman maksimum pit yang diizinkan?", None, []),
    (U, "Berapa jumlah unit dump truck yang beroperasi di Pit Barat?", None, []),
    (U, "Berapa konsumsi solar per jam dump truck DT-100?", None, []),
    (U, "Berapa nomor telepon klinik site?", None, []),
]

OUT_OF_SCOPE = [
    "Bagaimana resep rendang padang?", "Siapa juara piala dunia 2022?", "Rekomendasi film horor terbaru",
    "Cara membuat website dengan React", "Berapa harga saham BBCA hari ini?", "Tips diet sehat untuk menurunkan berat badan",
    "Jadwal kereta Jakarta ke Bandung", "Terjemahkan kalimat ini ke bahasa Jepang: selamat pagi",
    "Siapa penemu bola lampu?", "Cara merawat kucing persia",
]

if __name__ == "__main__":
    out = Path(__file__).with_name("rag_qa_v2.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for t, q, d, k in QA:
            f.write(json.dumps({"type": t, "question": q, "doc_id": d, "keywords": k}, ensure_ascii=False) + "\n")
        for q in OUT_OF_SCOPE:
            f.write(json.dumps({"type": "out_of_scope", "question": q, "doc_id": None, "keywords": []},
                               ensure_ascii=False) + "\n")
    from collections import Counter
    print(out, Counter(t for t, *_ in QA) | {"out_of_scope": len(OUT_OF_SCOPE)})
