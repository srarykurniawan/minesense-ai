# Benchmark Retrieval & RAG — set evaluasi v2

_Dibuat 2026-10-01 15:58 · 115 pertanyaan (parafrase, kolokial, skenario, tak-terjawab, di luar topik) · LLM: `extractive`_

## Retrieval (pertanyaan yang jawabannya ada di dokumen)

| Konfigurasi | Mode | Hit@1 | Hit@5 | MRR | Evidence@5 |
|---|---|---|---|---|---|
| Hashing n-gram (ambang lama manual 0,45) | dense | 0.78 | 0.89 | 0.83 | 0.84 |
| Hashing n-gram (ambang lama manual 0,45) | sparse | 0.75 | 0.97 | 0.84 | 0.87 |
| Hashing n-gram (ambang lama manual 0,45) | hybrid | 0.82 | 0.96 | 0.88 | 0.88 |
| Hashing n-gram + ambang terkalibrasi (default) | dense | 0.78 | 0.89 | 0.83 | 0.84 |
| Hashing n-gram + ambang terkalibrasi (default) | sparse | 0.75 | 0.97 | 0.84 | 0.87 |
| Hashing n-gram + ambang terkalibrasi (default) | hybrid | 0.82 | 0.96 | 0.88 | 0.88 |

## End-to-end (mode hybrid, termasuk gate relevansi & LLM)

| Konfigurasi | Jawaban benar | Penolakan keliru | Menolak yg tak terjawab/OOS | Ambang dense | Build index (dtk) | p50 latensi (ms) |
|---|---|---|---|---|---|---|
| Hashing n-gram (ambang lama manual 0,45) | 0.61 [0.51–0.72] | 0.29 [0.20–0.39] | 0.59 [0.41–0.82] | 0.45 | 1.07 | 2.4 |
| Hashing n-gram + ambang terkalibrasi (default) | 0.75 [0.67–0.84] | 0.05 [0.01–0.10] | 0.45 [0.27–0.68] | 0.35 | 0.09 | 2.5 |

## Rekomendasi

- Retrieval terbaik (Evidence@5 hybrid): **Hashing n-gram (ambang lama manual 0,45)**
- End-to-end terbaik (jawaban benar + 0,5 × penolakan tepat): **Hashing n-gram + ambang terkalibrasi (default)**

Angka dalam [ ] adalah interval kepercayaan 95% (bootstrap). Bila interval dua konfigurasi banyak tumpang-tindih, perbedaannya belum meyakinkan — perbesar set evaluasi sebelum memutuskan. Metrik penolakan sangat dipengaruhi LLM: mode `extractive` tidak bisa menolak pertanyaan yang jawabannya tidak ada; jalankan ulang dengan LLM aktif untuk angka penolakan yang representatif.

### 15 kegagalan pertama — Hashing n-gram + ambang terkalibrasi (default)

| Tipe | Pertanyaan | Harapan | Masalah | Sumber disitasi |
|---|---|---|---|---|
| paraphrase | Saat permukaan jalan angkut licin sehabis hujan, laju maksimal truk jungkit diba | SOP-K3-001: 25 km | fakta acuan tidak ada di jawaban | SOP-K3-001, SOP-K3-003 |
| paraphrase | Ketika membawa muatan di tanjakan curam di atas 8 persen, berapa kecepatan terti | SOP-K3-001: 20 km | fakta acuan tidak ada di jawaban | SOP-K3-003, SOP-K3-003, SOP-K3-001 |
| paraphrase | Apa prinsip dasar penguncian personal dalam prosedur isolasi energi? | SOP-K3-002: satu gembok | fakta acuan tidak ada di jawaban | SOP-K3-002, SOP-K3-002 |
| scenario | Mekanik A lupa melepas gemboknya lalu pulang. Unit harus segera dipakai. Bagaima | SOP-K3-002: KTT | fakta acuan tidak ada di jawaban | SOP-K3-002, SOP-K3-002, SOP-OPS-001 |
| paraphrase | Sebelum bekerja di kolong bak truk yang terangkat, alat penopang apa yang wajib  | SOP-K3-002: body prop | fakta acuan tidak ada di jawaban | SOP-K3-003, SOP-K3-003, SOP-K3-003 |
| paraphrase | Apa konsekuensi bagi pekerja yang melanggar prosedur penguncian energi? | SOP-K3-002: Life Saving Rules | fakta acuan tidak ada di jawaban | SOP-K3-002, SOP-K3-002 |
| paraphrase | Seberapa jauh semua orang harus menjauh dari lokasi blasting? | SOP-K3-004: 500 meter | fakta acuan tidak ada di jawaban | SOP-K3-007, SOP-K3-002, SOP-K3-007 |
| scenario | Setelah ledakan ada satu lubang yang tidak meletus. Kapan orang boleh mendekat? | SOP-K3-004: 30 menit | fakta acuan tidak ada di jawaban | SOP-K3-004, SOP-K3-002 |
| scenario | Batu hasil ledakan terlempar sampai ke luar zona aman. Masuk kategori risiko apa | SOP-K3-004: kritis | fakta acuan tidak ada di jawaban | SOP-ENV-002, SOP-K3-004, SOP-K3-005 |
| scenario | Radar menunjukkan dinding pit bergeser 20 mm sehari. Status apa dan apa yang dil | SOP-K3-005: Siaga | fakta acuan tidak ada di jawaban | SOP-K3-001, SOP-K3-001, SOP-K3-005 |
| colloquial | abis ujan deras >50mm geotek wajib ngecek ga? | SOP-K3-005: 50 mm | ditolak (false refusal) | - |
| scenario | Tensi operator terukur 165/95 saat cek awal shift. Boleh mengemudi? | SOP-K3-006: 160 | fakta acuan tidak ada di jawaban | SOP-K3-001, SOP-K3-006, SOP-OPS-001 |
| colloquial | kalo ada kecelakaan, manggil di radio ngomong apa? | SOP-K3-007: EMERGENCY | ditolak (false refusal) | - |
| paraphrase | Dalam berapa lama Kepala Teknik Tambang harus diberi tahu bila ada korban? | SOP-K3-007: 1 jam | fakta acuan tidak ada di jawaban | SOP-K3-002, REG-001, GLOS-001 |
| colloquial | ph outlet KPL dicek tiap kapan? | SOP-ENV-002: harian | ditolak (false refusal) | - |