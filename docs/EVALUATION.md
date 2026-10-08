# Metodologi Evaluasi RAG (v2)

## Mengapa v1 tidak cukup
Set v1 (`data/eval/rag_qa.jsonl`, 41 pertanyaan) memakai kosakata yang hampir sama dengan dokumen, sehingga
BM25 saja sudah mencapai Hit@5 = 1,00 dan jawaban "benar" 0,93. Angka itu **terlalu optimis**: pengguna
lapangan bertanya dengan sinonim, singkatan, dan situasi konkret, bukan menyalin kalimat SOP.

## Set v2 (`data/eval/rag_qa_v2.jsonl`, 115 item, dibangun oleh `data/eval/build_rag_qa_v2.py`)
| Tipe | n | Contoh | Menguji |
|---|---|---|---|
| paraphrase | 55 | "laju maksimal truk jungkit saat jalan licin" | sinonim, susunan kalimat berbeda |
| colloquial | 17 | "bro kalo jalan hauling kering DT boleh lari brp?" | singkatan, gaya chat |
| scenario | 21 | "Tensi operator 165/95, boleh mengemudi?" | memetakan situasi ke aturan |
| unanswerable | 12 | "Berapa kapasitas tangki solar DT-100?" (tidak ada di dokumen) | menolak, bukan mengarang |
| out_of_scope | 10 | "Resep rendang" | gate topik |

Aturan penulisan: pertanyaan ditulis tanpa melihat skor sistem; setiap fakta acuan (`keywords`) diverifikasi
otomatis memang ada di dokumen acuannya.

## Metrik
- **Retrieval**: Hit@1, Hit@5, MRR (dokumen acuan), **Evidence@5** (fakta acuan benar-benar ada di 5 chunk teratas — lebih ketat dari Hit).
- **End-to-end**: *answer_correct* (semua fakta acuan muncul di jawaban, toleran format angka 1,8/1.8 dan tanda –/-),
  *false_refusal* (menolak padahal jawabannya ada), *refusal_rate* untuk unanswerable + out-of-scope.
- **LLM-as-judge** (opsional, `eval/judge.py`): faithfulness, relevance, correctness 0–1. Aktif bila
  `MINESENSE_JUDGE_PROVIDER`/LLM provider bukan `extractive`; dibatasi `MINESENSE_JUDGE_MAX_ITEMS` (default 60).
  Validasi manual ±20 penilaian juri sebelum memercayainya.
- **Interval kepercayaan 95%** (bootstrap 1.000×) untuk setiap metrik.

## Kalibrasi gate relevansi (anti-overfitting)
Skala cosine berbeda antar embedder, sehingga ambang tetap 0,45 keliru. `minesense calibrate` / `benchmark`
mencari ambang dengan balanced accuracy terbaik menggunakan **set terpisah** (v1 + 18 pertanyaan OOS kalibrasi),
**bukan** set v2. Hasil di `artifacts/calibration.json`, dimuat otomatis per embedder.

## Regression gate (CI)
`minesense eval` membandingkan metrik v2 dengan `data/eval/baseline_metrics.json` untuk konfigurasi yang sama
(embedder | reranker | LLM). CI gagal bila Hit@5, Evidence@5, answer_correct, atau refusal_rate turun > 0,03.
Perbarui baseline secara sadar setelah perbaikan: `minesense eval --update-baseline` lalu commit file tersebut.

## Hasil (2026-10-01, LLM `extractive`, CPU)
| Konfigurasi | Hit@5 | Evidence@5 | Jawaban benar | Penolakan keliru | Menolak tak-terjawab/OOS |
|---|---|---|---|---|---|
| v1 (referensi lama) — hashing hybrid | 1,00 | 1,00 | 0,93 | – | 1,00 (OOS saja) |
| **v2** — hashing hybrid, ambang manual 0,45 | 0,96 | 0,88 | 0,61 [0,51–0,72] | 0,29 [0,20–0,39] | 0,59 [0,41–0,82] |
| **v2** — hashing hybrid, ambang terkalibrasi 0,35 (**default sekarang**) | 0,96 | 0,88 | **0,75** [0,67–0,84] | **0,05** [0,01–0,10] | 0,45 [0,27–0,68] |
| v2 — MiniLM / e5-large / + reranker | _jalankan `minesense benchmark --full` (butuh unduh model)_ | | | | |

Temuan:
1. Pada pertanyaan realistis, kualitas end-to-end turun dari 0,93 → 0,61. **Retrieval masih baik** (Hit@5 0,96);
   penyebab utama kegagalan adalah **gate relevansi yang terlalu ketat** (29% penolakan keliru), bukan pencarian.
2. Kalibrasi ambang pada set terpisah memangkas penolakan keliru 0,29 → 0,05 dan menaikkan jawaban benar
   0,61 → 0,75 (interval tidak tumpang-tindih → perbaikan nyata). **Harganya**: penolakan out-of-scope turun
   1,00 → 0,80 (2 dari 10 lolos: "tips diet", "terjemahkan ke bahasa Jepang") dan unanswerable 0,25 → 0,17.
   Trade-off ini diterima sebagai default karena dengan LLM aktif, prompt sistem tetap memaksa jawaban
   "tidak ditemukan" untuk konteks yang tidak relevan — verifikasi dengan menjalankan eval ber-LLM.
3. Pertanyaan **unanswerable** hanya ditolak ±25%: mode `extractive` selalu menyusun kalimat dari dokumen
   terdekat. Ini harus diukur ulang dengan LLM aktif — kemampuan menolak adalah alasan utama memakai LLM/fine-tune.
4. Mode dense hashing (Hit@1 0,78) lebih lemah dari BM25 untuk Hit@5 (0,89 vs 0,97) — hybrid tetap terbaik.
   Embedder neural diharapkan memperbaiki tipe *paraphrase* & *scenario*; buktikan dengan benchmark.

Laporan lengkap: `docs/eval/benchmark_baseline_hashing.md` (dibuat ulang di `artifacts/reports/benchmark.md`).
