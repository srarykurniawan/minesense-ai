# Model Card — MineSense AI

## 1. Incident Classifier (baseline)
- **Arsitektur**: FeatureUnion (leksikon domain 7 fitur + TF-IDF kata 1–2 gram + TF-IDF karakter 3–5 gram) → Logistic Regression (class_weight balanced). Dua head: kategori hazard (7 kelas) & tingkat risiko (4 kelas).
- **Aturan keselamatan**: risiko dinaikkan paksa ke `tinggi`/`kritis` bila teks memuat indikator LTI/fatal. Eskalasi bila risiko tinggi/kritis atau keyakinan kategori < 50%.
- **Data**: 1.400 laporan insiden sintetis (42 template kejadian × slot unit/lokasi/waktu/cuaca × konsekuensi), noise label 3%.
- **Evaluasi**:
  | Split | Macro-F1 kategori | Macro-F1 risiko | Recall "kritis" |
  |---|---|---|---|
  | Acak (stratified 80/20) | 1,00 | 0,92 | 0,98 |
  | Group split per template (pola kalimat tak pernah dilihat) | 0,95 | 0,90 | 1,00 |
  | Group split **tanpa** leksikon domain | 0,30 | — | — |
- **Catatan jujur**: leksikon ditulis oleh penulis yang mengetahui template — ada potensi kebocoran pengetahuan. Ukur ulang pada laporan insiden historis asli; target realistis macro-F1 0,75–0,85.

## 2. Incident Classifier (IndoBERT, opsional)
`indobenchmark/indobert-base-p1` fine-tune 5 epoch, lr 3e-5, early stopping, evaluasi group split. Jalankan notebook 02. Gunakan bila data asli > ~2.000 laporan berlabel dan pola bahasanya beragam.

## 3. LLM Domain (QLoRA)
- **Base**: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0). Alternatif: Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B (cek lisensi).
- **Metode**: QLoRA 4-bit NF4, r=16, α=32, dropout 0,05, semua proyeksi attention+MLP, lr 2e-4 cosine, 2 epoch, max_len 2048.
- **Dataset SFT** (`minesense sft`): ±1.260 contoh — incident JSON (500), ekstraksi JSON (300), diagnosa maintenance (300), QA bersitasi (82, upsample ×3), penolakan saat konteks salah (82, upsample ×2).
- **Tujuan**: format sitasi `[n]`, menolak bila konteks tidak relevan, output JSON valid, gaya bahasa domain. **Pengetahuan faktual tetap berasal dari RAG**, bukan dari bobot model.
- **Evaluasi**: `python -m minesense.finetune.eval_llm` — bandingkan base vs fine-tuned pada val set: json_valid, kategori_acc, risiko_acc, field_accuracy, citation_rate, keyword_overlap, refusal_acc, false_refusal, code_acc.
- **Status di repo ini**: skrip & notebook disediakan dan siap dijalankan di GPU; pelatihan tidak dijalankan di lingkungan pembuatan repo (tanpa GPU/akses HuggingFace). Catat hasil Anda di bagian ini setelah training.

## 4. Retrieval
Embedder default `hashing` (char n-gram + word n-gram, 768 dim). Eval 41 pertanyaan: hybrid Hit@5 1,00, MRR 0,97. Untuk data nyata dengan parafrase dan sinonim, gunakan `intfloat/multilingual-e5-base` atau `BAAI/bge-m3` dan **kalibrasi ulang** `MINESENSE_MIN_DENSE_SCORE` (cosine e5 untuk teks tak relevan bisa > 0,7).

## Keterbatasan & risiko
- Semua data sintetis; eval set kecil dan dekat dengan teks sumber (pertanyaan banyak memakai kata yang sama dengan dokumen) → angka optimistis.
- Ringkasan regulasi (REG-001) bersifat orientasi umum, **bukan nasihat hukum**; jawaban terkait kewajiban hukum harus diverifikasi ke teks resmi (JDIH ESDM) dan tim Legal/KTT.
- Sistem adalah **alat bantu keputusan**, bukan pengganti pengawas, juru ledak, geoteknik, atau KTT. Keputusan stop-work tetap pada manusia berwenang.
- Gate out-of-scope berbasis heuristik; pantau `intent=out_of_scope` dan feedback untuk kalibrasi.
