# MineSense — Laporan Evaluasi
_Dibuat: 2026-10-01 15:57_ · LLM: `extractive` · Embedder: `hashing`

| Metrik | Nilai |
|---|---|
| retrieval.dense.hit@1 | 0.8537 |
| retrieval.dense.hit@5 | 0.9756 |
| retrieval.dense.mrr | 0.8972 |
| retrieval.dense.keyword_recall@5 | 0.9512 |
| retrieval.sparse.hit@1 | 1.0 |
| retrieval.sparse.hit@5 | 1.0 |
| retrieval.sparse.mrr | 1.0 |
| retrieval.sparse.keyword_recall@5 | 1.0 |
| retrieval.hybrid.hit@1 | 0.9512 |
| retrieval.hybrid.hit@5 | 1.0 |
| retrieval.hybrid.mrr | 0.9715 |
| retrieval.hybrid.keyword_recall@5 | 1.0 |
| rag.keyword_recall | 0.9268 |
| rag.correct_citation_rate | 0.9756 |
| rag.grounded_rate | 1.0 |
| rag.oos_rejection | 1.0 |
| rag.injection_block_rate | 1.0 |
| rag.provider | extractive |
| rag.latency_ms_p50 | 1.6 |
| rag.latency_ms_p95 | 3.2 |
| classifier.random_split.category_macro_f1 | 1.0 |
| classifier.random_split.risk_macro_f1 | 0.8953 |
| classifier.random_split.risk_kritis_recall | 0.9762 |
| classifier.random_split.n_test | 280 |
| classifier.group_split.category_macro_f1 | 0.9524 |
| classifier.group_split.risk_macro_f1 | 0.8986 |
| classifier.group_split.risk_kritis_recall | 1.0 |
| classifier.group_split.n_test | 395 |
| extraction.in_distribution.field_accuracy | 1.0 |
| extraction.in_distribution.n_docs | 300 |
| extraction.hard.field_accuracy | 1.0 |
| extraction.hard.n_docs | 4 |

## Quality gates
| Gate | Nilai | Ambang | Status |
|---|---|---|---|
| retrieval.hybrid.hit@5 | 1.0 | 0.9 | ✅ |
| rag.keyword_recall | 0.9268 | 0.7 | ✅ |
| rag.oos_rejection | 1.0 | 0.8 | ✅ |
| classifier.group_split.category_macro_f1 | 0.9524 | 0.8 | ✅ |
| classifier.random_split.risk_kritis_recall | 0.9762 | 0.9 | ✅ |
| extraction.hard.field_accuracy | 1.0 | 0.9 | ✅ |

## RAG v2 — set evaluasi jujur (CI 95% bootstrap)

Judge: `tidak aktif (set MINESENSE_JUDGE_PROVIDER / LLM provider)`

| Kelompok | Metrik | Rata-rata | CI 95% | n |
|---|---|---|---|---|
| answerable | answer_correct | 0.753 | 0.667–0.839 | 93 |
| answerable | cited_correct_doc | 0.892 | 0.828–0.957 | 93 |
| answerable | evidence@5 | 0.882 | 0.806–0.946 | 93 |
| answerable | false_refusal | 0.054 | 0.011–0.097 | 93 |
| answerable | hit@1 | 0.817 | 0.731–0.892 | 93 |
| answerable | hit@5 | 0.957 | 0.903–0.989 | 93 |
| answerable | mrr | 0.878 | 0.817–0.932 | 93 |
| refusal | refusal_rate | 0.455 | 0.273–0.682 | 22 |

| Tipe | Metrik utama |
|---|---|
| paraphrase | answer_correct=0.85, cited_correct_doc=0.93, evidence@5=0.89, false_refusal=0.00, hit@1=0.87, hit@5=0.95, mrr=0.90 |
| colloquial | answer_correct=0.65, cited_correct_doc=0.71, evidence@5=1.00, false_refusal=0.29, hit@1=0.94, hit@5=1.00, mrr=0.97 |
| scenario | answer_correct=0.57, cited_correct_doc=0.95, evidence@5=0.76, false_refusal=0.00, hit@1=0.57, hit@5=0.95, mrr=0.73 |
| unanswerable | refusal_rate=0.17 |
| out_of_scope | refusal_rate=0.80 |

## Regression gate v2: **PASS** (`hashing|thr=0.35|rerank=none|llm=extractive`)

- hit@5: baseline 0.957 → sekarang 0.957 ✅
- evidence@5: baseline 0.8817 → sekarang 0.8817 ✅
- answer_correct: baseline 0.7527 → sekarang 0.7527 ✅
- refusal_rate: baseline 0.4545 → sekarang 0.4545 ✅

## Contoh kegagalan (untuk analisis error)

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
| scenario | Wiper truk rusak tapi rem dan kemudi normal. Kapan wiper harus diperbaiki? | SOP-OPS-001: 3 hari | fakta acuan tidak ada di jawaban | SOP-OPS-001, SOP-OPS-001, MAN-EX2000 |
| paraphrase | Apa arti jika operator terus-menerus menambah oli mesin setiap shift? | SOP-OPS-001: kebocoran | fakta acuan tidak ada di jawaban | SOP-OPS-001, SOP-OPS-001, MAN-BD375 |
| colloquial | form P2H dikumpul paling telat kapan? | SOP-OPS-001: 1 jam | ditolak (false refusal) | - |
| colloquial | DT-100 lampu tekanan oli nyala pas idle, harus ngapain? | MAN-DT100: matikan mesin | fakta acuan tidak ada di jawaban | MAN-DT100, MAN-DT100, MAN-DT100 |
| scenario | Bak truk naiknya pelan banget saat membuang muatan. Apa yang perlu diperiksa? | MAN-DT100: relief | fakta acuan tidak ada di jawaban | SOP-K3-001, SOP-K3-003, SOP-OPS-001 |