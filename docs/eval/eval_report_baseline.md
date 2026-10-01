# MineSense — Laporan Evaluasi
_Dibuat: 2026-10-01 14:14_ · LLM: `extractive` · Embedder: `hashing`

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
| rag.latency_ms_p50 | 2.4 |
| rag.latency_ms_p95 | 3.5 |
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