# SOURVETA Production Cleanup

This patch is based on the supplied full project audit and removes benchmark-specific logic from production while preserving the existing agent/RAG architecture.

## Removed from production logic
- Tata-specific page/value/entity targeting in `generation/rag.py`.
- Microbiology/bacteria-specific retrieval rescue and validation logic.
- Hardcoded benchmark financial answer generation.
- Old Tata loader/evaluation artifact.

## Preserved
- Multilingual E5 embeddings.
- Chroma `sourveta_multilingual` collection.
- BM25 + vector + RRF hybrid retrieval.
- Generic evidence rescue and reranking.
- Generic grounded generation and validation.
- Phase 7.3 generic OCR denoising.
- API security/rate limiting/document lifecycle.
- Streamlit frontend.

## Benchmark separation
Tata evaluation questions are moved under `evaluation/benchmarks/` and are no longer imported by production code. Set `SOURVETA_EVALUATION_FILE` to evaluate another benchmark.

## Runtime data
Existing local PDFs/chunks/status files are intentionally removed from the clean source tree. They should be generated at runtime and are ignored by Git. `scripts/cleanup_legacy_artifacts.ps1` can remove the old local artifacts from an existing checkout.

## Important
`.env` is never included in this patch. Keep API keys in environment variables or Streamlit secrets.
