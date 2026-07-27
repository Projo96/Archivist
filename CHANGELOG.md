# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [0.1.0-alpha] — 2026-07-27

First published version. The pipeline is operational, but interfaces and
index format may still change.

### Added
- Docling-based ingestion with header-aware chunking on H3/H4/H5 and a
  four-way classification of template fields
- `merge_wrapped_headers()` for titles split across lines by the original
  Word document
- LangChain/LCEL query chain, with optional query expansion
  (`MultiQueryRetriever`) behind a feature flag
- FastAPI service: native endpoints (`/query`, `/query/stream`) and
  OpenAI-compatible ones (`/v1/chat/completions`, `/v1/models`)
- RAG bypass for Open WebUI's background requests (title, tags,
  follow-up generation)
- Docker Compose stack: Qdrant + API + Open WebUI, with Ollama on the host
- `inspect_docling.py` for inspecting Docling output, including a
  `--redact` mode that never prints document content
- Makefile covering the recurring operations

### Fixed
- Chunks from a previous version of a modified document were left in
  Qdrant, so two versions of the same text coexisted in the index
- The synchronous chain call inside the async `/v1/chat/completions`
  handler blocked the event loop for the whole generation
- Fragile LLM access through `chain.steps[-2]`, replaced by the direct
  reference stored in `app.state.llm`
- `/sources` raised on points with a non-conforming payload
- Undeclared `openai` dependency: ingestion now uses `OllamaEmbeddings`,
  like the rest of the pipeline
- The API container reinstalled dependencies on every start; there is
  now a Dockerfile with cached layers
- Docker volume paths pointed outside the repository

### Notes
- No automated tests, no systematic retrieval evaluation
- Chunking thresholds are tuned to one specific document template
