# Architecture notes

The longer version of the decisions summarised in the README. Written to answer "why is it like this?" six months from now, when nobody remembers.

## Why three separate processes

| Process | Runs | Why there |
|---|---|---|
| Ollama | host, native | direct GPU and RAM access; models are already downloaded and shared with other uses |
| Qdrant | container | stateful but with no host dependencies; the volume lives in `data/qdrant` |
| API + Open WebUI | container | heavy Python dependencies isolated from the dev environment |
| Ingestion | host | uses Docling, which pulls ~500 MB of layout models; putting it in the API image would triple its size for a process that runs occasionally |

Practical consequence: two requirements files. `requirements.txt` for the host (ingestion + CLI + everything), `requirements-api.txt` for the container (no Docling).

## The WSL2 / Docker network boundary

Three different addresses for the same thing, depending on who's asking:

- **host → Qdrant**: `127.0.0.1:6333` (port published by compose)
- **container → Qdrant**: `qdrant:6333` (internal Docker DNS)
- **container → Ollama**: `host.docker.internal:11434`, plus `extra_hosts: host-gateway`, which is required on Linux and on some Docker Desktop setups

This is why `docker-compose.yml` overrides `QDRANT_HOST` and `OLLAMA_BASE_URL` from `.env`: the `.env` holds the values that are valid *from the host*, and compose corrects them for containers.

A note on `127.0.0.1` rather than `localhost`: on some setups `localhost` resolves to `::1` (IPv6) first, and the client hangs until timeout while the service is only listening on IPv4.

## Chunking: the four-way classification

`MarkdownHeaderTextSplitter` produces one "leaf" per block of text under a heading, carrying the heading chain as metadata. Each leaf is then classified:

| Condition | Outcome | Reason |
|---|---|---|
| field name in `ADMIN_FIELDS_BLACKLIST` | dropped | references, versions, statuses: noise for semantic retrieval |
| body `< NEAR_EMPTY_CHARS` (15) | dropped | placeholder or empty sign-off field |
| body `< MIN_CHUNK_CHARS` (150) | merged into the `Attributes` chunk | real value, but too short to be a sensible chunk on its own |
| body `>= MIN_CHUNK_CHARS` | standalone chunk | substantial content |

A standalone chunk longer than `CHUNK_SIZE` gets re-split by character count. The detail that matters: it re-splits **the body**, not the text with the context path already prepended. Otherwise only the first fragment would carry `[Procedure > Field]` and the rest would start orphaned, with no indication of where they came from. This way every fragment gets its own prefix.

### `merge_wrapped_headers`

Some headings in the original `.docx` files contain a manual line break. Docling exports it faithfully, but the heading regex works line by line: it sees half a title, and the other half lands in the body. The heuristic used is quote counting — a title like `Step "Name"` always has an even number of double quotes. An odd number means the title probably continues on the next line, so lines are merged until the count is even again.

It's a heuristic, not a rule. It works on the source documents because that specific title shape is where the problem comes from. On a different corpus, verify it with `inspect_docling.py` first.

## The Qdrant payload

```json
{
  "text": "[Procedure > Field]\nchunk content...",
  "metadata": {
    "source": "document.docx",
    "source_path": "/full/path/document.docx",
    "file_hash": "sha256...",
    "chunk_index": 3,
    "total_chunks": 27,
    "extension": ".docx",
    "section_title": "Field",
    "parent_section": "Parent section or null"
  }
}
```

`text` sits at the root because `content_payload_key="text"` points there explicitly. Everything else is nested under `metadata` because `metadata_payload_key` defaults to the string `"metadata"`, and the library does `payload.get("metadata") or {}`. With flat metadata that lookup returns `{}` without raising: sources vanish from the answers and there's no error to notice. It's a silent failure, and it's why `/sources` uses chained `.get()` instead of direct indexing.

### Point identity and reindexing

A point's ID is `sha256(file_hash + "_" + chunk_index)`. Deterministic: reindexing the exact same file overwrites the same points instead of duplicating them.

But if the file **changes**, the hash changes and so do all the IDs: new points sit alongside the old ones instead of replacing them. Hence `delete_stale_chunks()`, which removes every point with the same `source_path` before the upsert. The filter is on `source_path` rather than `source` so that two identically named files in different subfolders don't wipe each other out.

Still uncovered: a file **deleted** from disk. Nobody comes along to clean up its chunks. That needs a diff between the indexed `source_path` values and the files actually present (on the roadmap).

## The LCEL chain

```
{"context": retriever | format_docs, "question": RunnablePassthrough()}
    -> PROMPT_TEMPLATE
    -> llm
    -> StrOutputParser()
```

The leading dictionary is a fork: the same question goes both to the retriever (which produces the context) and straight into the prompt's `{question}` slot. The result is a dict that fills the template placeholders.

The variant that also returns sources (`build_rag_chain_with_sources`) is a function rather than a pure chain, because the retrieved `Document` objects have to be kept and returned alongside the answer. A pure LCEL fork would run the retriever twice unless caching is added.

Consequence: that function is **synchronous and blocking**. Calling it from an `async def` handler without `run_in_threadpool` blocks the event loop for the whole generation — the API stops responding to everyone else. Plain `def` handlers are already run in a threadpool by FastAPI, so there it isn't needed.

## Streaming

On the streaming paths, retrieval happens **before** the stream opens, and then `astream()` runs directly on the LLM rather than on the full chain. Two reasons: sources can be sent as the first event (the client shows them immediately, while the answer is still generating), and the full chain would redo a retrieval that has already happened.

## English prompt, answers in the user's language

The system prompt is in English even when questions come in another language: small models follow English instructions more reliably. Rule 4 of the prompt requires answering in the language of the question while **not translating** technical terms, commands, menu paths and UI labels — which in the source documents are often English and must be reproduced verbatim, otherwise the reader goes looking for a menu entry that doesn't exist.
