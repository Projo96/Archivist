"""
api.py — API REST per il RAG (LangChain LCEL)
==============================================

Espone la pipeline di query.py via HTTP, in due dialetti:

- endpoint "nativi" (/query, /query/stream) — formato nostro, comodo
  per script e test
- endpoint compatibile OpenAI (/v1/chat/completions, /v1/models) —
  serve per collegare Open WebUI (o qualunque client che parli il
  protocollo OpenAI) come front-end della chat

I client (llm, embeddings, vectorstore, retriever, chain) vengono
costruiti UNA VOLTA all'avvio in lifespan() e condivisi tra le richieste:
istanziare OllamaEmbeddings e QdrantVectorStore a ogni richiesta sarebbe
uno spreco misurabile.

Avvio locale:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient
from starlette.concurrency import run_in_threadpool

from config import cfg
from query import (
    PROMPT_TEMPLATE,
    build_rag_chain,
    build_rag_chain_with_sources,
    extract_sources,
    format_docs,
    get_embeddings,
    get_llm,
    get_retriever,
    get_vectorstore,
)

# Open WebUI, oltre ai messaggi dell'utente, manda richieste automatiche
# in background per generare il titolo della conversazione, i tag e le
# domande di follow-up. Tutte iniziano con questo prefisso.
# Se le lasciassimo passare dalla pipeline completa, ogni messaggio
# scatenerebbe 3-4 retrieval + generazioni inutili: latenza moltiplicata
# e log illeggibili. Le intercettiamo e le mandiamo dritte al LLM.
WEBUI_TASK_PREFIX = "### Task:"


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — inizializzazione e cleanup
#
# asynccontextmanager lifespan sostituisce gli event handler
# @app.on_event("startup"). Il codice prima di "yield" gira all'avvio,
# quello dopo allo shutdown. I componenti finiscono in app.state, il
# namespace condiviso di FastAPI.
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising RAG components...")
    logger.info(f"  Ollama:     {cfg.OLLAMA_BASE_URL} ({cfg.LLM_MODEL})")
    logger.info(f"  Qdrant:     {cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}/{cfg.QDRANT_COLLECTION}")
    logger.info(f"  Query exp.: {'on' if cfg.ENABLE_QUERY_EXPANSION else 'off'}")

    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    retriever = get_retriever(vectorstore, llm)

    # Il LLM è tenuto a parte perché lo streaming lo usa direttamente,
    # senza passare dalla catena (vedi generate() più sotto).
    app.state.llm = llm

    # Catena principale (risposta senza fonti)
    app.state.chain = build_rag_chain(retriever, llm)

    # Catena con fonti — usata da /query e dal ramo non-streaming
    app.state.chain_with_sources = build_rag_chain_with_sources(retriever, llm)

    # Retriever separato — serve per recuperare i doc PRIMA di iniziare
    # lo streaming, così le fonti si possono mandare come primo evento
    app.state.retriever = retriever

    # Client Qdrant diretto per /health e /sources (operazioni non-LCEL)
    app.state.qdrant = QdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT)

    logger.success("RAG components ready.")
    yield
    logger.info("API shutdown.")


app = FastAPI(
    title="Local RAG API",
    description="Local knowledge base API (LangChain LCEL + Qdrant + Ollama)",
    version="0.1.0-alpha",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Modelli Pydantic
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class ChunkResult(BaseModel):
    text: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]
    model: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints di utilità
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Verifica che l'API sia viva e che Qdrant risponda."""
    try:
        info = app.state.qdrant.get_collection(cfg.QDRANT_COLLECTION)
        return {
            "status": "ok",
            "collection": cfg.QDRANT_COLLECTION,
            "vectors": info.points_count,
            "model": cfg.LLM_MODEL,
            "query_expansion": cfg.ENABLE_QUERY_EXPANSION,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/sources")
def list_sources():
    """Elenca tutti i documenti indicizzati con il numero di chunk."""
    points, _ = app.state.qdrant.scroll(
        collection_name=cfg.QDRANT_COLLECTION,
        with_payload=True,
        with_vectors=False,
        limit=10000,
    )
    sources: dict[str, int] = {}
    for p in points:
        # .get a cascata: un punto scritto da una versione precedente
        # con payload piatto non deve far esplodere l'endpoint.
        src = (p.payload or {}).get("metadata", {}).get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    return {
        "total_documents": len(sources),
        "total_chunks": sum(sources.values()),
        "documents": [{"name": k, "chunks": v} for k, v in sorted(sources.items())],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint /query — risposta completa + fonti
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Retrieval + generazione, con la lista delle fonti usate.

    Handler sincrono di proposito: FastAPI esegue le funzioni `def`
    (non `async def`) in un threadpool, quindi la catena bloccante non
    congela l'event loop.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be empty")

    logger.info(f"Query: {req.question}")

    result = app.state.chain_with_sources(req.question)
    answer = result["answer"]
    docs = result["docs"]

    if not docs:
        return QueryResponse(
            answer="I could not find relevant information in the knowledge base.",
            sources=[],
            model=cfg.LLM_MODEL,
        )

    return QueryResponse(
        answer=answer,
        sources=[ChunkResult(**s) for s in extract_sources(docs)],
        model=cfg.LLM_MODEL,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint /query/stream — risposta streaming token per token
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Versione streaming di /query, in Server-Sent Events.

    Recuperiamo i documenti PRIMA di aprire lo stream, così possiamo
    mandare le fonti come primo evento: il client può mostrarle subito,
    mentre la risposta si sta ancora generando.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be empty")

    logger.info(f"Query stream: {req.question}")

    docs: list[Document] = await app.state.retriever.ainvoke(req.question)

    if not docs:

        async def empty():
            yield "data: {}\n\n".format(
                json.dumps({"token": "I could not find relevant information in the knowledge base."})
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(empty(), media_type="text/event-stream")

    sources = extract_sources(docs)

    prompt_value = PROMPT_TEMPLATE.invoke(
        {"context": format_docs(docs), "question": req.question}
    )

    async def generate():
        # Primo evento: le fonti
        yield f"data: {json.dumps({'sources': [{'source': s['source']} for s in sources]})}\n\n"

        logger.debug(f"Full prompt:\n{prompt_value.to_string()}")

        # Streaming direttamente sul LLM: la catena completa rifarebbe
        # il retrieval, che abbiamo già fatto qui sopra.
        async for chunk in app.state.llm.astream(prompt_value.to_messages()):
            if chunk.content:
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint /v1/chat/completions — compatibile OpenAI (per Open WebUI)
# ─────────────────────────────────────────────────────────────────────────────


def _openai_chunk(chat_id: str, delta: dict, finish_reason=None) -> str:
    """Serializza un chunk nel formato SSE dell'API OpenAI."""
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "model": cfg.LLM_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions")
async def openai_compat(body: dict):
    """Endpoint compatibile con l'API OpenAI.

    Open WebUI lo chiama esattamente come chiamerebbe OpenAI: intercettiamo
    l'ultimo messaggio utente, facciamo RAG, e restituiamo la risposta nel
    formato atteso.
    """
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages in request")

    question = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    if not question:
        raise HTTPException(status_code=400, detail="No user message in request")

    chat_id = f"rag-{int(time.time())}"
    stream = bool(body.get("stream", False))

    # ── Richieste di servizio di Open WebUI: bypass della pipeline RAG ──
    if question.lstrip().startswith(WEBUI_TASK_PREFIX):
        logger.info("Open WebUI background request - bypassing RAG")
        result = await app.state.llm.ainvoke(question)
        content = result.content

        if stream:

            async def task_stream():
                yield _openai_chunk(chat_id, {"role": "assistant", "content": content})
                yield _openai_chunk(chat_id, {}, finish_reason="stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(task_stream(), media_type="text/event-stream")

        return {
            "id": chat_id,
            "object": "chat.completion",
            "model": cfg.LLM_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    logger.info(f"Query: {question}")

    if stream:
        # ── Streaming ──────────────────────────────────────────────────
        t0 = time.time()
        logger.info("  [1/2] Retrieval...")
        docs: list[Document] = await app.state.retriever.ainvoke(question)
        logger.info(f"  [1/2] Retrieval in {time.time() - t0:.1f}s - {len(docs)} chunks")

        sources = extract_sources(docs) if docs else []
        context_str = format_docs(docs) if docs else "No relevant documents found."
        prompt_value = PROMPT_TEMPLATE.invoke({"context": context_str, "question": question})

        async def generate():
            # Primo chunk: dichiara il ruolo (richiesto dal protocollo
            # OpenAI streaming, alcuni client non partono senza)
            yield _openai_chunk(chat_id, {"role": "assistant", "content": ""})

            t2 = time.time()
            logger.info("  [2/2] Generating answer (LLM streaming)...")
            token_count = 0

            async for chunk in app.state.llm.astream(prompt_value.to_messages()):
                if chunk.content:
                    token_count += 1
                    yield _openai_chunk(chat_id, {"content": chunk.content})

            logger.info(f"  [2/2] Generation in {time.time() - t2:.1f}s - {token_count} chunks")

            # Fonti in coda alla risposta, come testo markdown
            if sources:
                nomi = sorted({s["source"] for s in sources if s["source"]})
                if nomi:
                    fonti = "\n\n**Sources:**\n" + "\n".join(f"- {n}" for n in nomi)
                    yield _openai_chunk(chat_id, {"content": fonti})

            yield _openai_chunk(chat_id, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            logger.info(f"  Total request time: {time.time() - t0:.1f}s")

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── Non-streaming ──────────────────────────────────────────────────────
    # chain_with_sources è sincrona e bloccante: in un handler async va
    # eseguita in un thread separato, altrimenti blocca l'event loop e
    # l'API smette di rispondere a chiunque altro per tutta la generazione.
    result = await run_in_threadpool(app.state.chain_with_sources, question)
    answer = result["answer"]
    docs = result["docs"]

    if docs:
        nomi = sorted({d.metadata.get("source", "") for d in docs} - {""})
        if nomi:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {n}" for n in nomi)

    return {
        "id": chat_id,
        "object": "chat.completion",
        "model": cfg.LLM_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/models")
def list_models():
    """Endpoint richiesto da Open WebUI per listare i modelli disponibili."""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-internal",
                "object": "model",
                "created": 1700000000,
                "owned_by": "local",
            }
        ],
    }
