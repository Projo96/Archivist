"""
query.py — Pipeline RAG con LangChain LCEL
===========================================

Contiene tutti i pezzi della catena di interrogazione:
  domanda → (opzionale) query expansion → retrieval → prompt → risposta

Il modulo è usabile in due modi:
- importato da api.py, che costruisce i componenti una volta all'avvio
- eseguito direttamente (`python query.py`) per una sessione interattiva
  da terminale, utile per testare il retrieval senza passare dall'API
"""

import argparse

# ── LangChain: modello e embeddings ──────────────────────────────────────────
# ChatOllama è il wrapper LangChain per Ollama. Internamente chiama lo
# stesso endpoint /api/chat, ma restituisce un AIMessage invece di una
# stringa grezza — è ciò che lo rende componibile in LCEL.
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ── LangChain: vector store ───────────────────────────────────────────────────
# QdrantVectorStore è il wrapper LangChain attorno al client Qdrant.
# Non sostituisce qdrant-client (lo usa internamente), ma espone
# .as_retriever(), che rende Qdrant componibile in una catena LCEL.
from langchain_qdrant import QdrantVectorStore

# ── LangChain: retrieval multi-query ─────────────────────────────────────────
# MultiQueryRetriever:
# 1. Prende la domanda originale
# 2. Chiede al LLM di generare N varianti
# 3. Fa retrieval su ogni variante
# 4. Deduplica i risultati
# È un Runnable, quindi entra direttamente nella catena con |
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# ── LangChain: catena LCEL ────────────────────────────────────────────────────
# ChatPromptTemplate: template dichiarativo per il prompt. Invece di
# costruire una f-string, dichiari gli "slot" ({context}, {question})
# e LCEL li riempie quando la catena viene invocata.
#
# RunnablePassthrough: lascia passare l'input invariato lungo la catena.
# Serve quando ti serve sia l'input originale (la domanda) sia un valore
# derivato (il contesto recuperato) nello stesso step successivo.
#
# StrOutputParser: estrae la stringa dall'AIMessage restituito dal modello.
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ── Qdrant client diretto ─────────────────────────────────────────────────────
# Usato solo per operazioni che non passano da LCEL (es. contare i vettori).
from qdrant_client import QdrantClient

from config import cfg

# ─────────────────────────────────────────────────────────────────────────────
# Prompt di sistema — le regole comportamentali del modello.
# Scritto in inglese anche per domande in italiano: i modelli piccoli
# seguono le istruzioni in inglese in modo più affidabile. La regola 4
# si occupa di far rispondere il modello nella lingua della domanda.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an internal technical assistant that answers questions
based exclusively on the company documents provided in the context.

Rules:
1. Answer ONLY using information present in the CONTEXT provided.
2. If the context doesn't contain enough information, say so clearly:
   "I couldn't find information about this in the available documents."
3. Always cite the source in square brackets at the end of each important
   statement, e.g.: [document-name.docx]
4. Answer in the same language as the question. Regardless of the answer
   language, keep technical terms, commands, menu paths, and UI labels
   exactly as they appear in the source documents — do not translate them.
5. For step-by-step procedures, list the steps as a numbered list.
6. Never invent information that isn't present in the context.
7. Get to the point immediately unless asked otherwise."""


# ─────────────────────────────────────────────────────────────────────────────
# Template del prompt utente
# ChatPromptTemplate.from_messages riceve una lista di tuple (ruolo, template).
# I {placeholder} vengono riempiti da LCEL quando la catena viene invocata.
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """CONTEXT (excerpts from company documents):
{context}

---

QUESTION: {question}

Answer based exclusively on the context above.""",
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions — costruiscono i client una volta sola
# ─────────────────────────────────────────────────────────────────────────────


def get_llm() -> ChatOllama:
    """Crea il client LLM.

    num_ctx (default 4096): il valore di default di Ollama è 2048 token.
    Con SYSTEM_PROMPT + TOP_K chunk di contesto (CHUNK_SIZE=1500 caratteri,
    che possono valere 2000+ token da soli) + num_predict, il totale sfora
    facilmente i 2048: la finestra di contesto scorre a metà generazione,
    le istruzioni di sistema scivolano fuori, e il modello continua a
    produrre testo senza più le sue istruzioni. È la causa più comune di
    risposte che partono bene e degenerano.

    Se cambi modello, alza num_ctx di conseguenza (LLM_NUM_CTX nel .env):
    non ha senso tenere 4096 su un modello che ne supporta 32k.

    repeat_penalty / repeat_last_n: valori più alti dei default di Ollama
    (1.1 / 64), che sui modelli piccoli non bastano a rompere i loop di
    paragrafi ripetuti. Ma attenzione a non esagerare: oltre ~1.3 il
    modello inizia a evitare parole comuni e produce testo malformato.
    """
    return ChatOllama(
        base_url=cfg.OLLAMA_BASE_URL,
        model=cfg.LLM_MODEL,
        temperature=cfg.LLM_TEMPERATURE,
        num_predict=cfg.LLM_NUM_PREDICT,
        num_ctx=cfg.LLM_NUM_CTX,
        repeat_penalty=cfg.LLM_REPEAT_PENALTY,
        repeat_last_n=cfg.LLM_REPEAT_LAST_N,
    )


def get_embeddings() -> OllamaEmbeddings:
    """Crea il client per gli embedding.

    Deve usare lo STESSO modello di ingest.py: vettori prodotti da modelli
    diversi non sono confrontabili, anche se hanno la stessa dimensione.
    """
    return OllamaEmbeddings(
        base_url=cfg.OLLAMA_BASE_URL,
        model=cfg.EMBEDDING_MODEL,
    )


def get_vectorstore(embeddings: OllamaEmbeddings) -> QdrantVectorStore:
    """Connette al Qdrant esistente (già popolato da ingest.py).

    from_existing_collection NON crea né svuota la collection: si connette
    soltanto a quella già esistente.

    content_payload_key="text" dice a LangChain che nel payload di ogni
    punto Qdrant il testo del documento si trova nella chiave "text"
    (esattamente quello che scrive ingest.py).

    metadata_payload_key viene lasciato al suo default. Il default della
    libreria è la stringa "metadata", e il codice fa
    `payload.get("metadata") or {}` — quindi si aspetta che TUTTI i
    metadata siano annidati sotto quella singola chiave, non sparsi a
    livello radice. ingest.py scrive il payload di conseguenza:
    {"text": ..., "metadata": {"source": ..., "chunk_index": ..., ...}}.
    Non serve passarlo esplicitamente perché corrisponde già.
    """
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=cfg.QDRANT_COLLECTION,
        host=cfg.QDRANT_HOST,
        port=cfg.QDRANT_PORT,
        content_payload_key="text",
    )


def get_retriever(vectorstore: QdrantVectorStore, llm: ChatOllama):
    """Crea il retriever — con o senza query expansion, secondo cfg.ENABLE_QUERY_EXPANSION.

    base_retriever fa una similarity search diretta: una sola chiamata di
    embedding sulla domanda originale, zero chiamate all'LLM prima del
    retrieval. È il percorso veloce.

    MultiQueryRetriever avvolge base_retriever: chiede all'LLM alcune
    varianti della domanda, fa similarity search per ciascuna, deduplica.
    Più lento (diverse chiamate sequenziali a Ollama prima di vedere il
    primo chunk), potenzialmente più recall su domande formulate in modo
    diverso dal linguaggio dei documenti.

    Entrambi i rami restituiscono un oggetto compatibile con .invoke() /
    .ainvoke() — chi chiama get_retriever() non deve sapere quale dei due
    sta usando.
    """
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.TOP_K})

    if not cfg.ENABLE_QUERY_EXPANSION:
        return base_retriever

    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


# ─────────────────────────────────────────────────────────────────────────────
# Funzioni di supporto per la catena
# ─────────────────────────────────────────────────────────────────────────────


def format_docs(docs: list[Document]) -> str:
    """Formatta i Document recuperati in una stringa di contesto.

    Niente "score" qui: il retriever standard di LangChain (e quindi anche
    MultiQueryRetriever, che ci si appoggia sopra) lo scarta per design —
    similarity_search() chiama similarity_search_with_score() ma
    restituisce solo i Document, buttando via il punteggio. Non è mai
    disponibile su doc.metadata, quindi non lo esponiamo come se lo fosse.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        # doc.page_content = il testo del chunk (chiave "text" del payload)
        # doc.metadata     = il contenuto di payload["metadata"]
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def extract_sources(docs: list[Document]) -> list[dict]:
    """Estrae le fonti dai Document per includerle nella risposta API.

    Non fa parte della catena LCEL principale — viene chiamata
    separatamente da api.py per costruire il campo "sources".
    """
    return [
        {
            "text": doc.page_content[:300],
            "source": doc.metadata.get("source", ""),
        }
        for doc in docs
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Costruzione della catena LCEL
# ─────────────────────────────────────────────────────────────────────────────


def build_rag_chain(retriever, llm: ChatOllama):
    """Costruisce la catena RAG principale con LCEL.

    L'operatore | è la composizione di LCEL: ogni componente riceve
    l'output del precedente.

        {"context": retriever | format_docs, "question": RunnablePassthrough()}
            → PROMPT_TEMPLATE
            → llm
            → StrOutputParser()

    Lo step iniziale è un dizionario con due chiavi:
    - "context":  passa la domanda al retriever, poi formatta i Document
    - "question": lascia passare la domanda invariata (RunnablePassthrough)

    Il risultato è un dict {"context": "...", "question": "..."} che
    riempie gli slot del PROMPT_TEMPLATE. Poi il prompt formattato va al
    LLM, che restituisce un AIMessage; StrOutputParser ne estrae il testo.
    """
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )


def build_rag_chain_with_sources(retriever, llm: ChatOllama):
    """Variante della catena che restituisce anche le fonti.

    Usata dall'endpoint /query, che deve restituire sia la risposta sia
    i chunk usati. Restituisce una funzione che, data la domanda, ritorna
    {"answer": str, "docs": list[Document]}.

    Perché una funzione e non una catena pura: ci serve tenere i Document
    dopo che sono stati usati, e farli uscire insieme alla risposta. Un
    fork LCEL con RunnableParallel farebbe girare il retriever due volte
    a meno di caching; così gira una volta sola ed è più leggibile.

    NOTA: è codice sincrono. Da un handler async chiamalo tramite
    run_in_threadpool (vedi api.py), altrimenti blocchi l'event loop.
    """
    # Sotto-catena che genera solo la risposta, a partire da doc già recuperati
    answer_chain = (
        {
            "context": lambda x: format_docs(x["docs"]),
            "question": lambda x: x["question"],
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    def chain_with_sources(question: str) -> dict:
        docs = retriever.invoke(question)
        answer = answer_chain.invoke({"docs": docs, "question": question})
        return {"answer": answer, "docs": docs}

    return chain_with_sources


# ─────────────────────────────────────────────────────────────────────────────
# Interfaccia per uso interattivo (python query.py)
# ─────────────────────────────────────────────────────────────────────────────


def ask_streaming(question: str, chain) -> str:
    """Invoca la catena in modalità streaming, stampando token per token.

    chain.stream() è fornito nativamente da LCEL — non serve implementarlo.
    Ogni elemento del generatore è un frammento di stringa (già estratto
    da StrOutputParser, non un AIMessage).
    """
    full_response = []
    for token in chain.stream(question):
        print(token, end="", flush=True)
        full_response.append(token)
    print()
    return "".join(full_response)


def interactive_loop(chain, qdrant_client: QdrantClient):
    """Sessione interattiva: legge domande dalla tastiera in loop."""
    print("\n+----------------------------------------+")
    print("|   RAG Assistant - Knowledge Base       |")
    print("|   'exit' or Ctrl+C to quit             |")
    print("+----------------------------------------+\n")

    info = qdrant_client.get_collection(cfg.QDRANT_COLLECTION)
    print(f"{info.points_count} chunks indexed in the knowledge base\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "esci"):
            print("Goodbye.")
            break

        print("\nAssistant: ", end="", flush=True)
        ask_streaming(question, chain)
        print()


def main():
    parser = argparse.ArgumentParser(description="Query the RAG knowledge base")
    parser.add_argument("--question", "-q", type=str, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable MultiQueryRetriever INFO logs (shows the expanded queries)",
    )
    args = parser.parse_args()

    if args.debug:
        import logging

        logging.basicConfig()
        logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    retriever = get_retriever(vectorstore, llm)
    chain = build_rag_chain(retriever, llm)

    # Client Qdrant diretto solo per leggere il conteggio dei vettori
    qdrant_client = QdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT)

    if args.question:
        print("\nAssistant: ", end="", flush=True)
        ask_streaming(args.question, chain)
        print()
    else:
        interactive_loop(chain, qdrant_client)


if __name__ == "__main__":
    main()
