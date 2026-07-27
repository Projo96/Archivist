"""
ingest.py — Pipeline di ingestion documenti (chunking header-aware)
====================================================================

Legge tutti i file da DOCS_PATH, li processa con Docling, li divide in
chunk RISPETTANDO la struttura a header del documento (invece che a
caratteri fissi), calcola gli embedding e li carica in Qdrant.

PERCHÉ IL CHUNKING È HEADER-AWARE
I documenti di riferimento di questo progetto seguono un template molto
regolare: un titolo di procedura (H3) seguito da una ventina di campi
(H4), alcuni dei quali hanno a loro volta sotto-campi (H5) — per esempio
i passi operativi, o le righe di approvazione. Le lunghezze dei campi
variano di due ordini di grandezza: da pochi caratteri (un campo di
firma vuoto) a diverse migliaia (un campo descrittivo lungo).

Uno splitter a caratteri fissi ignora completamente questi confini:
mischia più campi nello stesso chunk, o ne taglia uno a metà. E un
filtro unico "scarta i chunk sotto N caratteri" tiene o butta i campi
corti in modo arbitrario, non in base al loro valore informativo.
Da qui la classificazione a quattro vie implementata più sotto.

Uso:
    python ingest.py                   # indicizza tutti i documenti nuovi/modificati
    python ingest.py --reset           # svuota la collezione e reindicizza tutto
    python ingest.py --file mio.docx   # indicizza un solo file
"""

import argparse
import hashlib
import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from config import cfg

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".pptx", ".md", ".txt"}

# ─────────────────────────────────────────────────────────────────
# Soglie di chunking — tarate sui documenti reali, non un singolo
# MIN_CHUNK_CHARS onnicomprensivo. La logica: un campo del template
# va trattato diversamente a seconda di QUANTO contenuto ha, non
# solo se supera una soglia unica.
# ─────────────────────────────────────────────────────────────────

# Sotto questa soglia, il corpo di un campo è quasi certamente un
# placeholder vuoto o un valore di firma/approvazione (nei documenti
# analizzati questi campi hanno un corpo di 2-3 caratteri).
# Scartato SEMPRE, indipendentemente dal nome del campo — non serve
# sapere come si chiama per riconoscere che non porta informazione.
NEAR_EMPTY_CHARS = 15

# Sotto questa soglia (ma sopra NEAR_EMPTY_CHARS) il campo ha un
# valore reale ma troppo corto per essere un chunk a sé stante
# (es. una singola parola come valore di un attributo). Questi campi
# vengono raggruppati in UN chunk "Attributes" per procedura, invece
# di finire isolati e semanticamente ambigui.
MIN_CHUNK_CHARS = 150

# Campi amministrativi da scartare esplicitamente per nome: non hanno
# mai valore per il retrieval semantico. Il confronto avviene sul nome
# ripulito dalla numerazione prodotta da Docling (vedi strip_numbering).
# Adatta questa lista al tuo template.
ADMIN_FIELDS_BLACKLIST = {
    "procedure reference",
    "procedure version",
    "procedure status",
}

# Livelli di header su cui splittare. Docling produce H3 per il titolo
# della procedura, H4 per i campi del template, H5 per i sotto-campi.
# Se i tuoi documenti hanno una gerarchia diversa, è qui che si aggiusta.
HEADERS_TO_SPLIT_ON = [
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
]


def file_hash(path: Path) -> str:
    """SHA256 del file — usato per evitare reindicizzazioni inutili."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_markdown(text: str) -> str:
    """Pulizia leggera del Markdown prodotto da Docling.

    Rimuove:
    - commenti HTML come <!-- image --> che non portano informazione
    - righe di tabella completamente vuote (solo pipe e spazi)
    - sequenze di più di 2 righe vuote consecutive

    NOTA: le tabelle "piene" (con contenuto) non vengono toccate qui —
    restano sintassi Markdown grezza (pipe, trattini) dentro al corpo
    del campo che le contiene. È una limitazione nota: non facciamo
    ancora una conversione tabella → testo naturale. Se il retrieval
    su contenuti tabellari risulta scarso, è il prossimo punto da
    affrontare (vedi ROADMAP nel README).
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"^\|[\s|]+\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_numbering(title: str) -> str:
    """Rimuove il prefisso numerico che Docling aggiunge ai titoli
    (es. '1.1.1.4 Role' → 'Role'). Serve per confrontare i nomi dei
    campi in modo pulito, indipendente dalla numerazione di sezione
    che cambia da documento a documento."""
    return re.sub(r"^[\d.]+\s*", "", title).strip()


def merge_wrapped_headers(text: str) -> str:
    """Corregge un artefatto di conversione: alcuni titoli nel documento
    Word originale hanno un a-capo manuale al loro interno (un H5 che
    nel .docx era scritto su due righe). Docling esporta l'a-capo così
    com'è, ma la regex che riconosce gli header lavora riga per riga —
    vede solo la prima metà del titolo, e la seconda metà finisce
    trattata come inizio del corpo (titolo troncato, corpo "sporco"
    con un frammento di titolo davanti).

    Euristica di rilevamento: un titolo genuino con virgolette tipo
    'Step "Nome step"' ha sempre un numero PARI di doppi apici (aperta
    e chiusa sulla stessa riga). Se un header ha un numero DISPARI di
    doppi apici, sospettiamo che il titolo continui sulla riga
    successiva e la fondiamo — finché le virgolette non tornano pari.
    """
    lines = text.split("\n")
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_header = re.match(r"^#{1,6}\s+.+$", line.strip())
        if is_header and line.count('"') % 2 == 1:
            combined = line.rstrip()
            j = i + 1
            # Continua a fondere righe finché non troviamo la virgoletta
            # di chiusura, o finché non incontriamo una riga vuota /
            # un altro header (segnali che il titolo non stava
            # davvero continuando).
            while (
                j < len(lines)
                and combined.count('"') % 2 == 1
                and lines[j].strip()
                and not re.match(r"^#{1,6}\s+", lines[j].strip())
            ):
                combined += " " + lines[j].strip()
                j += 1
            merged.append(combined)
            i = j
        else:
            merged.append(line)
            i += 1
    return "\n".join(merged)


def get_qdrant() -> QdrantClient:
    """Client Qdrant, con creazione della collection se non esiste.

    Crea anche due indici sui campi di payload usati come filtro
    (file_hash e source_path). Senza indice Qdrant filtra comunque,
    ma con una scansione completa: su collection piccole non si nota,
    su collection grandi sì. Gli indici sono idempotenti, quindi
    ricrearli a ogni avvio è innocuo.
    """
    client = QdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT)
    existing = [c.name for c in client.get_collections().collections]
    if cfg.QDRANT_COLLECTION not in existing:
        logger.info(f"Creating collection '{cfg.QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=cfg.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=cfg.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    for field in ("metadata.file_hash", "metadata.source_path"):
        try:
            client.create_payload_index(
                collection_name=cfg.QDRANT_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Già esistente: non è un errore, è lo stato normale
            # a partire dal secondo avvio.
            pass

    return client


def get_embedder() -> OllamaEmbeddings:
    """Client di embedding.

    Usiamo lo stesso wrapper di query.py (OllamaEmbeddings) invece del
    client OpenAI-compatible: stesso modello, stesso endpoint Ollama,
    stessi vettori — ma una dipendenza in meno e la garanzia che
    ingestion e query usino esattamente la stessa configurazione.
    """
    return OllamaEmbeddings(
        base_url=cfg.OLLAMA_BASE_URL,
        model=cfg.EMBEDDING_MODEL,
    )


def parse_document(path: Path, converter: DocumentConverter) -> str:
    logger.info(f"  Parsing: {path.name}")
    result = converter.convert(str(path))
    raw = result.document.export_to_markdown()
    raw = merge_wrapped_headers(raw)
    cleaned = clean_markdown(raw)
    logger.debug(f"  {len(raw)} -> {len(cleaned)} characters after cleanup")
    return cleaned


def split_into_chunks(markdown: str) -> list[dict]:
    """Split header-aware: rispetta la struttura H3/H4/H5 invece di
    tagliare a caratteri fissi.

    Strategia:
    1. MarkdownHeaderTextSplitter spacca il markdown rispettando gli
       header, producendo un "documento" per ogni foglia di testo con
       la sua catena di header (h3/h4/h5) come metadata.
    2. Ogni foglia viene classificata in base al nome del campo (h4) e
       alla lunghezza del suo corpo:
         - campo in ADMIN_FIELDS_BLACKLIST  → scartato
         - corpo < NEAR_EMPTY_CHARS         → scartato (placeholder/firma)
         - corpo < MIN_CHUNK_CHARS          → accumulato nel chunk "Attributes"
         - corpo sostanzioso                → chunk a sé, con contesto anteposto
    3. Se un chunk "sostanzioso" supera CHUNK_SIZE, viene ri-splittato
       con RecursiveCharacterTextSplitter — ma ogni pezzo porta
       comunque il contesto del campo di origine.

    Ritorna una lista di dict {"text", "section_title", "parent_section"},
    non semplici stringhe: portiamo con noi i metadata strutturali da
    salvare poi nel payload Qdrant.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        # Il testo dell'header non serve nel corpo: lo ricostruiamo
        # noi come "context path" all'inizio del chunk.
        strip_headers=True,
    )
    leaf_docs = md_splitter.split_text(markdown)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    procedure_title = None
    attributes_buffer = []  # righe "Campo: valore" da fondere in un chunk unico
    result_chunks = []

    for doc in leaf_docs:
        meta = doc.metadata
        h3 = meta.get("h3")
        h4 = meta.get("h4")
        h5 = meta.get("h5")
        body = doc.page_content.strip()

        if h3 and procedure_title is None:
            procedure_title = strip_numbering(h3)

        field_name = strip_numbering(h5 or h4 or "")

        # 1. Campo amministrativo esplicito → scartato per nome
        if h4 and strip_numbering(h4).lower() in ADMIN_FIELDS_BLACKLIST:
            logger.debug(f"  Dropping administrative field: {h4}")
            continue

        # 2. Corpo quasi vuoto → placeholder/firma, scartato a prescindere dal nome
        if len(body) < NEAR_EMPTY_CHARS:
            logger.debug(
                f"  Dropping near-empty field ({len(body)} chars): {field_name or '(untitled)'}"
            )
            continue

        # 3. Corpo corto ma reale → confluisce nel chunk "Attributes"
        if len(body) < MIN_CHUNK_CHARS:
            attributes_buffer.append(f"{field_name}: {body}")
            continue

        # 4. Corpo sostanzioso → chunk a sé.
        # Il contesto (procedura > campo > sotto-campo) viene anteposto
        # al testo: aiuta l'embedding a "sapere" di cosa parla anche
        # un chunk che, isolato, sarebbe ambiguo — es. uno step senza
        # sapere a quale procedura appartiene.
        path_parts = [procedure_title]
        if h4:
            path_parts.append(strip_numbering(h4))
        if h5:
            path_parts.append(strip_numbering(h5))
        context_path = " > ".join(p for p in path_parts if p)
        full_text = f"[{context_path}]\n{body}"

        parent_section = strip_numbering(h4) if h5 else None

        if len(full_text) <= cfg.CHUNK_SIZE:
            result_chunks.append(
                {
                    "text": full_text,
                    "section_title": field_name,
                    "parent_section": parent_section,
                }
            )
        else:
            # Campo lungo oltre soglia: ri-split a caratteri.
            # IMPORTANTE: splittiamo il CORPO, non full_text (che ha già
            # il contesto anteposto) — altrimenti solo il primo frammento
            # porterebbe il contesto "[Procedura > Campo]" e i successivi
            # partirebbero nel mezzo del contenuto, orfani, senza sapere
            # a quale procedura/campo appartengono.
            for part in char_splitter.split_text(body):
                result_chunks.append(
                    {
                        "text": f"[{context_path}]\n{part}",
                        "section_title": field_name,
                        "parent_section": parent_section,
                    }
                )

    # Chunk "Attributes": tutti i campi corti-ma-utili fusi in uno solo
    if attributes_buffer:
        header = f"[{procedure_title} > Attributes]" if procedure_title else "[Attributes]"
        result_chunks.insert(
            0,
            {
                "text": header + "\n" + "\n".join(attributes_buffer),
                "section_title": "Attributes",
                "parent_section": None,
            },
        )

    return result_chunks


def embed_chunks(texts: list[str], embedder: OllamaEmbeddings) -> list[list[float]]:
    """Calcola gli embedding di una lista di testi."""
    return embedder.embed_documents(texts)


def already_indexed(qdrant: QdrantClient, hash_value: str) -> bool:
    """Esiste già almeno un chunk con questo hash di file?

    L'hash è del contenuto, non del nome: un file rinominato ma identico
    risulta già indicizzato, un file modificato no.
    """
    results = qdrant.scroll(
        collection_name=cfg.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    # dot-notation: campo annidato sotto "metadata"
                    key="metadata.file_hash",
                    match=MatchValue(value=hash_value),
                )
            ]
        ),
        limit=1,
    )
    return len(results[0]) > 0


def delete_stale_chunks(qdrant: QdrantClient, path: Path) -> None:
    """Elimina i chunk di una versione PRECEDENTE dello stesso file.

    Senza questo passaggio, modificare un documento significa: l'hash
    cambia → already_indexed() dice "no" → i nuovi chunk vengono
    inseriti con ID nuovi (l'ID deriva dall'hash) → i chunk vecchi
    restano in Qdrant per sempre. Risultato: il retriever pesca da due
    versioni dello stesso documento contemporaneamente, e le risposte
    citano contenuti non più validi.

    Filtriamo su source_path (percorso completo) e non su source (solo
    nome file), così due file omonimi in sottocartelle diverse non si
    cancellano a vicenda.
    """
    qdrant.delete(
        collection_name=cfg.QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_path",
                        match=MatchValue(value=str(path)),
                    )
                ]
            )
        ),
        wait=True,
    )


def index_file(
    path: Path,
    qdrant: QdrantClient,
    embedder: OllamaEmbeddings,
    converter: DocumentConverter | None = None,
) -> int:
    hash_value = file_hash(path)

    if already_indexed(qdrant, hash_value):
        logger.info(f"  Already indexed, skipping: {path.name}")
        return 0

    text = parse_document(path, converter or DocumentConverter())
    if not text.strip():
        logger.warning(f"  No text extracted from {path.name}, skipping.")
        return 0

    chunks = split_into_chunks(text)
    if not chunks:
        logger.warning(f"  No usable chunks in {path.name}, skipping.")
        return 0

    logger.info(f"  {len(chunks)} chunks to index (header-aware)")

    texts = [c["text"] for c in chunks]

    # Embedding in batch: una sola chiamata HTTP ogni 32 chunk
    # invece di una per chunk.
    BATCH = 32
    all_embeddings = []
    for i in range(0, len(texts), BATCH):
        all_embeddings.extend(embed_chunks(texts[i : i + BATCH], embedder))

    # Ripulisci le versioni precedenti dello stesso file PRIMA di
    # inserire le nuove (vedi delete_stale_chunks).
    delete_stale_chunks(qdrant, path)

    # Costruisci i punti Qdrant.
    #
    # IMPORTANTE: i metadata sono annidati sotto la chiave "metadata",
    # non piatti a livello radice. Motivo: QdrantVectorStore di
    # langchain_qdrant, quando costruisce i Document per il retriever,
    # legge il payload alla chiave indicata da metadata_payload_key —
    # e il default di quel parametro è letteralmente la stringa
    # "metadata" (verificato nel codice sorgente della libreria, non
    # per assunzione). Con un payload piatto quella lookup restituisce
    # sempre {}, quindi ogni Document arriva con metadata VUOTO al
    # retriever, e le fonti nelle risposte risultano vuote.
    # "text" resta a livello radice perché content_payload_key="text"
    # punta lì esplicitamente.
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        point_id = int(hashlib.sha256(f"{hash_value}_{i}".encode()).hexdigest()[:15], 16)
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "metadata": {
                        "source": path.name,
                        "source_path": str(path),
                        "file_hash": hash_value,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "extension": path.suffix,
                        "section_title": chunk["section_title"],
                        "parent_section": chunk["parent_section"],
                    },
                },
            )
        )

    qdrant.upsert(collection_name=cfg.QDRANT_COLLECTION, points=points)
    logger.success(f"  {len(points)} chunks inserted")
    return len(points)


def main():
    parser = argparse.ArgumentParser(description="Index documents into Qdrant")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe the collection before indexing"
    )
    parser.add_argument("--file", type=str, default=None, help="Index this file only")
    args = parser.parse_args()

    logger.info("=== RAG Ingestion Pipeline (header-aware chunking) ===")
    logger.info(f"Documents:  {cfg.DOCS_PATH}")
    logger.info(f"Collection: {cfg.QDRANT_COLLECTION}")

    qdrant = get_qdrant()
    embedder = get_embedder()
    converter = DocumentConverter()

    if args.reset:
        logger.warning("Reset: deleting the collection...")
        qdrant.delete_collection(cfg.QDRANT_COLLECTION)
        qdrant = get_qdrant()

    docs_path = cfg.DOCS_PATH
    if not docs_path.exists():
        logger.error(f"Folder not found: {docs_path}")
        logger.error("Check DOCS_PATH in your .env file")
        return

    if args.file:
        files = [docs_path / args.file]
    else:
        files = [
            f
            for f in docs_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    if not files:
        logger.warning(f"No documents found in {docs_path}")
        return

    logger.info(f"Found {len(files)} files")

    total_chunks = 0
    errors = []
    for file_path in tqdm(files, desc="Indexing"):
        try:
            total_chunks += index_file(file_path, qdrant, embedder, converter)
        except Exception as e:
            logger.error(f"Error on {file_path.name}: {e}")
            errors.append((file_path.name, str(e)))

    logger.info("=== Done ===")
    logger.info(f"Chunks inserted: {total_chunks}")
    info = qdrant.get_collection(cfg.QDRANT_COLLECTION)
    logger.info(f"Total vectors in Qdrant: {info.points_count}")
    if errors:
        logger.warning(f"Errors: {len(errors)} files")
        for name, err in errors:
            logger.warning(f"  {name}: {err}")


if __name__ == "__main__":
    main()
