"""
config.py — Configurazione centralizzata
=========================================

Legge le variabili da .env e le rende disponibili come oggetto `cfg`.
Tutti gli altri moduli importano da qui, mai da os.environ direttamente:
un solo punto in cui capire "da dove arriva questo valore".

RICERCA DEL .env
Il file .env vive nella root del repository, non accanto ai sorgenti.
Motivo: è un file di configurazione dell'installazione, non del codice,
e in Docker viene iniettato da `env_file:` senza esistere nel container.
Per questo cerchiamo il .env in più posizioni candidate e non falliamo
se non lo troviamo — in quel caso valgono le variabili d'ambiente reali
(che è esattamente lo scenario Docker).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Root del repository: app/config.py → app/ → <repo>
ROOT = Path(__file__).resolve().parent.parent

# Ordine di ricerca: root del repo, poi la cartella dei sorgenti,
# poi la working directory corrente. Il primo trovato vince.
for _candidate in (ROOT / ".env", Path(__file__).parent / ".env", Path.cwd() / ".env"):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break


def _resolve_docs_path(raw: str) -> Path:
    """Un DOCS_PATH relativo va inteso rispetto alla root del repo,
    non alla working directory: così `python app/ingest.py` e
    `cd app && python ingest.py` puntano alla stessa cartella."""
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


class Config:
    # ── Ollama ───────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "phi3:mini")

    # ── Qdrant ───────────────────────────────────────────────────────────
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")

    # ── Documenti ────────────────────────────────────────────────────────
    DOCS_PATH: Path = _resolve_docs_path(os.getenv("DOCS_PATH", "data/docs"))

    # ── Chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "300"))

    # Dimensione dei vettori del modello di embedding.
    # nomic-embed-text produce 768 dimensioni: se cambi EMBEDDING_MODEL
    # devi aggiornare anche questo valore E reindicizzare da zero
    # (una collection Qdrant ha una dimensione fissa, non convertibile).
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

    # Numero di chunk recuperati e passati al modello come contesto.
    TOP_K: int = int(os.getenv("TOP_K", "6"))

    # ── Query expansion (MultiQueryRetriever) ────────────────────────────
    # Prima della similarity search, genera N varianti della domanda con
    # l'LLM e cerca per ciascuna: può migliorare il recall, ma aggiunge
    # chiamate sequenziali a Ollama PRIMA che parta il retrieval vero e
    # proprio. Con un modello piccolo il costo di latenza è significativo.
    # false = una singola similarity search diretta, molto più veloce.
    ENABLE_QUERY_EXPANSION: bool = os.getenv("ENABLE_QUERY_EXPANSION", "false").lower() == "true"

    # ── Parametri di generazione ─────────────────────────────────────────
    # Esposti in .env perché sono i primi da toccare quando si cambia
    # modello: un modello più grande regge un num_ctx più alto.
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "4096"))
    LLM_NUM_PREDICT: int = int(os.getenv("LLM_NUM_PREDICT", "2048"))
    LLM_REPEAT_PENALTY: float = float(os.getenv("LLM_REPEAT_PENALTY", "1.15"))
    LLM_REPEAT_LAST_N: int = int(os.getenv("LLM_REPEAT_LAST_N", "256"))


cfg = Config()
