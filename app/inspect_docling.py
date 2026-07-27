"""
inspect_docling.py — Script diagnostico per ispezionare l'output di Docling
============================================================================

SCOPO
Prima di decidere COME spezzare i documenti, bisogna VEDERE che struttura
Markdown produce Docling sui documenti reali. Senza questo dato, scegliere
tra MarkdownHeaderTextSplitter, SemanticChunker o altro è un'ipotesi, non
una decisione informata.

COSA FA
1. Prende N file di esempio da DOCS_PATH (o un file specifico con --file)
2. Li converte con Docling (stesso converter di ingest.py)
3. Applica la stessa pulizia (clean_markdown) già in uso — così vediamo
   esattamente l'input che finisce nello splitter
4. Salva il markdown completo su disco per ispezione manuale riga per riga
5. Stampa un'analisi rapida della struttura: quanti header per livello,
   quante righe di tabella, la gerarchia completa dei titoli trovati con
   la lunghezza del corpo testo sotto ciascuno (per capire quali campi
   rischiano di essere scartati dalle soglie di chunking)

USO (dalla cartella app/):
    python inspect_docling.py                   # 3 file, provando estensioni diverse
    python inspect_docling.py --file nome.docx  # un file specifico
    python inspect_docling.py --sample 6        # più file
    python inspect_docling.py --redact          # non stampa il testo dei titoli
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from docling.document_converter import DocumentConverter

from config import ROOT, cfg

# Riusiamo la logica di ingest.py invece di duplicarla: così quello che
# vediamo qui è garantito identico a quello che fa la pipeline vera.
from ingest import MIN_CHUNK_CHARS, SUPPORTED_EXTENSIONS, clean_markdown

OUTPUT_DIR = ROOT / "data" / "docling_inspection"


def analyze_structure(markdown: str) -> dict:
    """Analizza la struttura del markdown: header, tabelle, e quanto testo
    "corpo" sta sotto ciascun header.

    Non è analisi NLP: è pattern matching sulla sintassi markdown. Ci
    interessano due domande concrete:
    1. quanto è strutturato, dal punto di vista dei titoli, questo documento?
    2. quanti campi rischiano di essere scartati o fusi male perché il loro
       contenuto è più corto delle soglie di chunking?
    """
    lines = markdown.split("\n")

    # Prima passata: trova tutti gli header con la loro posizione (riga)
    headers = []
    for idx, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            headers.append(
                {
                    "level": len(match.group(1)),
                    "title": match.group(2).strip(),
                    "line_idx": idx,
                }
            )

    # Seconda passata: per ogni header, quanti caratteri di "corpo" ci sono
    # fino al prossimo header di QUALSIASI livello — cioè esattamente il
    # testo che finirebbe in un chunk splittando per header.
    for i, h in enumerate(headers):
        start = h["line_idx"] + 1
        end = headers[i + 1]["line_idx"] if i + 1 < len(headers) else len(lines)
        h["body_chars"] = sum(len(line) for line in lines[start:end])
        h["below_min_chunk"] = h["body_chars"] < MIN_CHUNK_CHARS

    table_lines = sum(1 for line in lines if line.count("|") >= 2)
    header_levels = Counter(h["level"] for h in headers)

    return {
        "total_chars": len(markdown),
        "total_lines": len(lines),
        "header_levels": dict(sorted(header_levels.items())),
        "headers": headers,
        "table_lines": table_lines,
    }


def inspect_file(path: Path, converter: DocumentConverter, redact: bool = False):
    print(f"\n{'=' * 70}")
    print(f"FILE: {path.name}")
    print(f"{'=' * 70}")

    try:
        result = converter.convert(str(path))
        raw = result.document.export_to_markdown()
    except Exception as e:
        print(f"  PARSING ERROR: {e}")
        return

    cleaned = clean_markdown(raw)

    # Le statistiche servono a orientarsi; la verità sta nel file salvato,
    # da aprire e leggere per davvero.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{path.stem}.md"
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"  Markdown saved to: {out_path}")

    stats = analyze_structure(cleaned)
    print(f"  Characters: {stats['total_chars']} | Lines: {stats['total_lines']}")
    print(f"  Lines with table syntax: {stats['table_lines']}")

    if not stats["header_levels"]:
        print("  NO HEADINGS FOUND.")
        print("      Possible causes: the document has no real heading hierarchy,")
        print("      or Docling could not infer the layout (scanned PDF, unusual")
        print("      formatting). This matters: on these files")
        print("      MarkdownHeaderTextSplitter would have nothing to work with.")
        return

    print(f"  Headings by level: {stats['header_levels']}")
    n_below = sum(1 for h in stats["headers"] if h["below_min_chunk"])
    print(
        f"  Fields with body < MIN_CHUNK_CHARS ({MIN_CHUNK_CHARS} chars): "
        f"{n_below}/{len(stats['headers'])}"
    )

    if redact:
        # Modalità riservata: mostriamo SOLO la forma (livello, lunghezza
        # titolo, lunghezza corpo), mai il testo vero. Utile per condividere
        # l'analisi senza esporre il contenuto dei documenti.
        print("  Full structure (level, title length, body length), redacted:")
        for h in stats["headers"]:
            indent = "  " * (h["level"] - 1)
            flag = "  <- below threshold" if h["below_min_chunk"] else ""
            print(f"    {indent}H{h['level']} [title:{len(h['title'])}c] [body:{h['body_chars']}c]{flag}")
    else:
        print("  Full structure (real titles + body length):")
        for h in stats["headers"]:
            indent = "  " * (h["level"] - 1)
            flag = "  <- below threshold" if h["below_min_chunk"] else ""
            print(f"    {indent}{'#' * h['level']} {h['title']}  [body:{h['body_chars']}c]{flag}")


def main():
    parser = argparse.ArgumentParser(description="Inspect Docling's Markdown output")
    parser.add_argument(
        "--file", type=str, default=None, help="Analyse this file only (relative to DOCS_PATH)"
    )
    parser.add_argument(
        "--sample", type=int, default=3, help="How many files to sample (default: 3)"
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Do not print heading text, only level and length",
    )
    args = parser.parse_args()

    docs_path = cfg.DOCS_PATH
    if not docs_path.exists():
        print(f"Documents folder not found: {docs_path}")
        return

    converter = DocumentConverter()

    if args.file:
        files = [docs_path / args.file]
    else:
        all_files = [
            f for f in docs_path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        # Campioniamo estensioni DIVERSE quando possibile: Docling si
        # comporta in modo diverso per pdf/docx/html/pptx/md.
        by_ext: dict[str, list[Path]] = {}
        for f in all_files:
            by_ext.setdefault(f.suffix.lower(), []).append(f)

        files = []
        exts = list(by_ext.keys())
        i = 0
        while len(files) < args.sample and any(by_ext.values()) and exts:
            ext = exts[i % len(exts)]
            if by_ext[ext]:
                files.append(by_ext[ext].pop())
            i += 1

    if not files:
        print("No files found to analyse.")
        return

    print(f"Analysing {len(files)} files: {[f.name for f in files]}")

    for f in files:
        inspect_file(f, converter, redact=args.redact)

    print(f"\n\nDone. Full markdown files are in: {OUTPUT_DIR}")
    print("Open them in an editor to see the real structure line by line -")
    print("in particular where headings fall relative to the text, and how")
    print("Docling renders tables.")


if __name__ == "__main__":
    main()
