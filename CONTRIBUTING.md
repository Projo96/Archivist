# Contributing

Personal project in alpha. Pull requests are welcome, but there's no
guarantee on review times.

## Development environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Conventions

- **Source comments are in Italian.** The project started as a learning
  exercise and the comments are the notes that came with it. Everything
  user-facing — README, docs, CLI output, error messages — is in English.
  New comments may be in either language; consistency within a file wins.
- Comments explain the *why*, not the *what*. The code already says what
  it does; the comment is for whoever wonders, six months later, why it
  was done this way.
- No `os.environ` outside `config.py`: one place to find out where a
  value comes from.
- Lines up to 100 characters.
- Before opening a PR: `make lint`.

## What NOT to commit

- The `.env` file (it contains local paths)
- Anything under `data/`: documents, the Qdrant index, Open WebUI state
- Real document names, identifier codes or excerpts of internal content
  — not in examples, comments or prompts either

## Before opening an issue

Include: operating system, Ollama version, model in use, the output of
`curl -s localhost:8000/health`, and the relevant part of
`docker compose logs rag-api`.
