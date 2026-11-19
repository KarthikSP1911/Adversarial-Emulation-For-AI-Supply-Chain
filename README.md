# Hashvault

## Commits

Conventional Commits, enforced by commitlint.

Example: `feat: add login flow`

## Setup

```sh
npm install   # installs git hooks via husky
uv sync       # installs Python dependencies
uv run modelscan --help
```

## Running

```sh
uv run modelscan version
uv run modelscan scan <file-or-folder>          # human-readable output
uv run modelscan scan <file-or-folder> --json   # JSON output
uv run modelscan scan <file-or-folder> --out report.json
uv run pytest -q                                # run tests
```

Exit code is `2` if any CRITICAL finding is reported, `0` otherwise.

### Try it on a sample model

A tiny real model (a few MB) with the same file layout as a full model: config, tokenizer and random weights.

```sh
uv run --with huggingface_hub hf download hf-internal-testing/tiny-random-gpt2 --local-dir samples/tiny-gpt2
uv run modelscan scan samples/tiny-gpt2
```

`hf://` targets are not supported yet, so download the model first.

## Project structure

```
src/
└── modelscan/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── models.py
    ├── config.py
    ├── ingest/
    ├── static/
    ├── provenance/
    ├── behaviour/
    ├── verdict/
    ├── report/
    └── rules/
```
