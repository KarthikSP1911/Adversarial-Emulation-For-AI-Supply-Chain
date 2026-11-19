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
