# Contributing

## Setup

```sh
npm install   # installs husky git hooks
```

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), enforced by commitlint on every commit and on PRs.

```
type(scope): short summary
```

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Scope is optional.
- Header up to 72 characters.
- Example: `feat: add login flow`

## Formatting

Pre-commit runs lint-staged (Prettier). Run `npm run format` to format everything.

## Security

Never commit or print secrets. `.env.example` lists names only.

## Pull requests

Fill in the PR template and keep each PR focused.
