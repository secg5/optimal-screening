# Repository Guidelines

## Project Structure & Module Organization
- Source lives under `src/optimal_screening/` (create if missing). Keep modules small and focused.
- Tests live in `tests/` mirroring package structure (e.g., `tests/test_utils.py`).
- Utility scripts go in `scripts/` (executable, with a brief usage comment).
- Tooling and config are centralized in `pyproject.toml` (ruff, formatting). Assets or datasets belong under `assets/` or `data/` if introduced.

## Build, Test, and Development Commands
- Create/activate env: `python -m venv .venv && source .venv/bin/activate`.
- Install dev tooling (local): `pip install -U ruff pytest`.
- Lint: `ruff check .` (fails on style/errors).
- Format: `ruff format .` (applies formatting rules).
- Run tests: `pytest -q` (use `-k` to filter). Example: `pytest -q tests/test_utils.py::test_case`.
- Run package (if applicable): `python -m optimal_screening` or call the module you add an entry point for.

## Coding Style & Naming Conventions
- Python 3.11. Indent with 4 spaces; include type hints for public APIs.
- Ruff line length is `119`. Import sorting via Ruff with `lines-after-imports = 2`.
- Allowed ignores: `E501` and common complexity/stylistic rules per `pyproject.toml`.
- `__init__.py` may ignore unused imports (`F401`) when used for package exports.
- Naming: `snake_case` for modules/functions, `CapWords` for classes, `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- Use `pytest`. Name files `test_*.py`; name tests `test_*`.
- Keep unit tests close to code concerns; use fixtures/fakes over network or disk.
- Target high coverage (≥80%) for new/changed code; include edge cases and failure paths.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` (optional scope: `feat(api): ...`).
- Keep commits focused and descriptive; reference issues (`Closes #123`).
- PRs must include: what/why, how tested, any follow‑ups. Attach logs or screenshots when helpful.
- PR checklist: `ruff check .` passes, `ruff format .` applied, tests green.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` and keep it ignored (see `.gitignore`).
- Keep dependencies minimal and pinned when introduced; avoid storing PII in samples/tests.

## Agent‑Specific Instructions
- Follow these conventions and existing structure; prefer minimal, surgical diffs.
- Do not reformat unrelated files. Update docs/tests alongside code changes.
- If conventions change, update this AGENTS.md in the same PR.

