# AGENTS.md

## Purpose
- This repo is a local RAG app built with FastAPI, Streamlit, LlamaIndex, Qdrant, and LM Studio.
- It is script-driven and lightweight: no `pyproject.toml`, no `Makefile`, and no repo-owned lint config.
- Prefer small, local changes that preserve the current architecture and Spanish-facing UX.
- Keep backend and frontend behavior aligned: FastAPI serves the API, Streamlit drives the UI.

## Repository Layout
- `main.py`: FastAPI app and API endpoints.
- `frontend.py`: Streamlit UI for chat, uploads, and document management.
- `app/services/rag.py`: RAG initialization, LM Studio integration, Qdrant access, and streaming responses.
- `docker-compose.yml`: default Docker workflow using `host.docker.internal`.
- `docker-compose.wsl.yml`: WSL-specific Docker workflow using `network_mode: host`.
- `scripts/docker-compose-wsl.sh`: helper that loads `.venv` env values and runs WSL compose.
- `start_local.sh`: local non-Docker startup script.
- `requirements.txt`: pinned Python dependencies.
- `README.md`: main operational documentation.

## Agent Rules Present In Repo
- No `.cursor/rules/` directory was found.
- No `.cursorrules` file was found.
- No `.github/copilot-instructions.md` file was found.
- If any of those files are added later, update this file and follow them as higher-priority repo guidance.

## Environment Notes
- LM Studio is expected to expose an OpenAI-compatible API, usually at `http://localhost:1234/v1`.
- The backend reads `OPENAI_API_BASE`, `OPENAI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, and `QDRANT_PATH`.
- The frontend reads `API_URL` and defaults to `http://localhost:8000`.
- The repo tracks a file named `.venv` that stores env vars.
- The local startup path also expects a virtualenv directory named `.venv/` or `venv/`.
- Do not overwrite the tracked `.venv` file while creating a virtual environment.

## Build And Run Commands

### Docker
- Start everything: `docker compose up --build`
- Start in background: `docker compose up --build -d`
- Stop services: `docker compose down`
- Rebuild without cache: `docker compose build --no-cache`
- Follow logs: `docker compose logs -f`

### Docker In WSL
- Start everything: `./scripts/docker-compose-wsl.sh up --build`
- Start in background: `./scripts/docker-compose-wsl.sh up --build -d`
- Stop services: `./scripts/docker-compose-wsl.sh down`
- Rebuild without cache: `docker compose -f docker-compose.wsl.yml build --no-cache`
- Follow logs: `./scripts/docker-compose-wsl.sh logs -f`

### Local Python Workflow
- Start backend + frontend: `bash start_local.sh`
- Stop local services: `pkill -f uvicorn && pkill -f streamlit`
- Backend only: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Frontend only: `streamlit run frontend.py --server.port 8501`
- Backend log: `tail -f backend_local.log`
- Frontend log: `tail -f frontend_local.log`

### Quick Verification
- Health check: `curl http://localhost:8000/health`
- Manual indexing: `curl -X POST http://localhost:8000/api/indexar`
- Inspect LM Studio models: `curl http://localhost:1234/v1/models`

## Lint, Format, Type Check, And Test Commands
- No repo-owned `ruff`, `black`, `flake8`, `pylint`, `mypy`, or `pyright` config is checked in.
- No tracked Python test files are currently present in git.
- Because of that, there is no single canonical lint/test command for this repository today.

### Safe Validation Commands That Work Now
- Syntax-check one file: `python3 -m py_compile main.py`
- Syntax-check touched files: `python3 -m py_compile main.py frontend.py app/services/rag.py`
- Compile all Python files: `python3 -m compileall .`

### If You Add Pytest Tests
- Run all tests: `python3 -m pytest`
- Run one file: `python3 -m pytest tests/test_something.py`
- Run one test function: `python3 -m pytest tests/test_something.py::test_case_name`
- Run by keyword: `python3 -m pytest -k "chat and stream"`
- Stop on first failure: `python3 -m pytest -x`

### If You Add Ruff Or Black Later
- Lint: `python3 -m ruff check .`
- Format: `python3 -m black .`
- Import cleanup: `python3 -m ruff check . --fix`

## Testing Guidance For Agents
- If your change is pure Python logic, at minimum run `python3 -m py_compile` on touched files.
- If your change affects startup, run the relevant service command and verify `/health`.
- If your change affects ingestion, verify upload, index, list, and delete flows manually.
- If your change affects streaming chat, verify the Streamlit chat path or call `/api/chat` with a small request body.
- If you add tests, prefer `pytest` style and include at least one single-test command in your handoff.

## Code Style Guidelines

### Imports
- Group imports as: standard library, third-party, local imports.
- Separate groups with one blank line.
- Prefer one import per line unless multiple names are tightly related and still readable.
- Keep imports at module top unless lazy loading avoids startup cost or a circular import.

### Formatting
- Follow PEP 8 and keep formatting Black-compatible even without a checked-in formatter.
- Use 4 spaces for indentation.
- Avoid trailing whitespace.
- Prefer short functions with clear data flow over deep nesting.
- Keep user-facing Spanish strings consistent with the existing tone.

### Types
- Add type hints for new helpers and non-trivial return values.
- Preserve the existing standard-library `typing` style.
- Use `BaseModel` for structured FastAPI request bodies.
- Add explicit return types on parsing, normalization, and streaming helpers.

### Naming
- Spanish identifiers are acceptable and already common in this repo.
- Use `snake_case` for variables, functions, and helpers.
- Use `PascalCase` for classes and Pydantic models.
- Keep environment variable names uppercase.
- Prefer descriptive names that match behavior, such as `subir_documento` and `obtener_respuesta_rag_stream`.

### FastAPI Conventions
- Keep route handlers small and move non-trivial logic into `app/services/`.
- Return structured JSON with stable keys for success and error cases.
- Prefer `HTTPException` for API failures instead of silent fallbacks.
- Preserve existing endpoint paths unless the user explicitly requests API changes.

### Streamlit Conventions
- Store UI state in `st.session_state`.
- Update session state atomically before `st.rerun()`.
- Prefer clear feedback via `st.success`, `st.warning`, `st.error`, and `st.toast`.
- Handle backend connectivity issues gracefully; do not expose raw stack traces to users.

### Error Handling
- Prefer targeted exceptions over broad `except Exception` when modifying code.
- When a broad catch is unavoidable, return or surface actionable context.
- Do not suppress errors silently.
- Keep messages understandable for both local developers and end users.

### Configuration And Paths
- Read deployment-specific values from environment variables instead of new hardcoded constants.
- Reuse current defaults when extending LM Studio, Qdrant, or API URL configuration.
- Keep persisted local data under `./data` or the configured Qdrant path.
- Do not hardcode machine-specific absolute paths.

### RAG-Service Notes
- `app/services/rag.py` is stateful and initializes shared LLM, embeddings, vector store, and per-session chat engines.
- Be careful with module-level globals like `motores_de_chat`, `client`, and `vector_store`.
- Preserve streaming behavior, `<think>` formatting, and source citation emission unless the task explicitly changes them.

### Comments And Documentation
- Keep comments concise and useful.
- Preserve meaningful Spanish comments that explain architecture or operational intent.
- Update `README.md` when operational commands, ports, or environment expectations change.
- Update this `AGENTS.md` when repository conventions or tooling change.

## Practical Do And Don't
- Do keep changes small and compatible with the current script-first setup.
- Do validate touched Python files with `py_compile` when no stronger automated test exists.
- Do preserve Spanish-language UX text unless the task requests translation.
- Don't assume lint or test tooling exists unless you add and document it.
- Don't rename public endpoints, env vars, or compose services casually.
- Don't overwrite the tracked `.venv` env file while creating a virtual environment.
