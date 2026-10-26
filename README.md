# Repo Intel — AI GitHub Repository Intelligence & Code Review Platform

Enter one GitHub repository URL and get a scored review: code quality, security, performance,
best practices, documentation, architecture and technology detection. Every finding carries
**file, line, evidence, reason and recommendation**.

```
Browser ─► Django (auth, dashboard, history, reports) ─► PostgreSQL
              │  X-Internal-Token
              ▼
          FastAPI AI service ─► GitHub API (selected repository archive)
              │
              ▼
          LangGraph workflow ─► LangChain prompts + structured output ─► Groq
              │                          └──► Chroma (RAG over your uploaded guidelines)
              ▼
          report JSON  ─► Django stores it in PostgreSQL
```

## Quick start (Docker)

```bash
cp .env.example .env        # then edit: GROQ_API_KEY, AI_SERVICE_TOKEN, DJANGO_SECRET_KEY
docker compose up --build
docker compose exec web python manage.py migrate
```
Open http://localhost:8000, register, and choose **Analyze**.
FastAPI docs: http://localhost:8001/docs

* `GROQ_API_KEY` (free at console.groq.com) enables the AI reviewers. `GROQ_MODEL` defaults to
  `openai/gpt-oss-120b`; check Groq's model list if you change it. Without a key the app still runs
  with rule-based checks only and says so in the report. `LLM_MODE=mock` runs the whole pipeline with a fake LLM.
  The report note reflects whether an LLM was actually used for that run; restart the AI service after changing `.env`.
  To limit Groq usage, the default scan reviews at most 10 source files and sends one combined code-review request per run, covering one representative chunk per selected source file and every indexed guideline passage uploaded by that user. Large guideline collections or repositories may exceed the model's request limits; in that case the report warns that the AI review failed and does not claim the guidelines were used.
  Architecture and summary use local dependency/rule-based fallbacks, while static checks always run.
  If Groq rate-limits the combined review, the report shows one brief note and keeps its rule-based results.
  When running locally, the AI service reads the repository-root `.env`; its values take precedence over stale shell variables.
  Check `http://localhost:8001/health` to confirm the active model and request limits after restarting the service.
* Public repository analysis requires only a URL such as `https://github.com/owner/repository`. Repository visibility is checked before unauthenticated analysis; if GitHub reports that the repository is private or inaccessible without authorization, the app redirects to GitHub OAuth instead of starting an unauthenticated job. Create a GitHub OAuth App and set `GITHUB_OAUTH_CLIENT_ID`, `GITHUB_OAUTH_CLIENT_SECRET`, and `GITHUB_OAUTH_REDIRECT_URI` in the root `.env`. Both Django and the AI service need these settings; restart both services after editing `.env`. Use `http://localhost:8000/settings/github/callback/` as the local callback URL. Choose **Private repository**, then authorize with a GitHub account that can access the selected repository. GitHub may skip showing its consent page when that account has already authorized the app, but the OAuth callback still runs. The access token is held in memory for that analysis only and the OAuth grant is revoked after normal completion or a handled failure; credentials are not stored in the database. When developing locally, run the AI service with `uvicorn app.main:app --port 8001 --reload` so code changes are loaded automatically. If the AI service is forcibly terminated before cleanup, revoke Repo Intel's access in GitHub Settings → Applications.
* Account registration and forgotten-password recovery use six-digit email verification codes that expire after 10 minutes. Configure SMTP with `EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD`, `EMAIL_USE_TLS` (or `EMAIL_USE_SSL`), and `DEFAULT_FROM_EMAIL` in the root `.env`; see `.env.example`. Restart Django after changing email settings. Do not commit real SMTP credentials.
* Create an admin user: `docker compose exec web python manage.py createsuperuser` → http://localhost:8000/admin/

## Run without Docker

```bash
# terminal 1 — AI service
cd ai_service && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
set -a; source ../.env; set +a
uvicorn app.main:app --port 8001 --reload

# terminal 2 — Django (SQLite unless POSTGRES_DB is set)
cd web && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
set -a; source ../.env; set +a
export AI_SERVICE_URL=http://localhost:8001
python manage.py migrate && python manage.py runserver
```

## How the review works (and why recommendations are trustworthy)

1. **Fetch** — one zipball download via the GitHub API; `node_modules`, build output, lock files and binaries are
   skipped, real `.env` files are listed but never read.
2. **Detect technology** — languages by bytes, frameworks/databases/cloud/DevOps from dependency files, Dockerfiles, CI, etc.
3. **Select & split** — the top N source files (ranked by importance heuristics) are split with LangChain's
   language-aware splitter; lines are numbered so the model can cite them. **Secrets are masked before anything reaches the LLM.**
4. **Parallel reviewers (LangGraph fan-out)** — quality, security, performance, documentation.
   Each combines **deterministic rules** (regex secret scanner, `ast`-based complexity checks, risky-pattern rules) with an
   **LLM reviewer** returning Pydantic-validated structured findings.
5. **Evidence verification** — an LLM finding is kept only if the code it quotes really exists in the file; its line number
   is corrected from the match. Hallucinated findings are dropped.
6. **Best practices + RAG (join node)** — all indexed passages from your uploaded guidelines (PDF/txt/md) are supplied
   to one combined LLM review of a representative chunk from each selected source file. The LLM checks the passages
   against the code and cites reported violations to their source document ("Team guide, p.3").
7. **Architecture → Final reviewer** — dedupes findings, computes scores from severity × confidence, and writes the
   executive summary and top-5 recommendations from the actual findings.

LangGraph: `fetch_repository → (error? END) → detect_technology → collect_files → [quality | security | performance | documentation]
→ best_practices → architecture → final_reviewer → generate_report` (see `ai_service/app/graph.py`).

## Tests

```bash
cd ai_service && pip install -r requirements-dev.txt && python -m pytest -q     # 45 tests, no network/API key needed
cd web && python manage.py test                                                  # 35 tests
```

## Limits / honest notes

* A static review aid, not a penetration test; AI findings are suggestions. Regex secret detection can miss or over-flag.
* Only the highest-priority files/chunks get LLM review (`MAX_SOURCE_FILES`, `MAX_CHUNKS_PER_REVIEWER`) to stay within Groq free-tier limits; rule checks and the file tree cover the rest.
* Jobs run in the AI service's memory (single worker); if it restarts mid-analysis the report is marked failed and can be re-run.
* Scanned (image-only) PDFs can't be indexed without OCR. The first guideline upload downloads the ~80 MB embedding model (set `EMBEDDING_BACKEND=hash` to avoid it).
* Next steps: GitHub App integration with narrower repository permissions, Celery/Redis queue, Semgrep/Bandit integration, PDF export.

## Project structure

```text
repo-intel/
|-- ai_service/                         # FastAPI repository-analysis service
|   |-- app/
|   |   |-- __init__.py
|   |   |-- config.py                    # Environment-based service settings
|   |   |-- github_client.py             # GitHub repository access and archive handling
|   |   |-- graph.py                     # LangGraph analysis workflow
|   |   |-- jobs.py                      # In-memory background analysis jobs
|   |   |-- llm.py                       # Groq integration and mock LLM
|   |   |-- main.py                      # FastAPI routes and service entry point
|   |   |-- rag.py                       # Guideline indexing and Chroma retrieval
|   |   |-- reviewers.py                 # Chunking, review prompts, evidence checks, scoring
|   |   |-- scanners.py                  # Secret redaction and scanning helpers
|   |   |-- schemas.py                   # Pydantic request, finding, and report schemas
|   |   |-- static_checks.py              # Deterministic source-code checks
|   |   `-- tech_detect.py                # Language, framework, and technology detection
|   |-- tests/
|   |   |-- conftest.py                   # Shared AI-service test fixtures
|   |   |-- test_graph_api.py             # Workflow, API, and guideline tests
|   |   `-- test_units.py                 # Unit tests for analysis helpers
|   |-- Dockerfile
|   |-- requirements.txt
|   `-- requirements-dev.txt
|-- web/                                  # Django web application
|   |-- accounts/                         # Registration, authentication, and user settings
|   |   |-- migrations/
|   |   |   |-- __init__.py
|   |   |   |-- 0001_initial.py
|   |   |   `-- 0002_delete_githubconnection.py
|   |   |-- __init__.py
|   |   |-- admin.py
|   |   |-- apps.py
|   |   |-- forms.py
|   |   |-- models.py
|   |   |-- tests.py
|   |   |-- urls.py
|   |   `-- views.py
|   |-- config/                           # Django project configuration
|   |   |-- __init__.py
|   |   |-- asgi.py
|   |   |-- settings.py
|   |   |-- urls.py
|   |   `-- wsgi.py
|   |-- reviews/                          # Analysis, report, history, and guideline features
|   |   |-- migrations/
|   |   |   |-- __init__.py
|   |   |   |-- 0001_initial.py
|   |   |   |-- 0002_analysis_includes_private_analysis_is_profile.py
|   |   |   `-- 0003_analysis_target_repo_name.py
|   |   |-- templatetags/
|   |   |   |-- __init__.py
|   |   |   `-- review_extras.py
|   |   |-- __init__.py
|   |   |-- admin.py
|   |   |-- apps.py
|   |   |-- forms.py
|   |   |-- models.py
|   |   |-- services.py
|   |   |-- tests.py
|   |   |-- urls.py
|   |   `-- views.py
|   |-- static/
|   |   |-- css/
|   |   |   `-- app.css
|   |   |-- images/
|   |   |   `-- repo-intel-mark.svg
|   |   `-- js/
|   |       |-- dashboard.js
|   |       `-- report.js
|   |-- templates/
|   |   |-- accounts/
|   |   |   |-- otp_verify.html
|   |   |   |-- password_change.html
|   |   |   |-- password_reset_email.html
|   |   |   |-- password_reset_new.html
|   |   |   |-- profile.html
|   |   |   `-- register.html
|   |   |-- partials/
|   |   |   `-- brand_mark.html
|   |   |-- registration/
|   |   |   `-- login.html
|   |   |-- reviews/
|   |   |   |-- _analysis_table.html
|   |   |   |-- analyze.html
|   |   |   |-- dashboard.html
|   |   |   |-- guidelines.html
|   |   |   |-- history.html
|   |   |   |-- home.html
|   |   |   `-- report.html
|   |   `-- base.html
|   |-- Dockerfile
|   |-- manage.py
|   `-- requirements.txt
|-- .dockerignore
|-- .env.example                       # Configuration template; put real secrets in local .env
|-- .gitignore
|-- docker-compose.yml                 # PostgreSQL, Django web, and AI service
|-- README.md
`-- Repo_Intel_Project_Documentation.pdf
```

Generated and local-only files such as virtual environments, Chroma data, SQLite databases, caches,
and the populated `.env` file are intentionally excluded from this tree.
