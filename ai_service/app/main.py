"""FastAPI AI microservice. Django talks to this service; this service talks to GitHub, Groq and Chroma."""
from __future__ import annotations

import asyncio
import hmac
import logging
import re

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from . import rag
from .config import get_settings
from .github_client import GitHubAuthorizationRequired, GitHubError, ensure_public_repository, parse_repo
from .jobs import JOBS, create_job, run_job, run_profile_job
from .schemas import AnalyzeRequest, AnalyzeResponse, GuidelineResponse, ProfileAnalyzeRequest

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Repo Intel AI Service", version="1.0.0")
MAX_GUIDELINE_BYTES = 10 * 1024 * 1024


def require_token(x_internal_token: str = Header(default="")) -> None:
    if not hmac.compare_digest(x_internal_token, get_settings().internal_token):
        raise HTTPException(status_code=401, detail="Invalid service token")


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_mode": s.llm_mode,
        "llm_enabled": s.llm_enabled,
        "llm_model": s.groq_model if s.llm_mode == "groq" else None,
        "max_source_files": s.max_source_files,
        "max_chunks_per_file": s.max_chunks_per_file,
        "max_chunks_per_reviewer": s.max_chunks_per_file,
        "llm_concurrency": s.llm_concurrency,
        "embeddings": s.embedding_backend,
    }


@app.post("/analyze", response_model=AnalyzeResponse, status_code=202, dependencies=[Depends(require_token)])
async def analyze(req: AnalyzeRequest, background: BackgroundTasks) -> AnalyzeResponse:
    try:
        repo_ref = parse_repo(req.repo_url, req.branch)
    except GitHubError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    service_settings = get_settings()
    if not req.github_access_token:
        try:
            await ensure_public_repository(repo_ref)
        except GitHubAuthorizationRequired as exc:
            raise HTTPException(
                status_code=401,
                detail=f"GitHub authorization is required: {exc}",
            ) from exc
        except GitHubError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if req.github_access_token and not all((
        service_settings.github_oauth_client_id, service_settings.github_oauth_client_secret,
    )):
        raise HTTPException(
            status_code=503,
            detail="GitHub grant revocation is not configured on the AI service. Set GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET in the shared .env, then restart the AI service.",
        )
    job_id = create_job()
    background.add_task(run_job, job_id, req)
    return AnalyzeResponse(job_id=job_id)


@app.post("/analyze-profile", response_model=AnalyzeResponse, status_code=202, dependencies=[Depends(require_token)])
async def analyze_profile(req: ProfileAnalyzeRequest, background: BackgroundTasks) -> AnalyzeResponse:
    if not re.fullmatch(r"(?!-)(?!.*--)[A-Za-z0-9-]{1,39}(?<!-)", req.username):
        raise HTTPException(status_code=422, detail="Enter a valid GitHub username.")
    if req.repository_name and not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", req.repository_name):
        raise HTTPException(status_code=422, detail="Enter a valid repository name without a slash.")
    if req.include_private and not req.github_access_token:
        raise HTTPException(status_code=422, detail="GitHub authorization is required for private profile analysis.")
    service_settings = get_settings()
    if req.include_private and not all((
        service_settings.github_oauth_client_id, service_settings.github_oauth_client_secret,
    )):
        raise HTTPException(
            status_code=503,
            detail="GitHub grant revocation is not configured on the AI service. Set GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET in the shared .env, then restart the AI service.",
        )
    if not req.include_private and req.github_access_token:
        raise HTTPException(status_code=422, detail="Public profile analysis must not include an access token.")
    job_id = create_job()
    background.add_task(run_profile_job, job_id, req)
    return AnalyzeResponse(job_id=job_id)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
async def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {k: job[k] for k in ("status", "stage", "steps", "total_steps", "error", "report")}


@app.post("/guidelines", response_model=GuidelineResponse, dependencies=[Depends(require_token)])
async def upload_guideline(
    file: UploadFile = File(...), guideline_id: str = Form(...), user_id: int = Form(...), title: str = Form(""),
) -> GuidelineResponse:
    data = await file.read()
    if len(data) > MAX_GUIDELINE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
    try:
        chunks = await asyncio.to_thread(
            rag.index_guideline, guideline_id, user_id, title or file.filename or "guideline", file.filename or "guideline.txt", data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logging.exception("indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc.__class__.__name__}: {str(exc)[:200]}")
    return GuidelineResponse(guideline_id=guideline_id, chunks=chunks)


@app.delete("/guidelines/{guideline_id}", dependencies=[Depends(require_token)])
async def delete_guideline(guideline_id: str) -> dict:
    await asyncio.to_thread(rag.delete_guideline, guideline_id)
    return {"deleted": guideline_id}
