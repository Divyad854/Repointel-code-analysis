"""In-memory job registry. Django polls /jobs/{id}; results are persisted by Django in PostgreSQL."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import httpx

from .config import get_settings
from .graph import STEP_LABELS, build_graph
from .github_client import GitHubError, list_profile_repositories
from .llm import build_llm
from .schemas import AnalyzeRequest, ProfileAnalyzeRequest

log = logging.getLogger("repo-intel.jobs")

JOBS: dict[str, dict[str, Any]] = {}
_MAX_JOBS = 200
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def create_job() -> str:
    if len(JOBS) >= _MAX_JOBS:  # drop the oldest finished jobs
        finished = sorted((j["created"], k) for k, j in JOBS.items() if j["status"] in ("done", "failed"))
        for _, k in finished[: len(JOBS) - _MAX_JOBS + 1]:
            JOBS.pop(k, None)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "stage": "", "steps": [], "total_steps": len(STEP_LABELS),
                    "report": None, "error": None, "created": time.time()}
    return job_id


async def run_job(job_id: str, req: AnalyzeRequest) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    started = time.time()
    state = {
        "repo_url": req.repo_url,
        "branch": req.branch,
        "github_access_token": req.github_access_token,
        "user_id": req.user_id,
    }

    async def _run() -> None:
        async for update in _get_graph().astream(state, config={"configurable": {"llm": llm}}, stream_mode="updates"):
            for node, values in update.items():
                job["steps"].append(node)
                job["stage"] = node
                values = values or {}
                if values.get("error"):
                    job["error"] = values["error"]
                if values.get("report"):
                    job["report"] = values["report"]

    try:
        llm = build_llm()
        await asyncio.wait_for(_run(), timeout=get_settings().job_timeout_s)
        if job["error"] or not job["report"]:
            job["status"] = "failed"
            job["error"] = job["error"] or "Analysis finished without a report."
        else:
            job["report"]["stats"]["duration_s"] = round(time.time() - started, 1)
            job["status"] = "done"
    except asyncio.TimeoutError:
        job["status"], job["error"] = "failed", "The analysis timed out."
    except Exception as exc:  # never leave a job hanging
        log.exception("job %s crashed", job_id)
        job["status"], job["error"] = "failed", f"Internal error: {exc.__class__.__name__}"
    finally:
        if req.github_access_token:
            try:
                await _revoke_github_grant(req.github_access_token)
            except Exception:
                log.exception("GitHub grant revocation failed for repository job %s", job_id)
                revoke_error = (
                    "GitHub access could not be revoked automatically. "
                    "Revoke Repo Intel in GitHub Settings → Applications."
                )
                if job["status"] == "done" and job["report"]:
                    warnings = job["report"].setdefault("warnings", [])
                    if revoke_error not in warnings:
                        warnings.append(revoke_error)
                else:
                    job["error"] = f"{job['error'] or ''} {revoke_error}".strip()
                    job["status"] = "failed"
            req.github_access_token = None


def _aggregate_profile(username: str, reports: list[dict], warnings: list[str], duration: float) -> dict:
    findings = []
    for report in reports:
        repo_name = report["repository"]["name"]
        findings.extend({**finding, "file": f"{repo_name}/{finding['file']}"}
                        for finding in report.get("findings", []))

    score_names = ("quality", "security", "performance", "best_practices", "documentation")
    scores = {}
    for name in score_names:
        values = [report["scores"][name] for report in reports if report["scores"].get(name) is not None]
        scores[name] = round(sum(values) / len(values)) if values else None
    average_score = round(sum(report["overall_score"] for report in reports) / len(reports))
    repositories = [{
        "name": report["repository"]["name"], "url": report["repository"]["url"],
        "score": report["overall_score"], "summary": report["summary"],
        "findings": len(report.get("findings", [])),
        "architecture": report.get("architecture", {}).get("summary", ""),
        "technology": report.get("technology", {}).get("all", []),
    } for report in reports]
    technology = dict(reports[0].get("technology", {}))
    languages: dict[str, list[float]] = {}
    for report in reports:
        for language in report.get("technology", {}).get("languages", []):
            languages.setdefault(language["name"], []).append(language["percent"])
    technology["languages"] = sorted(
        ({"name": name, "percent": round(sum(values) / len(reports), 1)}
         for name, values in languages.items()),
        key=lambda item: item["percent"], reverse=True,
    )
    technology["all"] = list(dict.fromkeys(
        item for report in reports for item in report.get("technology", {}).get("all", [])
    ))
    tech_groups = {}
    for report in reports:
        for group, names in report.get("technology", {}).get("technologies", {}).items():
            tech_groups.setdefault(group, []).extend(names)
    technology["technologies"] = {
        group: list(dict.fromkeys(names)) for group, names in tech_groups.items()
    }
    severities = ("critical", "high", "medium", "low")
    categories = ("quality", "security", "performance", "best_practices", "documentation")
    checklist_keys = set().union(*(report.get("doc_checklist", {}).keys() for report in reports))
    doc_checklist = {
        key: all(report.get("doc_checklist", {}).get(key, False) for report in reports)
        for key in checklist_keys
    }
    architecture = reports[0].get("architecture", {}) if len(reports) == 1 else {
        "frontend": "Varies by repository", "backend": "Varies by repository",
        "database": "Varies by repository", "authentication": "Varies by repository",
        "deployment": "Varies by repository", "api_style": "Varies by repository",
        "components": [], "data_flow": "",
        "summary": "Architecture varies across repositories; see each repository's details below.",
    }
    return {
        "repository": {
            "owner": username, "name": "profile", "url": f"https://github.com/{username}",
            "branch": "", "description": f"GitHub profile analysis for @{username}", "stars": 0,
        },
        "repositories": repositories,
        "overall_score": average_score,
        "scores": scores,
        "severity_counts": {key: sum(f["severity"] == key for f in findings) for key in severities},
        "category_counts": {key: sum(f["category"] == key for f in findings) for key in categories},
        "technology": technology,
        "architecture": architecture,
        "doc_checklist": doc_checklist,
        "summary": f"Analyzed {len(reports)} repositories owned by @{username}. Average repository health score: {average_score}/100.",
        "strengths": list(dict.fromkeys(s for report in reports for s in report.get("strengths", [])))[:4],
        "top_recommendations": list(dict.fromkeys(
            item for report in reports for item in report.get("top_recommendations", [])
        ))[:5],
        "findings": findings,
        "files_reviewed": [
            f"{report['repository']['name']}/{path}"
            for report in reports for path in report.get("files_reviewed", [])
        ],
        "guidelines_used": any(report.get("guidelines_used") for report in reports),
        "guideline_sources": list(dict.fromkeys(
            item for report in reports for item in report.get("guideline_sources", [])
        )),
        "warnings": list(dict.fromkeys(warnings + [w for report in reports for w in report.get("warnings", [])])),
        "stats": {
            "files_in_repo": sum(r["stats"].get("files_in_repo", 0) for r in reports),
            "files_reviewed": sum(r["stats"].get("files_reviewed", 0) for r in reports),
            "chunks_available": sum(r["stats"].get("chunks_available", 0) for r in reports),
            "llm_calls": sum(r["stats"].get("llm_calls", 0) for r in reports),
            "duration_s": round(duration, 1),
        },
    }


async def _revoke_profile_grant(req: ProfileAnalyzeRequest) -> None:
    if not req.github_access_token:
        raise RuntimeError("GitHub access token is unavailable for revocation.")
    await _revoke_github_grant(req.github_access_token)


async def _revoke_github_grant(access_token: str) -> None:
    settings = get_settings()
    if not settings.github_oauth_client_id or not settings.github_oauth_client_secret:
        raise RuntimeError("GitHub grant revocation is not configured.")
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.delete(
                    f"https://api.github.com/applications/{settings.github_oauth_client_id}/grant",
                    auth=(settings.github_oauth_client_id, settings.github_oauth_client_secret),
                    json={"access_token": access_token},
                    headers={"Accept": "application/vnd.github+json"},
                )
            if response.status_code == 404:
                return
            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    f"GitHub rejected grant revocation (HTTP {response.status_code}). "
                    "Check the GitHub OAuth app client ID and secret."
                )
            response.raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))


async def run_profile_job(job_id: str, req: ProfileAnalyzeRequest) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    started = time.time()
    reports: list[dict] = []
    warnings: list[str] = []
    error = ""
    deadline = time.monotonic() + get_settings().job_timeout_s
    try:
        repos = await asyncio.wait_for(
            list_profile_repositories(
                req.username,
                include_private=req.include_private,
                access_token=req.github_access_token,
                repository_name=req.repository_name,
            ),
            timeout=max(0.1, deadline - time.monotonic()),
        )
        if not repos:
            if req.repository_name:
                raise GitHubError(
                    f"Repository {req.username}/{req.repository_name} was not found in the profile's owned repositories or is not accessible."
                )
            raise GitHubError(f"No repositories were found for @{req.username}.")
        llm = build_llm()
        for repo in repos:
            state = {
                "repo_url": repo.url,
                "branch": None,
                "github_access_token": req.github_access_token,
                "user_id": req.user_id,
            }
            if time.monotonic() >= deadline:
                raise asyncio.TimeoutError

            async def review_repo():
                current_report = None
                async for update in _get_graph().astream(
                    state, config={"configurable": {"llm": llm}}, stream_mode="updates"
                ):
                    for node, values in update.items():
                        values = values or {}
                        job["stage"] = f"{repo.name}: {node}"
                        job["steps"].append(node)
                        if values.get("report"):
                            current_report = values["report"]
                        if values.get("error"):
                            warnings.append(f"{repo.name}: {values['error']}")
                return current_report

            try:
                current_report = await asyncio.wait_for(
                    review_repo(), timeout=max(0.1, deadline - time.monotonic())
                )
                if current_report:
                    reports.append(current_report)
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                log.exception("profile repository analysis failed for %s", repo.full_name)
                warnings.append(f"{repo.name}: analysis failed ({exc.__class__.__name__}).")
        if not reports:
            raise RuntimeError("None of the profile's repositories could be analyzed.")
        job["report"] = _aggregate_profile(req.username, reports, warnings, time.time() - started)
    except Exception as exc:
        log.exception("profile job %s failed", job_id)
        error = str(exc)[:300]
    finally:
        if req.include_private:
            try:
                await _revoke_profile_grant(req)
            except Exception:
                log.exception("GitHub grant revocation failed for profile job %s", job_id)
                revoke_error = (
                    "GitHub access could not be revoked automatically. Revoke Repo Intel in GitHub Settings → Applications."
                )
                if job["report"] and not error:
                    warnings = job["report"].setdefault("warnings", [])
                    if revoke_error not in warnings:
                        warnings.append(revoke_error)
                else:
                    error = (error + " " if error else "") + revoke_error
        req.github_access_token = None

    job["error"] = error or None
    job["status"] = "failed" if error else "done"
