

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any


from .config import get_settings
from .graph import STEP_LABELS, build_graph
from .github_client import GitHubError, list_profile_repositories
from .llm import build_llm
from .schemas import AnalyzeRequest, ProfileAnalyzeRequest

# records job-related information.
log = logging.getLogger("repo-intel.jobs")

JOBS: dict[str, dict[str, Any]] = {}
_MAX_JOBS = 200
_graph = None


# Build the LangGraph workflow once and reuse it for later jobs.
def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# Create a job record so the UI can track its queued/running/done/failed status.
def create_job() -> str:
    if len(JOBS) >= _MAX_JOBS:  # drop the oldest finished jobs
        # Finds completed and failed jobs and sorts them by creation time, oldest first.
        finished = sorted((j["created"], k) for k, j in JOBS.items() if j["status"] in ("done", "failed"))
        for _, k in finished[: len(JOBS) - _MAX_JOBS + 1]:
            # remove old jobs
            JOBS.pop(k, None)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "stage": "", "steps": [], "total_steps": len(STEP_LABELS),
                    "report": None, "error": None, "created": time.time()}
    return job_id


# Run one repository analysis as a background job and collect graph updates.
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

    #Execute LangGraph using _run()
    # astream() executes the LangGraph workflow asynchronously and streams updates as nodes finish.
    #  Stream LangGraph node updates so progress, errors and the final report are captured.
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
        # Initialize the LLM once for this repository analysis.
        llm = build_llm()
        # Stop the analysis if it exceeds the configured job timeout.
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




# Combine individual repository reports into one GitHub-profile-level report.
def _aggregate_profile(username: str, reports: list[dict], warnings: list[str], duration: float) -> dict:
    # Merge findings and prefix each file path with its repository name to avoid ambiguity.
    findings = []
    for report in reports:
        repo_name = report["repository"]["name"]
        findings.extend({**finding, "file": f"{repo_name}/{finding['file']}"}
                        for finding in report.get("findings", []))



    # Calculate the average score for each category across repositories that have a score.
    score_names = ("quality", "security", "performance", "best_practices", "documentation")
    scores = {}
    for name in score_names:
        values = [report["scores"][name] for report in reports if report["scores"].get(name) is not None]
        scores[name] = round(sum(values) / len(values)) if values else None


    # Calculate the overall profile score from repositories with an available overall score.
    repository_scores = [
        report["overall_score"] for report in reports
        if report.get("overall_score") is not None
    ]
    average_score = (
        round(sum(repository_scores) / len(repository_scores))
        if repository_scores else None
    )


    # Keep a compact summary of each repository for the profile report's repository list.
    repositories = [{
        "name": report["repository"]["name"], "url": report["repository"]["url"],
        "score": report["overall_score"], "summary": report["summary"],
        "findings": len(report.get("findings", [])),
        "architecture": report.get("architecture", {}).get("summary", ""),
        "technology": report.get("technology", {}).get("all", []),
    } for report in reports]

    # Merge technology information across repositories, including language percentages.
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
    # Prepare aggregate counts for finding severity and analysis categories.
    severities = ("critical", "high", "medium", "low")
    categories = ("quality", "security", "performance", "best_practices", "documentation")
    checklist_keys = set().union(*(report.get("doc_checklist", {}).keys() for report in reports))
    doc_checklist = {
        key: all(report.get("doc_checklist", {}).get(key, False) for report in reports)
        for key in checklist_keys
    }
    # A single-repository profile can show its architecture; multiple repositories may differ.
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
        "summary": (
            f"Analyzed {len(reports)} repositories owned by @{username}. "
            + (
                f"Average repository health score: {average_score}/100."
                if average_score is not None
                else "AI review did not complete for any repository, so no health score is available."
            )
        ),
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


# Analyze all selected repositories in a GitHub profile and aggregate their reports.
# analyzes multiple repositories.
# Analyze multiple repositories sequentially
async def run_profile_job(job_id: str, req: ProfileAnalyzeRequest) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    started = time.time()
    reports: list[dict] = []
    warnings: list[str] = []
    error = ""
    # Returns an increasing clock value
    deadline = time.monotonic() + get_settings().job_timeout_s
    try:
        # Fetch the profile's repositories, respecting the remaining job time limit.
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
        # Reuse one LLM client while reviewing each repository in the profile.
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

            # Run the LangGraph workflow for this repository and capture its report.
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
        # A profile report requires at least one successfully analyzed repository.
        if not reports:
            raise RuntimeError("None of the profile's repositories could be analyzed.")
        job["report"] = _aggregate_profile(req.username, reports, warnings, time.time() - started)
    except Exception as exc:
        log.exception("profile job %s failed", job_id)
        error = str(exc)[:300]

    job["error"] = error or None
    job["status"] = "failed" if error else "done"
