"""Client for the FastAPI AI service, plus saving finished reports into PostgreSQL."""
from __future__ import annotations

from copy import deepcopy
import re

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import Analysis, Finding


class AIServiceError(Exception):
    """The AI service is unreachable or rejected the request."""


class PrivateRepositoryAuthRequired(AIServiceError):
    """The selected repository needs GitHub OAuth before it can be analyzed."""


class JobNotFound(AIServiceError):
    """The AI service no longer knows this job (it restarted)."""


_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?"
    r"(?:/tree/(?P<branch>[^/\s]+))?(?:/.*)?$"
)
_SHORT_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?$")


def parse_repo_url(text: str) -> tuple[str, str, str]:
    """Returns (owner, name, branch). Raises ValueError for anything that is not a GitHub repo."""
    text = (text or "").strip().rstrip("/")
    m = _URL_RE.match(text) or _SHORT_RE.match(text)
    if not m:
        raise ValueError("Enter a GitHub repository like https://github.com/owner/repo or owner/repo.")
    return m.group("owner"), m.group("name"), (m.groupdict().get("branch") or "")


def _headers() -> dict:
    return {"X-Internal-Token": settings.AI_SERVICE_TOKEN}


def _request(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{settings.AI_SERVICE_URL}{path}", headers=_headers(),
                                timeout=kwargs.pop("timeout", settings.AI_SERVICE_TIMEOUT), **kwargs)
    except requests.RequestException as exc:
        raise AIServiceError("The AI service is not reachable right now. Please try again in a moment.") from exc
    return resp


def _detail(resp) -> str:
    try:
        d = resp.json().get("detail")
        return d if isinstance(d, str) else str(d)
    except ValueError:
        return f"AI service error (HTTP {resp.status_code})"


def start_analysis(repo_url: str, branch: str, github_access_token: str, user_id: int) -> str:
    payload = {
        "repo_url": repo_url,
        "branch": branch or None,
        "github_access_token": github_access_token or None,
        "user_id": user_id,
    }
    resp = _request("POST", "/analyze", json=payload)
    if resp.status_code != 202:
        detail = _detail(resp)
        if resp.status_code == 401 and detail.startswith("GitHub authorization is required:"):
            raise PrivateRepositoryAuthRequired(detail)
        raise AIServiceError(detail)
    return resp.json()["job_id"]


def start_profile_analysis(
    username: str,
    access_token: str,
    user_id: int,
    *,
    include_private: bool,
    repository_name: str = "",
) -> str:
    payload = {
        "username": username,
        "repository_name": repository_name or None,
        "include_private": include_private,
        "github_access_token": access_token if include_private else None,
        "user_id": user_id,
    }
    resp = _request("POST", "/analyze-profile", json=payload)
    if resp.status_code != 202:
        raise AIServiceError(_detail(resp))
    return resp.json()["job_id"]


def get_job(job_id: str) -> dict:
    resp = _request("GET", f"/jobs/{job_id}")
    if resp.status_code == 404:
        raise JobNotFound("job lost")
    if resp.status_code != 200:
        raise AIServiceError(_detail(resp))
    return resp.json()


def upload_guideline(guideline_id: str, user_id: int, title: str, filename: str, data: bytes) -> int:
    resp = _request("POST", "/guidelines", data={"guideline_id": guideline_id, "user_id": str(user_id), "title": title},
                    files={"file": (filename, data)}, timeout=180)
    if resp.status_code != 200:
        raise AIServiceError(_detail(resp))
    return resp.json()["chunks"]


def delete_guideline(guideline_id: str) -> None:
    resp = _request("DELETE", f"/guidelines/{guideline_id}")
    if resp.status_code not in (200, 404):
        raise AIServiceError(_detail(resp))


# ------------------------------------------------------------------ syncing
def sync_analysis(analysis: Analysis) -> Analysis:
    """Pull the latest job state from the AI service and persist it. Safe to call repeatedly."""
    recover_revocation_failure = (
        analysis.status == Analysis.Status.FAILED
        and "GitHub access could not be revoked automatically" in analysis.error
        and bool(analysis.job_id)
    )
    if (analysis.is_finished and not recover_revocation_failure) or not analysis.job_id:
        return analysis
    try:
        job = get_job(analysis.job_id)
    except JobNotFound:
        if recover_revocation_failure:
            return analysis
        analysis.status, analysis.error = Analysis.Status.FAILED, "The analysis was interrupted (the AI service restarted). Please run it again."
        analysis.completed_at = timezone.now()
        analysis.save(update_fields=["status", "error", "completed_at"])
        return analysis
    # AIServiceError (service temporarily down) propagates; the caller keeps polling.

    analysis.stage = job.get("stage") or ""
    analysis.steps = job.get("steps") or []
    status = job["status"]
    report = job.get("report")
    if recover_revocation_failure and report:
        recovered_report = deepcopy(report)
        warning = (
            "GitHub access could not be revoked automatically. "
            "Revoke Repo Intel in GitHub Settings → Applications."
        )
        warnings = recovered_report.setdefault("warnings", [])
        if warning not in warnings:
            warnings.append(warning)
        save_report(analysis, recovered_report)
        return analysis
    if status == "done":
        save_report(analysis, report)
    elif status == "failed":
        analysis.status = Analysis.Status.FAILED
        analysis.error = job.get("error") or "Analysis failed."
        analysis.completed_at = timezone.now()
        analysis.save()
    else:
        analysis.status = Analysis.Status.RUNNING if status == "running" else Analysis.Status.QUEUED
        analysis.save(update_fields=["status", "stage", "steps"])
    return analysis


@transaction.atomic
def save_report(analysis: Analysis, report: dict) -> None:
    stats = report.setdefault("stats", {})
    rule_checks_used = bool(
        stats.get("rule_checks_used") or stats.get("rule_fallback_used")
    )
    findings = [
        finding for finding in report.pop("findings", [])
        if finding.get("source", "llm") in {"llm", "rag"}
        or (rule_checks_used and finding.get("source") == "rule")
    ]
    stats["ai_findings"] = sum(
        finding.get("source", "llm") in {"llm", "rag"}
        for finding in findings
    )
    stats["rule_findings"] = sum(finding.get("source") == "rule" for finding in findings)
    analysis.report = report
    analysis.scores = report.get("scores", {})
    analysis.overall_score = report.get("overall_score")
    analysis.status = Analysis.Status.DONE
    analysis.error = ""
    analysis.completed_at = timezone.now()
    analysis.save()
    analysis.findings.all().delete()
    Finding.objects.bulk_create([
        Finding(
            analysis=analysis, file=f["file"][:500], line=f.get("line"), category=f["category"], severity=f["severity"],
            issue=f["issue"][:300], evidence=f.get("evidence", ""), reason=f.get("reason", ""),
            recommendation=f.get("recommendation", ""), confidence=f.get("confidence", 0.7), source=f.get("source", "llm"),
            guideline=(f.get("guideline") or "")[:300],
        )
        for f in findings
    ])
