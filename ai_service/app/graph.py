"""The LangGraph workflow.

START -> fetch_repository -(error? END)-> detect_technology -> collect_files
      -> [quality | security | performance | documentation]   (parallel branches)
      -> best_practices (join, uses RAG when the user uploaded guidelines)
      -> architecture -> final_reviewer -> generate_report -> END
"""
from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from . import github_client, rag
from .config import get_settings
from .github_client import GitHubError, RepoSnapshot, parse_repo
from .llm import BaseLLM
from .reviewers import (
    Chunk, collect_chunks, compute_scores, dedupe, documentation_checklist, fallback_architecture,
    documentation_score, fallback_summary, file_priority, review_combined_chunks, select_chunks_per_file,
)
from .scanners import redact_secrets, scan_repository
from .schemas import SEVERITY_ORDER, Finding
from .static_checks import is_source_file, run_static_checks
from .tech_detect import detect_technology as detect_tech

log = logging.getLogger("repo-intel.graph")


class ReviewState(TypedDict, total=False):
    # input
    repo_url: str
    branch: str | None
    github_access_token: str | None
    user_id: int
    # produced by nodes
    snapshot: RepoSnapshot
    technology: dict
    source_files: list[str]
    chunks: list[Chunk]
    files_redacted: dict[str, str]
    doc_checklist: dict
    architecture: dict
    guideline_sources: list[str]
    guidelines_used: bool
    chunks_submitted_to_ai: int
    ai_review_completed: bool
    ai_reviewed_chunks: list[dict[str, Any]]
    guideline_passages_reviewed: int
    rule_fallback_used: bool
    rule_checks_used: bool
    final_findings: list[dict]
    scores: dict
    overall: int | None
    summary: dict
    error: str
    report: dict
    # parallel branches append to these (reducers merge their outputs)
    findings: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]
    evaluated: Annotated[list[str], operator.add]


def _llm(config: RunnableConfig) -> BaseLLM | None:
    return (config.get("configurable") or {}).get("llm")


def _dump(findings) -> list[dict]:
    return [f.model_dump() for f in findings]


# ------------------------------------------------------------------- nodes
async def fetch_repository(state: ReviewState, config: RunnableConfig) -> dict:
    try:
        ref = parse_repo(state["repo_url"], state.get("branch"))
        snap = await github_client.fetch_snapshot(ref, state.get("github_access_token"))
    except GitHubError as exc:
        return {"error": str(exc)}
    if not snap.files:
        return {"error": "The repository has no readable text files to analyze."}
    return {"snapshot": snap}


def route_after_fetch(state: ReviewState) -> str:
    return END if state.get("error") else "detect_technology"


async def detect_technology(state: ReviewState, config: RunnableConfig) -> dict:
    return {"technology": detect_tech(state["snapshot"])}


async def collect_files(state: ReviewState, config: RunnableConfig) -> dict:
    s = get_settings()
    snap = state["snapshot"]
    candidates = [p for p in snap.files if is_source_file(p)]
    ranked = sorted(candidates, key=lambda p: file_priority(p, snap.files[p]), reverse=True)[: s.max_source_files]
    chunks = collect_chunks(snap.files, ranked)
    return {
        "source_files": ranked,
        "chunks": chunks,
        "files_redacted": {p: redact_secrets(snap.files[p]) for p in ranked},
    }


async def quality(state: ReviewState, config: RunnableConfig) -> dict:
    llm, s = _llm(config), get_settings()
    snapshot = state["snapshot"]
    rule_findings = scan_repository(snapshot.files, snapshot.all_paths)
    rule_findings.extend(run_static_checks(snapshot.files, snapshot.all_paths))
    rule_findings.sort(
        key=lambda finding: (
            SEVERITY_ORDER[finding.severity],
            -finding.confidence,
            finding.file,
            finding.line or 0,
        )
    )
    rule_findings = [
        Finding.model_validate(item)
        for item in dedupe(_dump(rule_findings))[:5]
    ] if rule_findings else []
    findings = list(rule_findings)
    warnings: list[str] = []
    evaluated: list[str] = ["quality", "security"]
    guideline_sources: list[str] = []
    guidelines_used = False
    ai_review_completed = False
    ai_reviewed_chunks: list[dict[str, Any]] = []
    guideline_passages_reviewed = 0
    rule_fallback_used = False
    rule_checks_used = True
    chunks_submitted_to_ai = 0
    review_error: str | None = None
    if state.get("chunks"):
        chunks = select_chunks_per_file(
            state["chunks"],
            s.max_chunks_per_file,
            max_total=s.max_chunks_per_review,
        )
        if llm and chunks:
            guideline_docs = []
            try:
                guideline_docs = await asyncio.to_thread(
                    rag.get_user_guidelines, state["user_id"]
                )
            except Exception:
                warnings.append("Uploaded guidelines could not be loaded for this run.")

            (
                extra,
                error,
                succeeded,
                guideline_sources,
                reviewed_chunks,
                guideline_passages_reviewed,
            ) = await review_combined_chunks(
                llm,
                chunks,
                state["files_redacted"],
                state["technology"]["all"],
                guideline_docs,
                rag.citation,
                private_repository=state["snapshot"].is_private,
            )
            review_error = error
            findings += extra
            ai_review_completed = succeeded
            if succeeded:
                chunks_submitted_to_ai = len(reviewed_chunks)
                ai_reviewed_chunks = [
                    {
                        "file": chunk.path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.start_line + max(0, len(chunk.text.splitlines()) - 1),
                    }
                    for chunk in reviewed_chunks
                ]
            guidelines_used = bool(guideline_docs and succeeded)
            if succeeded:
                evaluated.extend(["performance", "best_practices"])
            elif error:
                if error.startswith((
                    "Groq rejected the single review request as too large",
                    "Groq review prompt",
                )):
                    rule_fallback_used = True
                    warnings.append(
                        "The AI review exceeded Groq's request limit. Up to five limited "
                        "rule-based findings are included; the overall score covers rule-based "
                        "quality, security, and documentation checks."
                    )
                else:
                    warnings.append(f"LLM review failed: {error}")
        elif not llm:
            try:
                if await asyncio.to_thread(rag.has_guidelines, state["user_id"]):
                    warnings.append(
                        "Uploaded coding guidelines were not reviewed because AI review is unavailable."
                    )
            except Exception:
                warnings.append("Uploaded guidelines could not be loaded for this run.")
    if ai_review_completed:
        log.info(
            "AI review completed (provider_requests_sent=%d, chunks_reviewed=%d, findings=%d).",
            getattr(llm, "provider_requests_sent", 0),
            chunks_submitted_to_ai,
            sum(f.source in {"llm", "rag"} for f in findings),
        )
    elif review_error:
        log.warning(
            "AI review did not complete (provider_requests_sent=%d): %s",
            getattr(llm, "provider_requests_sent", 0),
            review_error,
        )
    elif not llm:
        log.warning("AI review skipped: no LLM client was initialized; no provider request was sent.")
    elif not state.get("chunks"):
        log.warning("AI review skipped: no source-code chunks were available; no provider request was sent.")
    elif not chunks:
        log.warning("AI review skipped: no eligible source-code chunks; no provider request was sent.")
    return {
        "findings": _dump(findings),
        "warnings": warnings,
        "evaluated": evaluated,
        "guidelines_used": guidelines_used,
        "guideline_sources": guideline_sources,
        "chunks_submitted_to_ai": chunks_submitted_to_ai,
        "ai_review_completed": ai_review_completed,
        "ai_reviewed_chunks": ai_reviewed_chunks,
        "guideline_passages_reviewed": guideline_passages_reviewed,
        "rule_fallback_used": rule_fallback_used,
        "rule_checks_used": rule_checks_used,
    }


async def security(state: ReviewState, config: RunnableConfig) -> dict:
    return {
        "findings": [],
        "warnings": [],
        "evaluated": [],
    }


async def performance(state: ReviewState, config: RunnableConfig) -> dict:
    return {"findings": [], "warnings": [], "evaluated": []}


async def documentation(state: ReviewState, config: RunnableConfig) -> dict:
    snap = state["snapshot"]
    checklist, _, _ = documentation_checklist(snap.files, snap.all_paths)
    return {
        "findings": [],
        "warnings": [],
        "doc_checklist": checklist,
        "evaluated": ["documentation"],
    }


async def best_practices(state: ReviewState, config: RunnableConfig) -> dict:
    return {}


async def architecture(state: ReviewState, config: RunnableConfig) -> dict:
    return {"architecture": fallback_architecture(state["technology"]).model_dump()}


async def final_reviewer(state: ReviewState, config: RunnableConfig) -> dict:
    findings = dedupe(state.get("findings", []))
    evaluated = set(state.get("evaluated", []))
    scores, calculated_overall = compute_scores(
        findings,
        evaluated,
        {"documentation": documentation_score(state.get("doc_checklist", {}))},
    )
    if state.get("ai_review_completed"):
        overall = calculated_overall
        summary = fallback_summary(scores, overall, findings).model_dump()
    elif state.get("rule_checks_used"):
        overall = calculated_overall
        summary = {
            "summary": (
                f"AI review was unavailable. Rule-based quality, security, and documentation checks scored "
                f"{overall}/100; performance and best practices were not scored."
            ),
            "strengths": [],
            "top_recommendations": [
                finding["recommendation"]
                for finding in findings
                if finding.get("source") == "rule"
            ][:5],
        }
    else:
        overall = None
        summary = {
            "summary": "AI code review did not complete; no AI-generated findings or scores are available.",
            "strengths": [],
            "top_recommendations": [],
        }
    return {
        "final_findings": findings, "scores": scores, "overall": overall,
        "summary": summary, "warnings": [],
    }


async def generate_report(state: ReviewState, config: RunnableConfig) -> dict:
    snap, llm = state["snapshot"], _llm(config)
    findings = state["final_findings"]
    raw_warnings = list(dict.fromkeys(state.get("warnings", [])))
    llm_warnings = [
        warning for warning in raw_warnings
        if "llm review failed:" in warning.lower()
    ]
    warnings = [warning for warning in raw_warnings if warning not in llm_warnings]
    if llm_warnings:
        failure = llm_warnings[0].split(":", 1)[-1].strip()
        safe_prefixes = (
            "Groq rate limit reached.",
            "Groq rejected a private review request sent without response_format",
            "Groq returned malformed or truncated JSON for the private review",
            "Groq returned JSON that did not match the private review format",
            "Groq returned no text for the private repository review.",
            "Groq truncated the private review at its output-token limit",
            "Groq model ",
            "Groq rejected a review batch as too large",
            "Groq rejected the review request because it was too large.",
            "Groq rejected the model request.",
            "Groq authorization failed.",
            "Groq request failed after 3 attempts (",
        )
        if failure.startswith(safe_prefixes):
            cause = f"because {failure.rstrip('.')}"
        else:
            cause = "because of a Groq provider error"
        warnings.append(
            f"The AI review could not complete {cause}. "
            "The overall score reflects rule-based quality, security, and documentation checks only; "
            "performance and best-practices scores are unavailable."
        )
    if llm is None:
        settings = get_settings()
        if settings.llm_mode == "off":
            reason = "LLM_MODE is set to off."
        elif settings.llm_mode == "groq" and not settings.groq_api_key:
            reason = "GROQ_API_KEY is not set in the running AI service."
        elif settings.llm_enabled:
            reason = "LLM settings are available, but no LLM client was initialized for this run."
        else:
            reason = f"LLM_MODE={settings.llm_mode!r} does not enable a supported LLM."
        warnings.insert(
            0,
            f"No LLM was used for this run ({reason}); limited rule-based findings are shown "
            "when available. The overall score covers rule-based quality, security, and documentation checks; "
            "performance and best-practices scores are unavailable. Restart the AI service "
            "after changing its .env configuration.",
        )
    report: dict[str, Any] = {
        "repository": {
            "owner": snap.ref.owner, "name": snap.ref.name, "url": snap.ref.url, "branch": snap.default_branch,
            "description": snap.description, "stars": snap.stars,
        },
        "overall_score": state["overall"],
        "scores": state["scores"],
        "severity_counts": {sev: sum(1 for f in findings if f["severity"] == sev) for sev in SEVERITY_ORDER},
        "category_counts": {c: sum(1 for f in findings if f["category"] == c) for c in
                            ("quality", "security", "performance", "best_practices", "documentation")},
        "technology": state["technology"],
        "architecture": state["architecture"],
        "doc_checklist": state.get("doc_checklist", {}),
        "summary": state["summary"]["summary"],
        "strengths": state["summary"].get("strengths", []),
        "top_recommendations": state["summary"].get("top_recommendations", [])[:5],
        "findings": findings,
        "files_reviewed": state.get("source_files", []),
        "rule_files_scanned": len(snap.files),
        "guidelines_used": bool(state.get("guidelines_used")),
        "guideline_passages_reviewed": state.get("guideline_passages_reviewed", 0),
        "guideline_sources": state.get("guideline_sources", []),
        "ai_reviewed_chunks": state.get("ai_reviewed_chunks", []),
        "rule_fallback_used": state.get("rule_fallback_used", False),
        "rule_checks_used": state.get("rule_checks_used", False),
        "warnings": warnings,
        "stats": {
            "files_in_repo": len(snap.all_paths),
            "files_reviewed": len(state.get("source_files", [])),
            "rule_files_scanned": len(snap.files),
            "chunks_available": len(state.get("chunks", [])),
            "chunks_submitted_to_ai": state.get("chunks_submitted_to_ai", 0),
            "ai_review_completed": state.get("ai_review_completed", False),
            "ai_findings": sum(f.get("source") in {"llm", "rag"} for f in findings),
            "rule_findings": sum(f.get("source") == "rule" for f in findings),
            "rule_fallback_used": state.get("rule_fallback_used", False),
            "rule_checks_used": state.get("rule_checks_used", False),
            "llm_calls": llm.calls if llm else 0,
            "provider_requests_sent": getattr(llm, "provider_requests_sent", 0),
        },
    }
    return {"report": report}


# ------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_repository", fetch_repository), ("detect_technology", detect_technology), ("collect_files", collect_files),
        ("quality", quality), ("security", security), ("performance", performance), ("documentation", documentation),
        ("best_practices", best_practices), ("architecture", architecture), ("final_reviewer", final_reviewer),
        ("generate_report", generate_report),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "fetch_repository")
    g.add_conditional_edges("fetch_repository", route_after_fetch, {"detect_technology": "detect_technology", END: END})
    g.add_edge("detect_technology", "collect_files")
    for branch in ("quality", "security", "performance", "documentation"):  # fan-out: run in parallel
        g.add_edge("collect_files", branch)
    g.add_edge(["quality", "security", "performance", "documentation"], "best_practices")  # fan-in: wait for all
    g.add_edge("best_practices", "architecture")
    g.add_edge("architecture", "final_reviewer")
    g.add_edge("final_reviewer", "generate_report")
    g.add_edge("generate_report", END)
    return g.compile()


STEP_LABELS = [
    "fetch_repository", "detect_technology", "collect_files", "quality", "security", "performance",
    "documentation", "best_practices", "architecture", "final_reviewer", "generate_report",
]
