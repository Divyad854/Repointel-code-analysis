"""The LangGraph workflow.

START -> fetch_repository -(error? END)-> detect_technology -> collect_files
      -> [quality | security | performance | documentation]   (parallel branches)
      -> best_practices (join, uses RAG when the user uploaded guidelines)
      -> architecture -> final_reviewer -> generate_report -> END
"""
from __future__ import annotations

import asyncio
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
    fallback_summary, file_priority, rank_chunks, review_combined_chunks,
)
from .scanners import redact_secrets, scan_repository
from .schemas import SEVERITY_ORDER
from .static_checks import is_source_file, run_static_checks
from .tech_detect import detect_technology as detect_tech


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
    final_findings: list[dict]
    scores: dict
    overall: int
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
    snap, llm, s = state["snapshot"], _llm(config), get_settings()
    findings = run_static_checks(snap.files, snap.all_paths)
    warnings: list[str] = []
    evaluated = ["quality"]
    guideline_sources: list[str] = []
    guidelines_used = False
    if state.get("chunks"):
        ranked_chunks = rank_chunks(state["chunks"], None, len(state["chunks"]))
        chunks_by_file = {}
        for chunk in ranked_chunks:
            chunks_by_file.setdefault(chunk.path, chunk)
        chunks = list(chunks_by_file.values())
        if llm and chunks:
            guideline_docs = []
            try:
                guideline_docs = await asyncio.to_thread(
                    rag.get_user_guidelines, state["user_id"]
                )
            except Exception:
                warnings.append("Uploaded guidelines could not be loaded for this run.")

            extra, error, succeeded, guideline_sources = await review_combined_chunks(
                llm,
                chunks,
                state["files_redacted"],
                state["technology"]["all"],
                guideline_docs,
                rag.citation,
            )
            findings += extra
            guidelines_used = bool(guideline_docs and succeeded)
            if succeeded:
                evaluated.extend(["security", "performance", "best_practices"])
            elif error:
                warnings.append(f"LLM review failed: {error}")
        elif not llm:
            try:
                if await asyncio.to_thread(rag.has_guidelines, state["user_id"]):
                    warnings.append(
                        "Uploaded coding guidelines were not reviewed because AI review is unavailable."
                    )
            except Exception:
                warnings.append("Uploaded guidelines could not be loaded for this run.")
    return {
        "findings": _dump(findings),
        "warnings": warnings,
        "evaluated": evaluated,
        "guidelines_used": guidelines_used,
        "guideline_sources": guideline_sources,
    }


async def security(state: ReviewState, config: RunnableConfig) -> dict:
    snap = state["snapshot"]
    return {
        "findings": _dump(scan_repository(snap.files, snap.all_paths)),
        "warnings": [],
        "evaluated": ["security"],
    }


async def performance(state: ReviewState, config: RunnableConfig) -> dict:
    return {"findings": [], "warnings": [], "evaluated": []}


async def documentation(state: ReviewState, config: RunnableConfig) -> dict:
    snap = state["snapshot"]
    checklist, findings, _ = documentation_checklist(snap.files, snap.all_paths)
    return {
        "findings": _dump(findings),
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
    scores, overall = compute_scores(findings, set(state.get("evaluated", [])))
    summary = fallback_summary(scores, overall, findings)
    return {
        "final_findings": findings, "scores": scores, "overall": overall,
        "summary": summary.model_dump(), "warnings": [],
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
        reason = "rate limit" if "rate limit" in llm_warnings[0].lower() else "provider error"
        warnings.append(
            f"The AI review could not complete because of a Groq {reason}. "
            "Rule-based security and quality checks are included; AI-only scores are unavailable."
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
            f"No LLM was used for this run ({reason}); only rule-based checks ran, "
            "so performance and best-practices scores are not available. Restart the AI service "
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
        "guidelines_used": bool(state.get("guidelines_used")),
        "guideline_sources": state.get("guideline_sources", []),
        "warnings": warnings,
        "stats": {
            "files_in_repo": len(snap.all_paths),
            "files_reviewed": len(state.get("source_files", [])),
            "chunks_available": len(state.get("chunks", [])),
            "llm_calls": llm.calls if llm else 0,
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
