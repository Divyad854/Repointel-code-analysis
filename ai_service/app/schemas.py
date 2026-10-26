"""Pydantic models: API payloads and structured LLM outputs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low"]
Category = Literal["quality", "security", "performance", "best_practices", "documentation"]
Source = Literal["rule", "llm", "rag"]

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ---------------------------------------------------------------- findings
class Finding(BaseModel):
    """One evidence-backed review finding (what the UI and database store)."""

    file: str
    line: int | None = None
    category: Category
    severity: Severity
    issue: str
    evidence: str = ""
    reason: str = ""
    recommendation: str
    confidence: float = 0.7
    source: Source = "llm"
    guideline: str | None = None


# ------------------------------------------------- structured LLM outputs
class LLMFinding(BaseModel):
    """A finding as returned by the LLM for one numbered code chunk."""

    line: int = Field(description="Line number exactly as printed at the start of the offending line.")
    severity: Severity = Field(description="critical, high, medium or low.")
    issue: str = Field(description="Short title of the problem (max 15 words).")
    evidence: str = Field(description="The exact offending code copied verbatim from the chunk (1-2 lines, without the line-number prefix).")
    reason: str = Field(description="Why this is a problem, referring to this specific code.")
    recommendation: str = Field(description="Concrete fix for this code.")
    confidence: float = Field(description="0.0 to 1.0, how sure you are this is a real problem.")


class ReviewResult(BaseModel):
    findings: list[LLMFinding] = Field(default_factory=list)


class CategorizedLLMFinding(LLMFinding):
    file: str = Field(description="Exact repository-relative path of the file containing the finding.")
    category: Category = Field(description="One of quality, security, performance, or best_practices.")
    guideline_ref: int | None = Field(
        default=None,
        description="Guideline passage number if this finding violates a supplied guideline; otherwise null.",
    )


class CombinedReviewResult(BaseModel):
    findings: list[CategorizedLLMFinding] = Field(default_factory=list)


class RagFinding(LLMFinding):
    guideline_ref: int = Field(description="Number of the guideline passage [1], [2]... that this finding violates.")


class RagReviewResult(BaseModel):
    findings: list[RagFinding] = Field(default_factory=list)


class DocReview(BaseModel):
    """Qualitative review of the README."""

    problems: list[str] = Field(default_factory=list, description="Concrete documentation problems (max 5).")
    suggestions: list[str] = Field(default_factory=list, description="Concrete improvements (max 5).")


class ArchitectureInfo(BaseModel):
    frontend: str = Field(default="Not detected", description="Frontend technology or 'Not detected'.")
    backend: str = Field(default="Not detected")
    database: str = Field(default="Not detected")
    authentication: str = Field(default="Not detected")
    deployment: str = Field(default="Not detected")
    api_style: str = Field(default="Not detected", description="REST, GraphQL, RPC, CLI, none...")
    components: list[str] = Field(default_factory=list, description="Main components/modules, one short phrase each.")
    data_flow: str = Field(default="", description="One or two sentences describing how data flows through the system.")
    summary: str = Field(default="", description="Three to four sentence architecture explanation.")


class FinalSummary(BaseModel):
    summary: str = Field(description="Three to five sentence executive summary of the repository's health.")
    strengths: list[str] = Field(default_factory=list, description="Up to 4 genuine strengths seen in the repository.")
    top_recommendations: list[str] = Field(description="Up to 5 prioritised, actionable recommendations.")


# --------------------------------------------------------------- API models
class AnalyzeRequest(BaseModel):
    repo_url: str
    branch: str | None = None
    github_access_token: str | None = None
    user_id: int


class ProfileAnalyzeRequest(BaseModel):
    username: str
    repository_name: str | None = None
    include_private: bool = False
    github_access_token: str | None = None
    user_id: int


class AnalyzeResponse(BaseModel):
    job_id: str


class GuidelineResponse(BaseModel):
    guideline_id: str
    chunks: int
