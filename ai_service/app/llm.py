"""LLM access: Groq through LangChain (LCEL prompt | structured model), plus a mock for tests."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import TypeVar

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .config import Settings, get_settings
from .schemas import (
    ArchitectureInfo, CategorizedLLMFinding, CombinedReviewResult, DocReview, FinalSummary, LLMFinding,
    RagFinding, RagReviewResult, ReviewResult,
)

log = logging.getLogger("repo-intel.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class BaseLLM:
    """Interface used by the graph nodes."""

    def __init__(self, concurrency: int = 3) -> None:
        self._sem = asyncio.Semaphore(concurrency)
        self.calls = 0
        self._blocked_error: str | None = None

    async def structured(self, schema: type[T], system: str, user: str) -> T:
        async with self._sem:
            if self._blocked_error:
                raise LLMError(self._blocked_error)
            self.calls += 1
            return await self._run(schema, system, user)

    async def _run(self, schema: type[T], system: str, user: str) -> T:  # pragma: no cover - interface
        raise NotImplementedError


class GroqLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.llm_concurrency)
        from langchain_groq import ChatGroq  # imported lazily so tests don't need the package configured

        self._model = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.1,
            max_retries=0,
            timeout=90,
        )
        self._prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])

    async def _run(self, schema: type[T], system: str, user: str) -> T:
        if self._blocked_error:
            raise LLMError(self._blocked_error)
        # Groq's JSON Schema mode is supported by GPT-OSS and avoids tool-call parsing failures.
        chain = self._prompt | self._model.with_structured_output(
            schema,
            method="json_schema",
            strict=False,
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                result = await chain.ainvoke({"system": system, "user": user})
                if result is None:
                    raise LLMError("empty structured response")
                return result
            except Exception as exc:  # network, rate limit, validation...
                last = exc
                status_code = getattr(exc, "status_code", None)
                if status_code == 429:
                    self._blocked_error = (
                        "Groq rate limit reached. Remaining LLM reviews were skipped for this run to avoid "
                        "repeated requests."
                    )
                    log.warning("%s", self._blocked_error)
                    raise LLMError(self._blocked_error) from exc
                if status_code in (400, 401, 403, 404, 422):
                    if status_code == 404:
                        self._blocked_error = (
                            f"Groq model {get_settings().groq_model!r} was not found or is no longer available. "
                            "Set GROQ_MODEL to an active model from https://console.groq.com/docs/models."
                        )
                    elif status_code in (400, 422):
                        self._blocked_error = (
                            "Groq rejected the model request. Check model capabilities and request settings."
                        )
                    else:
                        self._blocked_error = (
                            "Groq authorization failed. Check the configured API key and account access."
                        )
                    error_code = None
                    body = getattr(exc, "body", None)
                    if isinstance(body, dict) and isinstance(body.get("error"), dict):
                        error_code = body["error"].get("code")
                    log.warning(
                        "Stopping LLM calls for this run after provider rejection (HTTP %s, code=%s).",
                        status_code,
                        error_code or "unspecified",
                    )
                    raise LLMError(self._blocked_error) from exc
                log.warning("LLM call failed (attempt %s): %s", attempt + 1, str(exc)[:200])
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
        raise LLMError(str(last)[:300])


class MockLLM(BaseLLM):
    """Deterministic fake used by tests and for trying the pipeline without an API key."""

    async def _run(self, schema: type[T], system: str, user: str) -> T:
        if schema is ReviewResult or schema is RagReviewResult:
            findings = []
            for line in user.splitlines():
                m = re.match(r"^(\d+): (.*)$", line)
                if not m:
                    continue
                n, code = int(m.group(1)), m.group(2)
                if "requests.get(" in code and "timeout" not in code:
                    findings.append(dict(line=n, severity="medium", issue="HTTP request without timeout", evidence=code.strip(),
                                         reason="Requests without a timeout can hang forever.", recommendation="Pass timeout=10.", confidence=0.8))
                if "def " in code and "temp_" in code:
                    findings.append(dict(line=n, severity="low", issue="Poor function name", evidence=code.strip(),
                                         reason="Names should describe intent.", recommendation="Rename it.", confidence=0.6))
            if findings:  # one invented finding with evidence that is not in the code: must be dropped by verification
                findings.append(dict(line=1, severity="high", issue="Hallucinated issue", evidence="this text is not in the file",
                                     reason="n/a", recommendation="n/a", confidence=0.9))
            if schema is RagReviewResult:
                return RagReviewResult(findings=[RagFinding(guideline_ref=1, **f) for f in findings])  # type: ignore[return-value]
            return ReviewResult(findings=[LLMFinding(**f) for f in findings])  # type: ignore[return-value]
        if schema is CombinedReviewResult:
            findings = []
            timeout_guideline = (
                "User coding guidelines" in system and "timeout" in system.lower()
            )
            current_file = ""
            for line in user.splitlines():
                file_match = re.match(r"^File: (.+)$", line)
                if file_match:
                    current_file = file_match.group(1)
                    continue
                match = re.match(r"^(\d+): (.*)$", line)
                if not match:
                    continue
                number, code = int(match.group(1)), match.group(2)
                if "requests.get(" in code and "timeout" not in code:
                    findings.append(CategorizedLLMFinding(
                        file=current_file,
                        category="performance",
                        line=number,
                        severity="medium",
                        issue="HTTP request without timeout",
                        evidence=code.strip(),
                        reason="Requests without a timeout can hang forever.",
                        recommendation="Pass timeout=10.",
                        confidence=0.8,
                        guideline_ref=1 if timeout_guideline else None,
                    ))
            return CombinedReviewResult(findings=findings)  # type: ignore[return-value]
        if schema is DocReview:
            return DocReview(problems=["README does not explain how to run tests"], suggestions=["Add a Testing section"])  # type: ignore[return-value]
        if schema is ArchitectureInfo:
            return ArchitectureInfo(frontend="Not detected", backend="Python service", database="Not detected",
                                    authentication="Not detected", deployment="Docker", api_style="REST",
                                    components=["API layer", "Business logic"], data_flow="Requests flow from the API layer into business logic.",
                                    summary="A small Python backend packaged with Docker.")  # type: ignore[return-value]
        if schema is FinalSummary:
            return FinalSummary(summary="Mock summary of the repository.", strengths=["Has a README"],
                                top_recommendations=["Add timeouts to HTTP calls"])  # type: ignore[return-value]
        raise LLMError(f"MockLLM cannot produce {schema.__name__}")


def build_llm(settings: Settings | None = None) -> BaseLLM | None:
    """Returns None when no LLM is available (rule-based analysis only)."""
    s = settings or get_settings()
    if s.llm_mode == "mock":
        return MockLLM(s.llm_concurrency)
    if s.llm_enabled:
        return GroqLLM(s)
    return None
