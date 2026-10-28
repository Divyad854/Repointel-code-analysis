"""LLM access: Groq through LangChain (LCEL prompt | structured model), plus a mock for tests."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import NoReturn, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings
from .schemas import (
    ArchitectureInfo, CategorizedLLMFinding, CombinedReviewResult, DocReview, FinalSummary, LLMFinding,
    RagFinding, RagReviewResult, ReviewResult,
)

log = logging.getLogger("repo-intel.llm")
T = TypeVar("T", bound=BaseModel)
GROQ_OUTPUT_TOKEN_RESERVE = 1000
PRIVATE_REVIEW_OUTPUT_TOKENS = 1600


class LLMError(Exception):
    pass


class LLMRequestTooLarge(LLMError):
    """The provider rejected the request size; no second request is made."""


class BaseLLM:
    """Interface used by the graph nodes."""

    def __init__(self, concurrency: int = 3) -> None:
        self._sem = asyncio.Semaphore(concurrency)
        self.calls = 0
        self.provider_requests_sent = 0
        self._blocked_error: str | None = None

    async def structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        plain_json: bool = False,
    ) -> T:
        async with self._sem:
            if self._blocked_error:
                raise LLMError(self._blocked_error)
            self.calls += 1
            runner = self._run_json if plain_json else self._run
            return await runner(schema, system, user)

    async def _run(self, schema: type[T], system: str, user: str) -> T:  # pragma: no cover - interface
        raise NotImplementedError

    async def _run_json(self, schema: type[T], system: str, user: str) -> T:
        return await self._run(schema, system, user)


class GroqLLM(BaseLLM):
    _OUTPUT_TOKENS_RESERVE = GROQ_OUTPUT_TOKEN_RESERVE

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.llm_concurrency)
        self._tpm_limit = max(1, settings.groq_tpm_limit)
        self._api_key = settings.groq_api_key
        self._model_name = settings.groq_model
        from langchain_groq import ChatGroq  # imported lazily so tests don't need the package configured

        self._model = ChatGroq(
            model=self._model_name,
            api_key=self._api_key,
            temperature=0.1,
            max_tokens=self._OUTPUT_TOKENS_RESERVE,
            max_retries=0,
            timeout=90,
        )
        self._prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])

    async def _check_tpm_budget(
        self,
        system: str,
        user: str,
        output_tokens: int = GROQ_OUTPUT_TOKEN_RESERVE,
    ) -> None:
        if not hasattr(self, "_tpm_limit"):
            return
        estimated_tokens = math.ceil((len(system) + len(user)) / 2) + output_tokens
        if estimated_tokens > self._tpm_limit:
            raise LLMRequestTooLarge(
                "Groq rejected the single review request as too large for the configured tokens-per-minute limit."
            )

    async def _run(self, schema: type[T], system: str, user: str) -> T:
        return await self._run_with_method(schema, system, user, "json_schema")

    async def _run_json(self, schema: type[T], system: str, user: str) -> T:
        if self._blocked_error:
            raise LLMError(self._blocked_error)
        from groq import AsyncGroq

        try:
            await self._check_tpm_budget(
                system,
                user,
                output_tokens=PRIVATE_REVIEW_OUTPUT_TOKENS,
            )
            log.info("Sending private repository review without a provider response format.")
            async with AsyncGroq(
                api_key=self._api_key,
                timeout=90,
                max_retries=0,
            ) as client:
                self.provider_requests_sent = getattr(self, "provider_requests_sent", 0) + 1
                response = await client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1,
                    max_tokens=PRIVATE_REVIEW_OUTPUT_TOKENS,
                )
        except LLMRequestTooLarge:
            raise
        except Exception as exc:
            self._raise_request_error(exc, plain_json=True)

        choice = response.choices[0] if response.choices else None
        if choice and getattr(choice, "finish_reason", None) == "length":
            raise LLMError(
                "Groq truncated the private review at its output-token limit; "
                "the report uses rule-based fallback."
            )
        content = choice.message.content if choice else None
        if not isinstance(content, str):
            raise LLMError("Groq returned no text for the private repository review.")
        content = content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("Groq private review response was not valid JSON.")
            raise LLMError(
                "Groq returned malformed or truncated JSON for the private review; "
                "the report uses rule-based fallback."
            ) from exc
        try:
            result = schema.model_validate(data)
        except ValidationError as exc:
            log.warning("Groq private review response did not match the expected review format.")
            raise LLMError(
                "Groq returned JSON that did not match the private review format; "
                "the report uses rule-based fallback."
            ) from exc
        log.info(
            "Groq private review succeeded (model=%s, findings=%d, provider_requests_sent=%d).",
            self._model_name,
            len(getattr(result, "findings", [])),
            self.provider_requests_sent,
        )
        return result

    async def _run_with_method(
        self,
        schema: type[T],
        system: str,
        user: str,
        method: str,
    ) -> T:
        if self._blocked_error:
            raise LLMError(self._blocked_error)
        options: dict[str, str | bool] = {"method": method}
        if method == "json_schema":
            options["strict"] = True
        chain = self._prompt | self._model.with_structured_output(schema, **options)
        try:
            await self._check_tpm_budget(system, user)
            self.provider_requests_sent = getattr(self, "provider_requests_sent", 0) + 1
            result = await chain.ainvoke({"system": system, "user": user})
            if result is None:
                raise LLMError("empty structured response")
            log.info(
                "Groq structured review succeeded (model=%s, findings=%d, provider_requests_sent=%d).",
                self._model_name,
                len(getattr(result, "findings", [])),
                self.provider_requests_sent,
            )
            return result
        except LLMRequestTooLarge:
            raise
        except Exception as exc:  # network, rate limit, validation...
            self._raise_request_error(exc)

    def _raise_request_error(
        self,
        exc: Exception,
        *,
        plain_json: bool = False,
    ) -> NoReturn:
        response = getattr(exc, "response", None)
        status_code = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        body = getattr(exc, "body", None)
        error_code = (
            body.get("error", {}).get("code")
            if isinstance(body, dict) and isinstance(body.get("error"), dict)
            else None
        )
        if status_code == 429:
            self._blocked_error = (
                "Groq rate limit reached. One AI request was used for this analysis; "
                "no additional requests were sent."
            )
            log.warning("%s", self._blocked_error)
            raise LLMError(self._blocked_error) from exc
        if status_code == 413:
            log.warning(
                "Provider rejected the single review request (HTTP %s, code=%s).",
                status_code,
                error_code or "unspecified",
            )
            raise LLMRequestTooLarge(
                "Groq rejected the single review request as too large."
            ) from exc
        if status_code in (400, 401, 403, 404, 422):
            if status_code == 404:
                self._blocked_error = (
                    f"Groq model {get_settings().groq_model!r} was not found or is no longer available. "
                    "Set GROQ_MODEL to an active model from https://console.groq.com/docs/models."
                )
            elif status_code in (400, 422):
                if error_code == "json_validate_failed":
                    if plain_json:
                        self._blocked_error = (
                            "Groq rejected a private review request sent without response_format "
                            "(HTTP 400, code=json_validate_failed). Confirm the AI service is running "
                            "the latest code and check the configured Groq model."
                        )
                    else:
                        self._blocked_error = (
                            "Groq could not validate the structured review output. "
                            "The report uses rule-based fallback; check the model's structured-output support."
                        )
                else:
                    self._blocked_error = (
                        "Groq rejected the model request. Check model capabilities and request settings."
                    )
            else:
                self._blocked_error = (
                    "Groq authorization failed. Check the configured API key and account access."
                )
            log.warning(
                "Stopping LLM calls for this run after provider rejection (HTTP %s, code=%s).",
                status_code,
                error_code or "unspecified",
            )
            raise LLMError(self._blocked_error) from exc
        log.warning("Single LLM call failed: %s", str(exc)[:200])
        detail = f"HTTP {status_code}" if status_code else type(exc).__name__
        raise LLMError(
            f"Groq request failed ({detail}). No retry was made; check AI-service logs for details."
        ) from exc


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
    """Returns None when an AI reviewer is not configured."""
    s = settings or get_settings()
    if s.llm_mode == "mock":
        return MockLLM(s.llm_concurrency)
    if s.llm_enabled:
        return GroqLLM(s)
    return None
