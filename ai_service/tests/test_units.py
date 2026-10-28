import asyncio

import pytest
import httpx
import pytest

from app.github_client import GitHubError, _explain, parse_repo, snapshot_from_zip, RepoRef, list_profile_repositories
from app.reviewers import (
    Chunk, compute_scores, dedupe, documentation_score, make_chunks,
    select_chunks_per_file, verify_evidence,
)
from app.scanners import redact_secrets, scan_repository
from app.schemas import Finding
from app.static_checks import run_static_checks
from app.tech_detect import detect_technology


# ------------------------------------------------------------- parse_repo
@pytest.mark.parametrize("text,owner,name,branch", [
    ("https://github.com/user/project", "user", "project", None),
    ("https://github.com/user/project.git", "user", "project", None),
    ("https://github.com/user/project/", "user", "project", None),
    ("github.com/user/my-project/tree/dev/src", "user", "my-project", "dev"),
    ("user/project", "user", "project", None),
    ("http://www.github.com/a.b/c_d", "a.b", "c_d", None),
])
def test_parse_repo_ok(text, owner, name, branch):
    r = parse_repo(text)
    assert (r.owner, r.name, r.branch) == (owner, name, branch)


@pytest.mark.parametrize("text", ["", "https://gitlab.com/a/b", "not a repo", "https://github.com/onlyowner", "http://evil.com/github.com/a/b"])
def test_parse_repo_bad(text):
    with pytest.raises(GitHubError):
        parse_repo(text)


def test_explicit_branch_wins():
    assert parse_repo("user/project", branch="release").branch == "release"


def test_private_repository_404_explains_github_account_access():
    ref = RepoRef("Divyad854", "Privaterepo")
    response = httpx.Response(404, request=httpx.Request("GET", "https://api.github.com/repos/Divyad854/Privaterepo"))

    error = str(_explain(response, ref))

    assert "repository does not exist" in error
    assert "private repository" in error
    assert "authorize the GitHub account" in error


def test_groq_model_not_found_fails_without_retry(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app.llm import GroqLLM, LLMError
    from app.schemas import ReviewResult

    class ModelNotFound(Exception):
        status_code = 404

    class Chain:
        async def ainvoke(self, _input):
            self.calls += 1
            raise ModelNotFound("model_not_found")

        def __init__(self):
            self.calls = 0

    class Prompt:
        def __or__(self, _model):
            return chain

    chain = Chain()
    llm = GroqLLM.__new__(GroqLLM)
    llm._model = SimpleNamespace(
        with_structured_output=lambda _schema, **_kwargs: SimpleNamespace()
    )
    llm._prompt = Prompt()
    llm._blocked_error = None
    monkeypatch.setattr(
        "app.llm.get_settings",
        lambda: SimpleNamespace(groq_model="openai/gpt-oss-120b"),
    )

    with pytest.raises(LLMError, match="was not found or is no longer available"):
        asyncio.run(llm._run(ReviewResult, "system", "user"))
    assert chain.calls == 1


def test_groq_structured_validation_error_is_explained_without_retry():
    import asyncio
    from types import SimpleNamespace

    from app.llm import GroqLLM, LLMError
    from app.schemas import CombinedReviewResult

    class InvalidStructuredOutput(Exception):
        status_code = 400
        body = {"error": {"code": "json_validate_failed"}}

    class Chain:
        calls = 0

        async def ainvoke(self, _input):
            self.calls += 1
            raise InvalidStructuredOutput()

    chain = Chain()

    class Prompt:
        def __or__(self, _model):
            return chain

    llm = GroqLLM.__new__(GroqLLM)
    llm._model = SimpleNamespace(
        with_structured_output=lambda _schema, **_kwargs: SimpleNamespace()
    )
    llm._prompt = Prompt()
    llm._blocked_error = None

    with pytest.raises(LLMError, match="could not validate the structured review output"):
        asyncio.run(llm._run(CombinedReviewResult, "system", "user"))
    assert chain.calls == 1


def test_groq_private_repository_plain_json_skips_provider_schema_validation(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    from app.llm import GroqLLM
    from app.schemas import CombinedReviewResult

    class Chain:
        def __init__(self):
            self.calls = 0
            self.kwargs = None

        async def create(self, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"findings": []}'),
                    finish_reason="stop",
                )]
            )

    chain = Chain()

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"api_key": "test-key", "timeout": 90, "max_retries": 0}
            self.chat = SimpleNamespace(completions=chain)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr("groq.AsyncGroq", Client)
    caplog.set_level(logging.INFO, logger="repo-intel.llm")
    llm = GroqLLM.__new__(GroqLLM)
    llm._api_key = "test-key"
    llm._model_name = "openai/gpt-oss-120b"
    llm._blocked_error = None

    import asyncio

    result = asyncio.run(llm._run_json(CombinedReviewResult, "system", "user"))

    assert result.findings == []
    assert chain.calls == 1
    assert llm.provider_requests_sent == 1
    assert "Groq private review succeeded" in caplog.text
    assert chain.kwargs == {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 0.1,
        "max_tokens": 1600,
    }


def test_groq_private_json_output_is_validated_locally(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app.llm import GroqLLM, LLMError
    from app.schemas import CombinedReviewResult

    class Completions:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"unexpected": true}'))]
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr("groq.AsyncGroq", Client)
    llm = GroqLLM.__new__(GroqLLM)
    llm._api_key = "test-key"
    llm._model_name = "openai/gpt-oss-120b"
    llm._blocked_error = None

    with pytest.raises(LLMError, match="did not match the private review format"):
        asyncio.run(llm._run_json(CombinedReviewResult, "system", "user"))


def test_groq_private_json_output_reports_token_truncation(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app.llm import GroqLLM, LLMError
    from app.schemas import CombinedReviewResult

    class Completions:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"findings":'),
                    finish_reason="length",
                )]
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr("groq.AsyncGroq", Client)
    llm = GroqLLM.__new__(GroqLLM)
    llm._api_key = "test-key"
    llm._model_name = "openai/gpt-oss-120b"
    llm._blocked_error = None

    with pytest.raises(LLMError, match="truncated the private review at its output-token limit"):
        asyncio.run(llm._run_json(CombinedReviewResult, "system", "user"))


def test_groq_private_json_provider_error_identifies_missing_response_format():
    from app.llm import GroqLLM, LLMError

    class InvalidProviderRequest(Exception):
        status_code = 400
        body = {"error": {"code": "json_validate_failed"}}

    llm = GroqLLM.__new__(GroqLLM)
    llm._blocked_error = None

    with pytest.raises(LLMError, match="sent without response_format"):
        llm._raise_request_error(InvalidProviderRequest(), plain_json=True)


def test_combined_review_schema_is_valid_for_strict_structured_output():
    from app.schemas import CombinedReviewResult

    schema = CombinedReviewResult.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["findings"]
    finding_schema = schema["$defs"]["CategorizedLLMFinding"]
    assert finding_schema["additionalProperties"] is False
    assert set(finding_schema["required"]) == set(finding_schema["properties"])


def test_groq_request_too_large_fails_without_retry():
    import asyncio
    from types import SimpleNamespace

    from app.llm import GroqLLM, LLMError
    from app.schemas import ReviewResult

    class RequestTooLarge(Exception):
        response = SimpleNamespace(status_code=413)

    class Chain:
        calls = 0

        async def ainvoke(self, _input):
            self.calls += 1
            raise RequestTooLarge()

    class Prompt:
        def __or__(self, _model):
            return chain

    chain = Chain()
    llm = GroqLLM.__new__(GroqLLM)
    llm._model = SimpleNamespace(
        with_structured_output=lambda _schema, **_kwargs: SimpleNamespace()
    )
    llm._prompt = Prompt()
    llm._blocked_error = None

    with pytest.raises(LLMError, match="single review request as too large"):
        asyncio.run(llm._run(ReviewResult, "system", "user"))
    assert chain.calls == 1


def test_groq_transient_failure_is_not_retried():
    import asyncio
    from types import SimpleNamespace

    from app.llm import GroqLLM, LLMError
    from app.schemas import ReviewResult

    class NetworkFailure(Exception):
        pass

    class Chain:
        calls = 0

        async def ainvoke(self, _input):
            self.calls += 1
            raise NetworkFailure("connection failed")

    chain = Chain()

    class Prompt:
        def __or__(self, _model):
            return chain

    llm = GroqLLM.__new__(GroqLLM)
    llm._model = SimpleNamespace(
        with_structured_output=lambda _schema, **_kwargs: SimpleNamespace()
    )
    llm._prompt = Prompt()
    llm._blocked_error = None

    with pytest.raises(LLMError, match="No retry was made"):
        asyncio.run(llm._run(ReviewResult, "system", "user"))
    assert chain.calls == 1


def test_groq_rate_limit_blocks_following_requests_for_the_run():
    import asyncio
    from types import SimpleNamespace

    from app.llm import GroqLLM, LLMError
    from app.schemas import ReviewResult

    class RateLimited(Exception):
        status_code = 429

    class Chain:
        calls = 0
        structured_options = None

        async def ainvoke(self, _input):
            self.calls += 1
            raise RateLimited("tokens per minute limit exceeded")

    class Prompt:
        def __or__(self, _model):
            return chain

    chain = Chain()
    llm = GroqLLM.__new__(GroqLLM)

    def structured_output(_schema, **kwargs):
        chain.structured_options = kwargs
        return object()

    llm._model = SimpleNamespace(with_structured_output=structured_output)
    llm._prompt = Prompt()
    llm._blocked_error = None

    with pytest.raises(LLMError, match="Groq rate limit reached"):
        asyncio.run(llm._run(ReviewResult, "system", "user"))
    assert chain.structured_options == {"method": "json_schema", "strict": True}
    with pytest.raises(LLMError, match="no additional requests were sent"):
        asyncio.run(llm._run(ReviewResult, "system", "user"))
    assert chain.calls == 1


def test_settings_use_low_call_defaults(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("MAX_SOURCE_FILES", raising=False)
    monkeypatch.delenv("MAX_CHUNKS_PER_REVIEWER", raising=False)
    monkeypatch.delenv("MAX_CHUNKS_PER_FILE", raising=False)
    monkeypatch.delenv("MAX_CHUNKS_PER_REVIEW", raising=False)
    monkeypatch.delenv("LLM_CONCURRENCY", raising=False)
    monkeypatch.delenv("GROQ_TPM_LIMIT", raising=False)

    settings = Settings.from_env()

    assert settings.max_source_files == 12
    assert settings.max_chunks_per_file == 2
    assert settings.max_chunks_per_review == 5
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 100
    assert settings.llm_concurrency == 1
    assert settings.groq_tpm_limit == 8000


def test_new_chunk_limit_overrides_legacy_setting(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("MAX_CHUNKS_PER_REVIEWER", "1")
    monkeypatch.delenv("MAX_CHUNKS_PER_FILE", raising=False)

    assert Settings.from_env().max_chunks_per_file == 2


def test_chunk_selection_caps_total_and_spreads_across_files():
    from app.reviewers import Chunk, select_chunks_per_file

    chunks = [
        Chunk(f"src/file_{file_index}.py", f"{line}: code", line, 1.0)
        for file_index in range(8)
        for line in range(1, 5)
    ]

    selected = select_chunks_per_file(chunks, limit=2, max_total=5)

    assert len(selected) == 5
    assert len({chunk.path for chunk in selected}) == 5


def test_chunk_selection_reaches_minimum_when_repository_has_few_files():
    from app.reviewers import Chunk, select_chunks_per_file

    chunks = [
        Chunk("src/one.py", f"{line}: code", line, 1.0)
        for line in range(1, 8)
    ]

    selected = select_chunks_per_file(chunks, limit=2, max_total=5)

    assert len(selected) == 5


def test_profile_repository_listing_paginates_and_limits_results_to_profile_owner(monkeypatch):
    from app import github_client

    calls = []

    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    class Client:
        def __init__(self, *, headers, timeout):
            self.headers = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, *, params):
            calls.append((url, params, self.headers))
            if url.endswith("/user/repos"):
                return Response([
                    {"owner": {"login": "dev"}, "name": "private-repo"},
                    {"owner": {"login": "org"}, "name": "organization-repo"},
                ])
            page = params["page"]
            if page == 1:
                return Response([
                    {"owner": {"login": "dev"}, "name": f"repo-{n}"} for n in range(100)
                ])
            return Response([
                {"owner": {"login": "dev"}, "name": "last-repo"},
                {"owner": {"login": "org"}, "name": "collaborator-repo"},
            ])

    monkeypatch.setattr(github_client.httpx, "AsyncClient", Client)
    repos = asyncio.run(list_profile_repositories("dev"))

    assert len(repos) == 101
    assert repos[-1].full_name == "dev/last-repo"
    assert [call[1]["page"] for call in calls] == [1, 2]
    assert calls[0][0] == "https://api.github.com/users/dev/repos"
    assert "Authorization" not in calls[0][2]

    private_repos = asyncio.run(list_profile_repositories(
        "dev", include_private=True, access_token="temporary-token"
    ))
    private_call = calls[-1]
    assert [repo.full_name for repo in private_repos] == ["dev/private-repo"]
    assert private_call[0] == "https://api.github.com/user/repos"
    assert private_call[1]["visibility"] == "all"
    assert private_call[2]["Authorization"].startswith("Bearer ")

    selected = asyncio.run(list_profile_repositories("dev", repository_name="Repo-1"))
    assert [repo.full_name for repo in selected] == ["dev/repo-1"]


# ---------------------------------------------------------------- zipball
def test_snapshot_filters(snapshot):
    assert "app/main.py" in snapshot.files
    assert "node_modules/leftpad/index.js" not in snapshot.all_paths
    assert ".env" in snapshot.all_paths          # listed by path...
    assert ".env" not in snapshot.files           # ...but never read
    assert snapshot.ref.full_name == "demo-user/demo"


# ---------------------------------------------------------------- scanners
def test_secret_scanner(snapshot):
    found = scan_repository(snapshot.files, snapshot.all_paths)
    issues = {f.issue for f in found}
    assert "Hardcoded OpenAI API key" in issues
    assert "Database URL with embedded password" in issues
    assert "subprocess called with shell=True" in issues
    assert "Sensitive file committed to the repository" in issues
    for f in found:
        assert "abcdefghijklmnopqrstuvwxyz0123456789" not in f.evidence   # secrets are masked
        assert "hunter2pass" not in f.evidence


def test_redact_keeps_placeholders():
    assert "your_api_key_here" in redact_secrets('API_KEY = "your_api_key_here"')
    assert "supersecretvalue123" not in redact_secrets('password = "supersecretvalue123"')


def test_model_eval_not_flagged():
    found = scan_repository({"train.py": "model.eval()\nresult = eval(user_input)\n"}, ["train.py"])
    lines = [f.line for f in found if f.issue == "Use of eval()"]
    assert lines == [2]


# ----------------------------------------------------------- static checks
def test_static_checks_python():
    code = "def big():\n" + "    x = 1\n" * 70 + "\ndef f():\n    try:\n        pass\n    except:\n        pass\n"
    found = run_static_checks({"a.py": code}, ["a.py"])
    issues = [f.issue for f in found]
    assert any("is 71 lines long" in i for i in issues)
    assert "Bare 'except:' clause" in issues
    assert "No automated tests found" in issues


def test_static_checks_nesting():
    code = "def f(a):\n" + "".join("    " * (i + 1) + "if a:\n" for i in range(6)) + "    " * 7 + "return 1\n"
    found = run_static_checks({"a.py": code}, ["a.py", "tests/test_a.py"])
    assert any("levels of nesting" in f.issue for f in found)


# -------------------------------------------------------------- technology
def test_tech_detection(snapshot):
    tech = detect_technology(snapshot)
    assert tech["languages"][0] == {"name": "Python", "percent": 90.0}
    assert {"FastAPI", "PostgreSQL", "Docker"} <= set(tech["all"])


# ------------------------------------------------- chunking + verification
def test_chunk_line_numbers():
    content = "\n".join(f"line_{i} = {i}" for i in range(1, 400))
    chunks = make_chunks("x.py", content)
    assert len(chunks) > 1
    for c in chunks:
        first = c.text.splitlines()[0]
        n, code = first.split(": ", 1)
        assert int(n) == c.start_line
        assert code == f"line_{c.start_line} = {c.start_line}"


def test_verify_evidence_fixes_line_and_drops_hallucinations():
    src = "a = 1\nb = 2\nrequests.get(url)\n"
    good = Finding(file="x.py", line=9, category="quality", severity="low", issue="i", recommendation="r", evidence="3: requests.get(url)")
    assert verify_evidence(good, src).line == 3
    bad = Finding(file="x.py", line=1, category="quality", severity="low", issue="i", recommendation="r", evidence="not there")
    assert verify_evidence(bad, src) is None
    empty = Finding(file="x.py", line=1, category="quality", severity="low", issue="i", recommendation="r", evidence="")
    assert verify_evidence(empty, src) is None


# ----------------------------------------------------------------- scoring
def _f(sev, cat="security", conf=1.0, line=1, file="a"):
    return {"file": file, "line": line, "category": cat, "severity": sev, "issue": f"{sev}{line}", "confidence": conf, "source": "rule", "recommendation": "r"}


def test_scores():
    findings = [_f("critical"), _f("medium", "quality", line=2)]
    scores, overall = compute_scores(findings, {"security", "quality", "documentation"})
    assert scores["security"] == 75 and scores["quality"] == 95 and scores["documentation"] == 100
    assert scores["performance"] is None
    assert 0 < overall < 100


def test_documentation_findings_reduce_checklist_score_and_overall():
    finding = _f("medium", "documentation", conf=0.9)
    finding["source"] = "rag"
    evaluated = {"quality", "security", "performance", "best_practices", "documentation"}

    baseline_scores, baseline_overall = compute_scores(
        [],
        evaluated,
        {"documentation": 100},
    )
    scores, overall = compute_scores(
        [finding],
        evaluated,
        {"documentation": 100},
    )

    assert baseline_scores["documentation"] == 100
    assert scores["documentation"] == 95
    assert overall < baseline_overall


def test_guideline_findings_reduce_their_category_and_overall_only():
    finding = _f("medium", "best_practices", conf=0.9)
    finding["source"] = "rag"
    evaluated = {"quality", "security", "performance", "best_practices", "documentation"}

    baseline_scores, baseline_overall = compute_scores(
        [],
        evaluated,
        {"documentation": 100},
    )
    scores, overall = compute_scores(
        [finding],
        evaluated,
        {"documentation": 100},
    )

    assert scores["best_practices"] == 95
    assert scores["best_practices"] < baseline_scores["best_practices"]
    assert scores["quality"] == baseline_scores["quality"]
    assert overall < baseline_overall


def test_documentation_score_uses_checklist_completion():
    assert documentation_score({"readme": True, "installation": True}) == 100
    assert documentation_score({"readme": True, "installation": False}) == 50
    assert documentation_score({}) == 0


def test_dedupe_keeps_best():
    a = _f("low", conf=0.5); a["source"] = "llm"
    b = _f("high", conf=0.9)
    assert dedupe([a, b]) == [b]
