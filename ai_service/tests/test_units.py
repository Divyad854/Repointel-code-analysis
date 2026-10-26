import asyncio

import pytest
import httpx
import pytest

from app.github_client import GitHubError, _explain, parse_repo, snapshot_from_zip, RepoRef, list_profile_repositories
from app.reviewers import Chunk, make_chunks, verify_evidence, compute_scores, dedupe
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
    assert chain.structured_options == {"method": "json_schema", "strict": False}
    with pytest.raises(LLMError, match="Remaining LLM reviews were skipped"):
        asyncio.run(llm._run(ReviewResult, "system", "user"))
    assert chain.calls == 1


def test_settings_use_low_call_defaults(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("MAX_SOURCE_FILES", raising=False)
    monkeypatch.delenv("MAX_CHUNKS_PER_REVIEWER", raising=False)
    monkeypatch.delenv("LLM_CONCURRENCY", raising=False)

    settings = Settings.from_env()

    assert settings.max_source_files == 10
    assert settings.max_chunks_per_reviewer == 1
    assert settings.llm_concurrency == 1


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


def test_dedupe_keeps_best():
    a = _f("low", conf=0.5); a["source"] = "llm"
    b = _f("high", conf=0.9)
    assert dedupe([a, b]) == [b]
