import asyncio

import pytest
from fastapi.testclient import TestClient

from app import github_client, rag
from app.graph import build_graph, STEP_LABELS
from app.llm import MockLLM


@pytest.fixture(autouse=True)
def fake_github(monkeypatch, snapshot):
    async def fake_fetch(ref, token=None):
        if ref.name == "missing":
            raise github_client.GitHubError("Repository demo/missing was not found.")
        return snapshot
    monkeypatch.setattr(github_client, "fetch_snapshot", fake_fetch)


async def run(llm, user_id=1, url="https://github.com/demo-user/demo"):
    graph = build_graph()
    steps, final = [], {}
    async for upd in graph.astream({"repo_url": url, "user_id": user_id}, config={"configurable": {"llm": llm}}, stream_mode="updates"):
        for node, vals in upd.items():
            steps.append(node)
            final.update(vals or {})
    return steps, final


def test_graph_full_run_with_mock_llm():
    steps, final = asyncio.run(run(MockLLM()))
    assert steps[0] == "fetch_repository" and steps[-1] == "generate_report"
    assert set(STEP_LABELS) == set(steps)
    report = final["report"]
    issues = {(f["issue"], f["source"]) for f in report["findings"]}
    assert ("HTTP request without timeout", "llm") in issues
    assert not any(f["issue"] == "Hallucinated issue" for f in report["findings"])   # evidence check dropped it
    assert ("Hardcoded OpenAI API key", "rule") in issues
    assert report["scores"]["security"] < 100
    assert all(report["scores"][k] is not None for k in report["scores"])
    assert report["technology"]["languages"][0]["name"] == "Python"
    assert report["architecture"]["backend"] == "FastAPI"
    assert report["summary"] and report["top_recommendations"]
    # no secret leaked anywhere in the report
    assert "abcdefghijklmnopqrstuvwxyz0123456789" not in str(report)
    assert "hunter2pass" not in str(report)
    # every finding with a line number points at a real line
    assert report["stats"]["llm_calls"] == 1


def test_graph_without_llm_uses_rules_only():
    _, final = asyncio.run(run(None))
    report = final["report"]
    assert report["scores"]["performance"] is None and report["scores"]["best_practices"] is None
    assert report["scores"]["security"] is not None
    assert any("No LLM was used for this run" in w for w in report["warnings"])
    assert report["findings"]


def test_graph_rate_limit_uses_one_clear_note_and_marks_failed_categories_unscored():
    from app.llm import BaseLLM, LLMError

    class LimitedLLM(BaseLLM):
        async def _run(self, schema, system, user):
            raise LLMError("Groq rate limit reached. Remaining LLM reviews were skipped for this run.")

    _, final = asyncio.run(run(LimitedLLM(concurrency=1)))
    report = final["report"]
    provider_notes = [warning for warning in report["warnings"] if "Groq" in warning]

    assert len(provider_notes) == 1
    assert "rate limit" in provider_notes[0]
    assert report["scores"]["performance"] is None
    assert report["scores"]["best_practices"] is None
    assert "First error:" not in str(report["warnings"])


def test_graph_error_branch_ends_early():
    steps, final = asyncio.run(run(MockLLM(), url="demo/missing"))
    assert steps == ["fetch_repository"]
    assert "not found" in final["error"]


def test_rag_roundtrip_and_graph_uses_guidelines():
    text = b"Always set a timeout on every HTTP request made with the requests library.\n" * 3
    n = rag.index_guideline("g-1", 7, "Team Python Guide", "guide.md", text)
    assert n >= 1 and rag.has_guidelines(7) and not rag.has_guidelines(8)
    assert len(rag.get_user_guidelines(7)) == n
    assert rag.get_user_guidelines(8) == []
    docs = rag.retrieve(7, "requests timeout http", 2)
    assert docs and docs[0].metadata["title"] == "Team Python Guide"
    assert rag.retrieve(8, "requests timeout", 2) == []          # isolated per user

    _, final = asyncio.run(run(MockLLM(), user_id=7))
    report = final["report"]
    assert report["guidelines_used"] is True
    assert report["stats"]["llm_calls"] == 1
    assert "Team Python Guide, p.1" in report["guideline_sources"]
    assert report["stats"]["llm_calls"] == 1

    rag.delete_guideline("g-1")
    assert not rag.has_guidelines(7)


def test_graph_checks_all_user_guidelines_and_applies_violation_to_report(monkeypatch):
    from types import SimpleNamespace

    from app.llm import BaseLLM
    from app.schemas import CategorizedLLMFinding, CombinedReviewResult

    docs = [
        SimpleNamespace(
            page_content="Function names must describe their purpose; avoid temporary names.",
            metadata={"title": "React Express guide", "page": 1},
        ),
        SimpleNamespace(
            page_content="Route handlers must validate user input.",
            metadata={"title": "API requirements", "page": 3},
        ),
    ]
    monkeypatch.setattr(rag, "get_user_guidelines", lambda _user_id: docs)

    class GuidelineAwareLLM(BaseLLM):
        async def _run(self, schema, system, user):
            assert schema is CombinedReviewResult
            assert docs[0].page_content in system
            assert docs[1].page_content in system
            assert "File: app/main.py" in user
            return CombinedReviewResult(findings=[CategorizedLLMFinding(
                file="app/main.py",
                line=7,
                category="best_practices",
                severity="medium",
                issue="Function name does not describe its purpose",
                evidence="def temp_fetch(url):",
                reason="The function uses a temporary name, contrary to the uploaded naming guideline.",
                recommendation="Rename it to describe the data it fetches.",
                confidence=0.9,
                guideline_ref=1,
            )])

    llm = GuidelineAwareLLM(concurrency=1)
    _, final = asyncio.run(run(llm, user_id=21))
    report = final["report"]

    assert llm.calls == report["stats"]["llm_calls"] == 1
    assert report["guidelines_used"] is True
    assert report["guideline_sources"] == ["API requirements, p.3", "React Express guide, p.1"]
    assert report["scores"]["best_practices"] < 100
    assert any(
        finding["source"] == "rag"
        and finding["category"] == "best_practices"
        and finding["guideline"] == "React Express guide, p.1"
        for finding in report["findings"]
    )


def test_graph_reports_guidelines_as_not_reviewed_when_llm_is_unavailable(monkeypatch):
    monkeypatch.setattr(rag, "has_guidelines", lambda _user_id: True)

    _, final = asyncio.run(run(None, user_id=19))

    report = final["report"]
    assert report["guidelines_used"] is False
    assert any(
        "Uploaded coding guidelines were not reviewed because AI review is unavailable."
        in warning
        for warning in report["warnings"]
    )


def test_graph_warns_when_guideline_store_cannot_be_read(monkeypatch):
    def fail_lookup(_user_id):
        raise RuntimeError("guideline store unavailable")

    monkeypatch.setattr(rag, "get_user_guidelines", fail_lookup)

    _, final = asyncio.run(run(MockLLM(), user_id=20))

    report = final["report"]
    assert report["guidelines_used"] is False
    assert any(
        "Uploaded guidelines could not be loaded for this run." in warning
        for warning in report["warnings"]
    )


def test_combined_guideline_review_checks_multiple_files_in_one_call():
    from types import SimpleNamespace

    from app.reviewers import Chunk, review_combined_chunks
    from app.schemas import CategorizedLLMFinding, CombinedReviewResult
    from app.llm import BaseLLM

    class MultiFileLLM(BaseLLM):
        async def _run(self, schema, system, user):
            assert schema is CombinedReviewResult
            assert "All uploaded coding-guideline passages" in system
            assert "File: client/src/App.jsx" in user
            assert "File: server/routes.js" in user
            return CombinedReviewResult(findings=[CategorizedLLMFinding(
                file="server/routes.js",
                line=7,
                category="best_practices",
                severity="medium",
                issue="Missing endpoint validation",
                evidence="app.get('/users', handler)",
                reason="The supplied guide requires validating query inputs.",
                recommendation="Validate the request query before processing it.",
                confidence=0.9,
                guideline_ref=1,
            )])

    chunks = [
        Chunk("client/src/App.jsx", "4: export default function App() { return <main />; }", 4, 9.0),
        Chunk("server/routes.js", "7: app.get('/users', handler)", 7, 8.0),
    ]
    files = {
        "client/src/App.jsx": "export default function App() { return <main />; }",
        "server/routes.js": "\n".join(["// route setup"] * 6 + ["app.get('/users', handler)"]),
    }
    document = SimpleNamespace(page_content="Validate query parameters on every endpoint.", metadata={"title": "React Express guide", "page": 1})
    llm = MultiFileLLM(concurrency=1)

    findings, error, succeeded, sources = asyncio.run(review_combined_chunks(
        llm,
        chunks,
        files,
        ["React", "Express"],
        [document],
        lambda doc: f"{doc.metadata['title']}, p.{doc.metadata['page']}",
    ))

    assert error is None and succeeded
    assert llm.calls == 1
    assert len(findings) == 1
    assert findings[0].file == "server/routes.js"
    assert findings[0].line == 7
    assert findings[0].guideline == "React Express guide, p.1"
    assert sources == ["React Express guide, p.1"]


def test_combined_review_lists_guidelines_checked_even_when_no_violation_found():
    from types import SimpleNamespace

    from app.llm import BaseLLM
    from app.reviewers import Chunk, review_combined_chunks
    from app.schemas import CombinedReviewResult

    class NoViolationLLM(BaseLLM):
        async def _run(self, schema, system, user):
            assert "Validate request parameters." in system
            assert "File: server/routes.js" in user
            return CombinedReviewResult(findings=[])

    docs = [
        SimpleNamespace(
            page_content="Validate request parameters.",
            metadata={"title": "React Express guide", "page": 2},
        ),
        SimpleNamespace(
            page_content="Use descriptive names for route handlers.",
            metadata={"title": "Team naming rules", "page": 1},
        ),
    ]
    findings, error, succeeded, sources = asyncio.run(review_combined_chunks(
        NoViolationLLM(concurrency=1),
        [Chunk("server/routes.js", "7: router.get('/users', handler)", 7, 8.0)],
        {"server/routes.js": "router.get('/users', handler)"},
        ["React", "Express"],
        docs,
        lambda item: f"{item.metadata['title']}, p.{item.metadata['page']}",
    ))

    assert findings == []
    assert error is None and succeeded
    assert sources == ["React Express guide, p.2", "Team naming rules, p.1"]


def test_api_flow(monkeypatch):
    from app.main import app
    import app.main as main

    async def ensure_public(_ref):
        return None

    monkeypatch.setattr(main, "ensure_public_repository", ensure_public)
    headers = {"X-Internal-Token": "test-token"}
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert "llm_model" in health
        assert "max_chunks_per_reviewer" in health
        assert client.post("/analyze", json={"repo_url": "x/y", "user_id": 1}).status_code == 401
        assert client.post("/analyze", json={"repo_url": "garbage", "user_id": 1}, headers=headers).status_code == 422
        r = client.post("/analyze", json={"repo_url": "demo-user/demo", "user_id": 3}, headers=headers)
        assert r.status_code == 202
        job = client.get(f"/jobs/{r.json()['job_id']}", headers=headers).json()
        assert job["status"] == "done" and job["report"]["overall_score"] >= 0
        assert client.get("/jobs/nope", headers=headers).status_code == 404

        bad = client.post("/analyze", json={"repo_url": "demo/missing", "user_id": 3}, headers=headers).json()["job_id"]
        job = client.get(f"/jobs/{bad}", headers=headers).json()
        assert job["status"] == "failed" and "not found" in job["error"]

        up = client.post("/guidelines", headers=headers, data={"guideline_id": "g-2", "user_id": "3", "title": "Sec"},
                         files={"file": ("sec.txt", b"Never store passwords in source code.", "text/plain")})
        assert up.status_code == 200 and up.json()["chunks"] == 1
        assert client.post("/guidelines", headers=headers, data={"guideline_id": "g-3", "user_id": "3"},
                           files={"file": ("x.exe", b"zzz", "application/octet-stream")}).status_code == 422
        assert client.delete("/guidelines/g-2", headers=headers).status_code == 200


def test_single_repository_api_requires_github_authorization_for_private_repo(monkeypatch):
    from app import github_client
    import app.main as main

    async def private_repository(_ref):
        raise github_client.GitHubAuthorizationRequired("Repository demo/private is private.")

    monkeypatch.setattr(main, "ensure_public_repository", private_repository)
    headers = {"X-Internal-Token": "test-token"}
    with TestClient(main.app) as client:
        response = client.post("/analyze", headers=headers, json={
            "repo_url": "https://github.com/demo/private",
            "user_id": 3,
        })

    assert response.status_code == 401
    assert "GitHub authorization is required" in response.json()["detail"]


def test_profile_api_analyzes_public_and_private_and_revokes_private_grant(monkeypatch):
    from app import jobs
    from app.github_client import RepoRef
    from types import SimpleNamespace
    import app.main as main

    revoked = []

    async def fake_list(username, *, include_private=False, access_token=None, repository_name=None):
        assert username == "demo-user"
        if include_private:
            assert access_token == "one-run-token"
        else:
            assert not access_token
        return [RepoRef("demo-user", "demo")]

    async def fake_revoke(request):
        revoked.append(request.github_access_token)

    monkeypatch.setattr(jobs, "list_profile_repositories", fake_list)
    monkeypatch.setattr(jobs, "_revoke_profile_grant", fake_revoke)
    monkeypatch.setattr(jobs, "build_llm", lambda: None)
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
        internal_token="test-token",
        github_oauth_client_id="client-id",
        github_oauth_client_secret="client-secret",
    ))
    headers = {"X-Internal-Token": "test-token"}

    with TestClient(main.app) as client:
        public = client.post("/analyze-profile", headers=headers, json={
            "username": "demo-user", "user_id": 4,
        })
        assert public.status_code == 202
        public_job = client.get(f"/jobs/{public.json()['job_id']}", headers=headers).json()
        assert public_job["status"] == "done"
        assert len(public_job["report"]["repositories"]) == 1
        assert not revoked

        private = client.post("/analyze-profile", headers=headers, json={
            "username": "demo-user", "user_id": 4, "include_private": True,
            "github_access_token": "one-run-token", "repository_name": "demo",
        })
        assert private.status_code == 202
        private_job = client.get(f"/jobs/{private.json()['job_id']}", headers=headers).json()
        assert private_job["status"] == "done"
        assert revoked == ["one-run-token"]
        assert "one-run-token" not in str(jobs.JOBS[private.json()["job_id"]])

        missing_credentials = client.post("/analyze-profile", headers=headers, json={
            "username": "demo-user", "user_id": 4, "include_private": True,
        })
        assert missing_credentials.status_code == 422

        monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
            internal_token="test-token",
            github_oauth_client_id="",
            github_oauth_client_secret="",
        ))
        unconfigured_revocation = client.post("/analyze-profile", headers=headers, json={
            "username": "demo-user", "user_id": 4, "include_private": True,
            "github_access_token": "one-run-token",
        })
        assert unconfigured_revocation.status_code == 503
        assert "GITHUB_OAUTH_CLIENT_ID" in unconfigured_revocation.json()["detail"]


def test_single_private_repository_job_revokes_grant_after_completion(monkeypatch):
    from app import jobs
    from app.schemas import AnalyzeRequest

    revoked = []

    async def fake_revoke(token):
        revoked.append(token)

    monkeypatch.setattr(jobs, "_revoke_github_grant", fake_revoke)
    monkeypatch.setattr(jobs, "build_llm", lambda: None)
    job_id = jobs.create_job()
    request = AnalyzeRequest(
        repo_url="https://github.com/demo-user/demo",
        github_access_token="single-run-token",
        user_id=8,
    )

    asyncio.run(jobs.run_job(job_id, request))

    assert jobs.JOBS[job_id]["status"] == "done"
    assert revoked == ["single-run-token"]
    assert request.github_access_token is None


def test_single_private_repository_failure_still_revokes_grant(monkeypatch):
    from app import jobs
    from app.schemas import AnalyzeRequest

    revoked = []

    async def fake_revoke(token):
        revoked.append(token)

    monkeypatch.setattr(jobs, "_revoke_github_grant", fake_revoke)
    monkeypatch.setattr(jobs, "build_llm", lambda: None)
    job_id = jobs.create_job()
    request = AnalyzeRequest(
        repo_url="https://github.com/demo-user/missing",
        github_access_token="single-run-token",
        user_id=8,
    )

    asyncio.run(jobs.run_job(job_id, request))

    assert jobs.JOBS[job_id]["status"] == "failed"
    assert "not found" in jobs.JOBS[job_id]["error"]
    assert revoked == ["single-run-token"]
    assert request.github_access_token is None


def test_single_private_repository_keeps_completed_report_when_revocation_fails(monkeypatch):
    from app import jobs
    from app.schemas import AnalyzeRequest

    async def fail_revoke(_token):
        raise RuntimeError("GitHub rejected grant revocation")

    monkeypatch.setattr(jobs, "_revoke_github_grant", fail_revoke)
    monkeypatch.setattr(jobs, "build_llm", lambda: MockLLM())
    job_id = jobs.create_job()
    request = AnalyzeRequest(
        repo_url="https://github.com/demo-user/demo",
        github_access_token="single-run-token",
        user_id=8,
    )

    asyncio.run(jobs.run_job(job_id, request))

    job = jobs.JOBS[job_id]
    assert job["status"] == "done"
    assert job["report"]
    assert any("could not be revoked automatically" in warning for warning in job["report"]["warnings"])
    assert not job["error"]
    assert request.github_access_token is None


def test_github_revoke_reports_authentication_status_without_retrying(monkeypatch):
    from types import SimpleNamespace
    from app import jobs

    class FakeResponse:
        status_code = 401

    class FakeClient:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def delete(self, *_args, **_kwargs):
            self.calls += 1
            return FakeResponse()

    client = FakeClient()
    monkeypatch.setattr(jobs.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(
        github_oauth_client_id="client-id",
        github_oauth_client_secret="client-secret",
    ))

    with pytest.raises(RuntimeError, match="HTTP 401"):
        asyncio.run(jobs._revoke_github_grant("one-run-token"))

    assert client.calls == 1
