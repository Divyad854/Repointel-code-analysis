import re
from unittest import mock

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse

from . import services
from .models import Analysis, Finding, Guideline
from .services import AIServiceError, JobNotFound, PrivateRepositoryAuthRequired, parse_repo_url

User = get_user_model()

REPORT = {
    "repository": {"owner": "u", "name": "demo", "url": "https://github.com/u/demo", "branch": "main", "description": "Demo repo", "stars": 3},
    "overall_score": 72,
    "scores": {"quality": 83, "security": 37, "performance": None, "best_practices": None, "documentation": 82},
    "severity_counts": {"critical": 1, "high": 1, "medium": 1, "low": 0},
    "category_counts": {"quality": 1, "security": 2, "performance": 0, "best_practices": 0, "documentation": 0},
    "technology": {"languages": [{"name": "Python", "percent": 90.0}, {"name": "HTML", "percent": 10.0}],
                   "technologies": {"frameworks": ["FastAPI"], "databases": ["PostgreSQL"]}, "all": ["FastAPI", "PostgreSQL"]},
    "architecture": {"frontend": "Not detected", "backend": "FastAPI", "database": "PostgreSQL", "authentication": "Not detected",
                     "deployment": "Docker", "api_style": "REST", "components": ["API"], "data_flow": "Client -> API -> DB", "summary": "A FastAPI service."},
    "doc_checklist": {"readme": True, "description": True, "installation": False, "usage": True, "environment_variables": False,
                      "api_docs": False, "tests": False, "license": True},
    "summary": "Overall score 72/100.",
    "strengths": ["Has a README"],
    "top_recommendations": ["Rotate the exposed key"],
    "findings": [
        {"file": "app/main.py", "line": 4, "category": "security", "severity": "critical", "issue": "Hardcoded OpenAI API key",
         "evidence": 'API_KEY = "sk-a****"', "reason": "Anyone can use it.", "recommendation": "Use env vars.", "confidence": 0.95, "source": "llm", "guideline": None},
        {"file": "app/main.py", "line": 8, "category": "quality", "severity": "high", "issue": "Bare except",
         "evidence": "except:", "reason": "Hides bugs.", "recommendation": "Catch specific errors.", "confidence": 0.9, "source": "rag", "guideline": "Team guide, p.2"},
        {"file": "README.md", "line": None, "category": "security", "severity": "medium", "issue": "Something",
         "evidence": "", "reason": "", "recommendation": "Fix it.", "confidence": 0.5, "source": "llm"},
    ],
    "files_reviewed": ["app/main.py"], "guidelines_used": True, "guideline_passages_reviewed": 3,
    "guideline_sources": ["Team guide, p.2"], "ai_reviewed_chunks": [
        {"file": "app/main.py", "start_line": 1, "end_line": 25},
        {"file": "app/main.py", "start_line": 40, "end_line": 65},
    ], "warnings": ["A warning"],
    "stats": {
        "files_in_repo": 5, "files_reviewed": 1, "chunks_available": 2,
        "chunks_submitted_to_ai": 2, "ai_review_completed": True,
        "ai_findings": 3, "rule_findings": 0, "llm_calls": 1,
        "provider_requests_sent": 1, "rule_files_scanned": 5, "duration_s": 12.3,
    },
}


class Base(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("divya", "d@example.com", "pass12345!")
        self.client.login(username="divya", password="pass12345!")

    def make(self, **kw):
        defaults = dict(user=self.user, repo_url="https://github.com/u/demo", owner="u", repo_name="demo", job_id="job1")
        defaults.update(kw)
        return Analysis.objects.create(**defaults)


class ParseTests(TestCase):
    def test_parse(self):
        self.assertEqual(parse_repo_url("https://github.com/a/b.git"), ("a", "b", ""))
        self.assertEqual(parse_repo_url("github.com/a/b/tree/dev/x"), ("a", "b", "dev"))
        self.assertEqual(parse_repo_url("a/b"), ("a", "b", ""))
        for bad in ("", "https://gitlab.com/a/b", "nonsense", "https://github.com/a"):
            with self.assertRaises(ValueError):
                parse_repo_url(bad)


class ProfileServiceTests(TestCase):
    @override_settings(
        GITHUB_OAUTH_CLIENT_ID="oauth-client",
        GITHUB_OAUTH_CLIENT_SECRET="oauth-secret",
    )
    def test_public_profile_request_has_no_credentials_and_private_request_has_one_run_credentials(self):
        response = mock.Mock(status_code=202)
        response.json.return_value = {"job_id": "profile-job"}
        with mock.patch("reviews.services.requests.request", return_value=response) as request:
            self.assertEqual(
                services.start_profile_analysis("dev", "", 7, include_private=False, repository_name="repo"),
                "profile-job",
            )
            public_payload = request.call_args.kwargs["json"]
            self.assertIsNone(public_payload["github_access_token"])
            self.assertNotIn("oauth_client_secret", public_payload)
            self.assertEqual(public_payload["repository_name"], "repo")

            self.assertEqual(
                services.start_profile_analysis(
                    "dev", "one-run-token", 7, include_private=True, repository_name="repo"
                ),
                "profile-job",
            )
            private_payload = request.call_args.kwargs["json"]
            self.assertEqual(private_payload["github_access_token"], "one-run-token")
            self.assertNotIn("oauth_client_secret", private_payload)
            self.assertEqual(private_payload["repository_name"], "repo")


class AuthTests(TestCase):
    def test_pages_require_login(self):
        home = self.client.get(reverse("dashboard"))
        self.assertEqual(home.status_code, 200)
        self.assertContains(home, "Understand your GitHub repository")
        self.assertContains(home, "Create your account")
        for name in ("analyze", "history", "guidelines", "profile", "password_change"):
            r = self.client.get(reverse(name))
            self.assertEqual(r.status_code, 302, name)
            self.assertIn("/login/", r["Location"])

    @override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
    def test_register_login_logout_after_email_verification(self):
        mail.outbox.clear()
        r = self.client.post(reverse("register"), {"username": "new", "email": "n@example.com", "password1": "Str0ngPass!!", "password2": "Str0ngPass!!"})
        self.assertRedirects(r, reverse("verify_registration"))
        self.assertFalse(User.objects.filter(username="new").exists())
        code = re.search(r"\b(\d{6})\b", mail.outbox[-1].body).group(1)
        r = self.client.post(reverse("verify_registration"), {"code": code})
        self.assertRedirects(r, reverse("dashboard"))
        self.assertTrue(User.objects.filter(username="new", email="n@example.com").exists())
        self.assertEqual(self.client.get(reverse("dashboard")).status_code, 200)
        self.client.post(reverse("logout"))
        self.assertContains(self.client.get(reverse("dashboard")), "Understand your GitHub repository")
        r = self.client.post(reverse("login"), {"username": "new", "password": "Str0ngPass!!"})
        self.assertRedirects(r, reverse("dashboard"))

    def test_duplicate_email_rejected(self):
        User.objects.create_user("a", "dup@example.com", "x")
        r = self.client.post(reverse("register"), {"username": "b", "email": "DUP@example.com", "password1": "Str0ngPass!!", "password2": "Str0ngPass!!"})
        self.assertContains(r, "already exists")

    def test_password_change_updates_password_and_keeps_user_logged_in(self):
        user = User.objects.create_user("password-user", "password@example.com", "OriginalPass123!")
        self.client.login(username="password-user", password="OriginalPass123!")

        page = self.client.get(reverse("password_change"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Change password")
        self.assertContains(page, "old_password")

        response = self.client.post(reverse("password_change"), {
            "old_password": "OriginalPass123!",
            "new_password1": "NewSecurePass456!",
            "new_password2": "NewSecurePass456!",
        })

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("profile"))
        self.assertContains(self.client.get(reverse("profile")), "Your password has been changed.")
        user.refresh_from_db()
        self.assertTrue(user.check_password("NewSecurePass456!"))
        self.assertEqual(self.client.get(reverse("profile")).status_code, 200)

    def test_password_change_rejects_incorrect_current_password(self):
        user = User.objects.create_user("incorrect-password-user", "incorrect@example.com", "OriginalPass123!")
        self.client.login(username="incorrect-password-user", password="OriginalPass123!")
        response = self.client.post(reverse("password_change"), {
            "old_password": "incorrect",
            "new_password1": "NewSecurePass456!",
            "new_password2": "NewSecurePass456!",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your old password was entered incorrectly")


class AnalyzeTests(Base):
    def test_repository_analyzer_offers_public_and_private_modes(self):
        response = self.client.get(reverse("analyze"))
        self.assertContains(response, "Public repository")
        self.assertContains(response, "Private repository")
        self.assertContains(response, "GitHub repository URL")
        self.assertNotContains(response, "Choose what to analyze")
        self.assertNotContains(response, "GitHub profile URL")

    def test_invalid_url(self):
        r = self.client.post(reverse("analyze"), {"repo_url": "https://example.com/x", "analysis_type": "public"})
        self.assertContains(r, "Enter a repository URL like")
        self.assertEqual(Analysis.objects.count(), 0)

    def test_repository_url_is_normalized_to_owner_and_name(self):
        from .forms import AnalyzeForm

        form = AnalyzeForm(data={
            "repo_url": "https://github.com/Divyad854/Privaterepo",
            "analysis_type": "public",
        })

        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["repo_url"], "https://github.com/Divyad854/Privaterepo")
        self.assertEqual((form.owner, form.name), ("Divyad854", "Privaterepo"))

    @mock.patch("reviews.views.start_analysis", return_value="job-123")
    def test_start(self, start):
        r = self.client.post(reverse("analyze"), {"repo_url": "https://github.com/u/demo", "analysis_type": "public"})
        a = Analysis.objects.get()
        self.assertRedirects(r, reverse("report", args=[a.pk]))
        self.assertEqual((a.owner, a.repo_name, a.is_profile, a.job_id, a.status), ("u", "demo", False, "job-123", "queued"))
        start.assert_called_once_with("https://github.com/u/demo", "", "", self.user.id)

    @mock.patch("reviews.views.start_analysis", return_value="job-specific")
    def test_public_analysis_passes_only_the_selected_repository(self, start):
        response = self.client.post(reverse("analyze"), {
            "repo_url": "https://github.com/u/specific-repo",
            "analysis_type": "public",
        })

        analysis = Analysis.objects.get()
        self.assertRedirects(response, reverse("report", args=[analysis.pk]))
        self.assertEqual((analysis.owner, analysis.repo_name), ("u", "specific-repo"))
        start.assert_called_once_with("https://github.com/u/specific-repo", "", "", self.user.id)

    @mock.patch("reviews.views.start_analysis")
    def test_private_mode_starts_oauth_instead_of_calling_ai_service(self, start):
        response = self.client.post(reverse("analyze"), {
            "repo_url": "https://github.com/u/private-repo", "analysis_type": "private",
        })
        self.assertRedirects(response, reverse("github_connect"), fetch_redirect_response=False)
        self.assertEqual(self.client.session["pending_private_repo_url"], "https://github.com/u/private-repo")
        start.assert_not_called()

    @mock.patch(
        "reviews.views.start_analysis",
        side_effect=PrivateRepositoryAuthRequired("GitHub authorization is required."),
    )
    def test_public_mode_private_repository_is_redirected_to_oauth(self, start):
        response = self.client.post(reverse("analyze"), {
            "repo_url": "https://github.com/u/private-repo", "analysis_type": "public",
        })

        self.assertRedirects(response, reverse("github_connect"), fetch_redirect_response=False)
        self.assertEqual(
            self.client.session["pending_private_repo_url"],
            "https://github.com/u/private-repo",
        )
        start.assert_called_once()

    @mock.patch("reviews.views.start_analysis", side_effect=AIServiceError("The AI service is not reachable right now."))
    def test_service_down(self, _):
        r = self.client.post(reverse("analyze"), {
            "repo_url": "https://github.com/u/demo", "analysis_type": "public",
        })
        self.assertContains(r, "not reachable")
        self.assertEqual(Analysis.objects.count(), 0)

    @override_settings(MAX_ACTIVE_ANALYSES_PER_USER=1)
    @mock.patch("reviews.views.start_analysis", return_value="j")
    def test_active_limit(self, _):
        self.make(status="running")
        r = self.client.post(reverse("analyze"), {
            "repo_url": "https://github.com/other/repo", "analysis_type": "public",
        })
        self.assertContains(r, "already have 1 analyses running")


class PollingTests(Base):
    def test_failed_revocation_job_recovers_saved_report(self):
        import copy

        analysis = self.make(
            status=Analysis.Status.FAILED,
            error="GitHub access could not be revoked automatically. Revoke Repo Intel in GitHub Settings → Applications.",
        )
        job = {
            "status": "failed",
            "error": analysis.error,
            "stage": "generate_report",
            "steps": ["generate_report"],
            "report": copy.deepcopy(REPORT),
        }

        with mock.patch.object(services, "get_job", return_value=job):
            response = self.client.get(reverse("report", args=[analysis.pk]))

        analysis.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hardcoded OpenAI API key")
        self.assertEqual(analysis.status, Analysis.Status.DONE)
        self.assertEqual(analysis.overall_score, 72)
        self.assertEqual(analysis.findings.count(), 3)
        self.assertTrue(any(
            "could not be revoked automatically" in warning
            for warning in analysis.report["warnings"]
        ))

    def test_progress_then_done_saves_findings(self):
        a = self.make()
        with mock.patch.object(services, "get_job", return_value={"status": "running", "stage": "quality", "steps": ["fetch_repository", "quality"]}):
            data = self.client.get(reverse("analysis_status", args=[a.pk])).json()
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["done_steps"], ["fetch_repository", "quality"])
        self.assertFalse(data["finished"])

        import copy
        job = {"status": "done", "stage": "generate_report", "steps": ["generate_report"], "report": copy.deepcopy(REPORT)}
        with mock.patch.object(services, "get_job", return_value=job):
            data = self.client.get(reverse("analysis_status", args=[a.pk])).json()
        self.assertTrue(data["finished"])
        a.refresh_from_db()
        self.assertEqual((a.status, a.overall_score), ("done", 72))
        self.assertEqual(a.findings.count(), 3)
        self.assertEqual(list(a.findings.values_list("severity", flat=True)), ["critical", "high", "medium"])  # ordered by severity rank
        self.assertNotIn("findings", a.report)
        # idempotent: finished analyses are never re-synced
        with mock.patch.object(services, "get_job", side_effect=AssertionError("must not be called")):
            self.client.get(reverse("analysis_status", args=[a.pk]))

    def test_failed_job(self):
        a = self.make()
        with mock.patch.object(services, "get_job", return_value={"status": "failed", "error": "Repository u/demo was not found.", "steps": ["fetch_repository"]}):
            self.client.get(reverse("analysis_status", args=[a.pk]))
        a.refresh_from_db()
        self.assertEqual(a.status, "failed")
        resp = self.client.get(reverse("report", args=[a.pk]))
        self.assertContains(resp, "was not found")

    def test_lost_job(self):
        a = self.make()
        with mock.patch.object(services, "get_job", side_effect=JobNotFound("x")):
            self.client.get(reverse("analysis_status", args=[a.pk]))
        a.refresh_from_db()
        self.assertEqual(a.status, "failed")
        self.assertIn("interrupted", a.error)

    def test_service_temporarily_down_keeps_status(self):
        a = self.make()
        with mock.patch.object(services, "get_job", side_effect=AIServiceError("down")):
            data = self.client.get(reverse("analysis_status", args=[a.pk])).json()
        self.assertEqual(data["warning"], "down")
        a.refresh_from_db()
        self.assertEqual(a.status, "queued")


class ReportAndDashboardTests(Base):
    def done(self, **kw):
        import copy
        a = self.make(status="done", **kw)
        services.save_report(a, copy.deepcopy(REPORT))
        return a

    def test_report_page(self):
        a = self.done()
        r = self.client.get(reverse("report", args=[a.pk]))
        self.assertEqual(r.status_code, 200)
        for text in (
            "Hardcoded OpenAI API key", "Team guide, p.2", "Rotate the exposed key",
            "FastAPI", "Python", "Installation steps", "A warning", "your guidelines",
            "1 guideline violation reported", "3 indexed guideline passages",
        ):
            self.assertContains(r, text)
        for text in (
            "AI review coverage",
            "code chunks were successfully reviewed by AI",
            "Source files selected for analysis",
            "provider request sent to the AI provider",
        ):
            self.assertNotContains(r, text)
        self.assertContains(r, 'data-score="72"')
        self.assertContains(r, 'data-progress="83"')
        self.assertContains(r, 'data-progress="90.0"')
        self.assertNotContains(r, "Rule-based checks scanned")
        self.assertNotContains(r, "No guideline violations were detected")
        self.assertNotContains(r, "AI code review did not complete successfully")
        self.assertNotContains(r, "rule-based")

    def test_save_report_discards_non_ai_findings(self):
        import copy

        report = copy.deepcopy(REPORT)
        report["findings"].append({
            "file": "app/main.py", "line": 12, "category": "quality",
            "severity": "low", "issue": "Rule-only issue", "evidence": "pass",
            "reason": "Created by a deterministic rule.",
            "recommendation": "Review it.", "confidence": 0.8, "source": "rule",
        })
        report["findings"].append({
            "file": "app/main.py", "line": 13, "category": "quality",
            "severity": "low", "issue": "Unknown-source issue", "evidence": "pass",
            "reason": "Source is not recognized as an AI reviewer.",
            "recommendation": "Review it.", "confidence": 0.8, "source": "other",
        })
        analysis = self.make()

        services.save_report(analysis, report)

        self.assertEqual(
            set(analysis.findings.values_list("source", flat=True)),
            {"llm", "rag"},
        )
        self.assertEqual(analysis.report["stats"]["ai_findings"], 3)
        self.assertEqual(analysis.report["stats"]["rule_findings"], 0)

    def test_save_report_keeps_limited_rule_findings_for_fallback_report(self):
        import copy

        report = copy.deepcopy(REPORT)
        report["stats"]["rule_fallback_used"] = True
        report["findings"] = [{
            "file": "app/main.py", "line": 12, "category": "quality",
            "severity": "low", "issue": "Rule fallback issue", "evidence": "pass",
            "reason": "Detected by a limited static check.",
            "recommendation": "Review it.", "confidence": 0.8, "source": "rule",
        }]
        analysis = self.make()

        services.save_report(analysis, report)

        self.assertEqual(
            list(analysis.findings.values_list("source", flat=True)),
            ["rule"],
        )
        self.assertEqual(analysis.report["stats"]["rule_findings"], 1)
        self.assertEqual(analysis.report["stats"]["ai_findings"], 0)

        response = self.client.get(reverse("report", args=[analysis.pk]))

        self.assertContains(response, "Rule fallback issue")
        self.assertNotContains(response, "Rule-based check")

    def test_save_report_keeps_capped_rule_findings_alongside_ai_findings(self):
        import copy

        report = copy.deepcopy(REPORT)
        report["stats"]["rule_checks_used"] = True
        report["findings"].append({
            "file": "app/main.py", "line": 12, "category": "security",
            "severity": "high", "issue": "Rule security check",
            "evidence": "subprocess.run(cmd, shell=True)",
            "reason": "Shell execution can be unsafe.",
            "recommendation": "Avoid shell execution.", "confidence": 0.8, "source": "rule",
        })
        analysis = self.make()

        services.save_report(analysis, report)

        self.assertEqual(
            set(analysis.findings.values_list("source", flat=True)),
            {"llm", "rag", "rule"},
        )
        self.assertEqual(analysis.report["stats"]["ai_findings"], 3)
        self.assertEqual(analysis.report["stats"]["rule_findings"], 1)

    def test_legacy_non_ai_findings_are_hidden_from_report(self):
        analysis = self.make(status="done")
        services.save_report(analysis, {
            **{key: value for key, value in REPORT.items() if key != "findings"},
            "findings": [],
        })
        Finding.objects.create(
            analysis=analysis, file="app/main.py", line=12, category="quality",
            severity="low", issue="Legacy rule-only issue", source="rule",
            recommendation="Review it.",
        )
        Finding.objects.create(
            analysis=analysis, file="app/main.py", line=13, category="quality",
            severity="low", issue="Legacy unknown-source issue", source="other",
            recommendation="Review it.",
        )

        response = self.client.get(reverse("report", args=[analysis.pk]))

        self.assertNotContains(response, "Legacy rule-only issue")
        self.assertNotContains(response, "Legacy unknown-source issue")
        self.assertNotContains(response, "rule-based")

    def test_report_does_not_claim_guideline_compliance_when_none_reported(self):
        import copy

        report = copy.deepcopy(REPORT)
        report["findings"] = [finding for finding in report["findings"] if finding["source"] != "rag"]
        report["guideline_sources"] = ["Team guide, p.2"]
        analysis = self.make(status="done")
        services.save_report(analysis, report)

        response = self.client.get(reverse("report", args=[analysis.pk]))

        self.assertContains(
            response,
            "No guideline violations were detected in the reviewed code chunks. This result covers only the samples listed below, not the entire repository.",
        )

    def test_report_hides_only_github_revocation_warning(self):
        import copy

        report = copy.deepcopy(REPORT)
        report["warnings"] = [
            "GitHub access could not be revoked automatically. Revoke Repo Intel in GitHub Settings → Applications.",
            "A relevant AI review is incomplete.",
        ]
        analysis = self.make(status="done")
        services.save_report(analysis, report)

        response = self.client.get(reverse("report", args=[analysis.pk]))

        self.assertNotContains(response, "GitHub access could not be revoked automatically")
        self.assertContains(response, "A relevant AI review is incomplete.")
        self.assertIn("GitHub access could not be revoked automatically", analysis.report["warnings"][0])

    def test_profile_report_lists_repository_results(self):
        import copy
        profile_report = copy.deepcopy(REPORT)
        profile_report["repositories"] = [{
            "name": "demo", "url": "https://github.com/u/demo", "score": 72,
            "summary": "Good foundations.", "findings": 3,
        }]
        analysis = self.make(status="done", is_profile=True, owner="u", repo_name="profile")
        services.save_report(analysis, profile_report)

        response = self.client.get(reverse("report", args=[analysis.pk]))

        self.assertContains(response, "GitHub profile: @u")
        self.assertContains(response, "Repositories analyzed (1)")
        self.assertContains(response, "Good foundations.")

    def test_running_report_page(self):
        a = self.make(status="running", steps=["fetch_repository"])
        with mock.patch.object(services, "get_job", return_value={"status": "running", "stage": "fetch_repository", "steps": ["fetch_repository"]}):
            r = self.client.get(reverse("report", args=[a.pk]))
        self.assertContains(r, "Analyzing")
        self.assertContains(r, reverse("analysis_status", args=[a.pk]))

    def test_other_users_report_is_404(self):
        a = self.done()
        other = User.objects.create_user("other", "o@example.com", "pass12345!")
        self.client.force_login(other)
        for name in ("report", "analysis_status"):
            self.assertEqual(self.client.get(reverse(name, args=[a.pk])).status_code, 404)
        self.assertEqual(self.client.post(reverse("delete_analysis", args=[a.pk])).status_code, 404)

    def test_dashboard(self):
        self.client.get(reverse("dashboard"))
        analysis = self.done()
        Finding.objects.create(
            analysis=analysis, file="app/main.py", line=24, category="performance",
            severity="medium", issue="Repeated database query", source="llm",
            recommendation="Reuse the fetched result.",
        )
        Finding.objects.create(
            analysis=analysis, file="README.md", line=None, category="documentation",
            severity="low", issue="Missing installation steps", source="llm",
            recommendation="Document the setup process.",
        )
        Finding.objects.create(
            analysis=analysis, file="app/main.py", line=20, category="security",
            severity="critical", issue="Legacy rule-only issue", source="rule",
            recommendation="Review it.",
        )
        r = self.client.get(reverse("dashboard"))
        self.assertEqual(r.status_code, 200)
        ctx = r.context
        self.assertEqual((ctx["analyzed"], ctx["avg_score"], ctx["security_issues"], ctx["code_issues"]), (1, 72, 2, 2))
        self.assertEqual((ctx["performance_issues"], ctx["doc_issues"]), (1, 1))
        self.assertEqual(ctx["chart_data"]["categories"]["values"], [1, 2, 1, 0, 1])
        self.assertNotContains(r, "Language distribution")
        self.assertNotContains(r, "Technologies detected across reports")
        self.assertEqual(ctx["chart_data"]["severity"]["values"], [1, 1, 2, 1])
        self.assertNotIn("languages", ctx["chart_data"])
        self.assertNotIn("technologies", ctx["chart_data"])

    def test_dashboard_counts_rule_findings_for_explicit_rule_check_reports(self):
        import copy

        fallback_report = copy.deepcopy(REPORT)
        fallback_report["stats"]["rule_checks_used"] = True
        fallback_report["findings"] = [{
            "file": "app/main.py", "line": 12, "category": "security",
            "severity": "high", "issue": "Fallback finding",
            "recommendation": "Review it.", "source": "rule",
        }]
        analysis = self.make(status="done")
        services.save_report(analysis, fallback_report)

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["security_issues"], 1)

    def test_dashboard_shows_average_performance_and_documentation_scores(self):
        import copy

        self.done()
        second_report = copy.deepcopy(REPORT)
        second_report["scores"]["performance"] = 60
        second_report["scores"]["documentation"] = None
        second_analysis = self.make(status="done", job_id="job2")
        services.save_report(second_analysis, second_report)

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.context["avg_performance_score"], 60)
        self.assertEqual(response.context["avg_documentation_score"], 82)
        self.assertContains(response, "Performance score")
        self.assertContains(response, "Documentation score")
        self.assertContains(response, 'data-metric="avg_performance_score">60')
        self.assertContains(response, 'data-metric="avg_documentation_score">82')

    def test_dashboard_data_returns_live_saved_metrics_and_charts(self):
        analysis = self.done()
        Finding.objects.create(
            analysis=analysis, file="app/main.py", line=24, category="performance",
            severity="medium", issue="Repeated database query", source="llm",
            recommendation="Reuse the fetched result.",
        )

        response = self.client.get(reverse("dashboard_data"))

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["analyzed"], 1)
        self.assertEqual(data["performance_issues"], 1)
        self.assertEqual(data["chart_data"]["categories"]["values"], [1, 2, 1, 0, 0])
        self.assertEqual(data["chart_data"]["scores"]["labels"], ["u/demo"])

    def test_dashboard_does_not_render_language_or_technology_charts(self):
        self.done()
        response = self.client.get(reverse("dashboard"))
        self.assertNotContains(response, "Language distribution")
        self.assertNotContains(response, "Technologies detected across reports")

    def test_dashboard_data_syncs_running_analysis_before_refresh(self):
        import copy

        analysis = self.make(status="running")
        report = copy.deepcopy(REPORT)
        with mock.patch("reviews.services.get_job", return_value={"status": "done", "report": report}):
            response = self.client.get(reverse("dashboard_data"))

        analysis.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(analysis.status, Analysis.Status.DONE)
        self.assertEqual(response.json()["analyzed"], 1)

    def test_history_delete_rerun(self):
        a = self.done()
        self.assertContains(self.client.get(reverse("history")), "u/demo")
        with mock.patch("reviews.views.start_analysis", return_value="newjob"):
            r = self.client.post(reverse("rerun_analysis", args=[a.pk]))
        self.assertEqual(Analysis.objects.count(), 2)
        self.assertEqual(r.status_code, 302)
        self.client.post(reverse("delete_analysis", args=[a.pk]))
        self.assertFalse(Analysis.objects.filter(pk=a.pk).exists())
        self.assertEqual(Finding.objects.count(), 0)

    def test_legacy_single_repository_rerun_uses_public_access(self):
        a = self.make(status="failed")
        with mock.patch("reviews.views.start_analysis", return_value="newjob") as start:
            response = self.client.post(reverse("rerun_analysis", args=[a.pk]))

        start.assert_called_once_with(a.repo_url, a.branch, "", self.user.id)
        self.assertEqual(response.status_code, 302)
        rerun = Analysis.objects.get(job_id="newjob")
        self.assertEqual(rerun.repo_url, a.repo_url)

    def test_legacy_profile_rerun_prompts_for_specific_repository(self):
        analysis = self.make(status="failed", is_profile=True, includes_private=True, owner="github-member")
        response = self.client.post(reverse("rerun_analysis", args=[analysis.pk]))
        self.assertRedirects(response, reverse("analyze"))
        self.assertNotIn("pending_private_repo_url", self.client.session)

    def test_private_repository_rerun_requires_a_new_authorization(self):
        analysis = self.make(
            status="failed", includes_private=True, is_profile=False,
            repo_url="https://github.com/org/private-repo",
        )
        response = self.client.post(reverse("rerun_analysis", args=[analysis.pk]))

        self.assertRedirects(response, reverse("github_connect"), fetch_redirect_response=False)
        self.assertEqual(
            self.client.session["pending_private_repo_url"],
            "https://github.com/org/private-repo",
        )


class GuidelineTests(Base):
    def test_guidelines_page_wraps_table_for_narrow_screens(self):
        response = self.client.get(reverse("guidelines"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="table-scroll"')
        self.assertContains(response, "No guidelines uploaded yet.")

    def test_upload_and_delete(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        f = SimpleUploadedFile("py.md", b"Never use bare except.", content_type="text/markdown")
        with mock.patch("reviews.views.upload_guideline", return_value=4) as up:
            r = self.client.post(reverse("guidelines"), {"title": "Py guide", "file": f})
        self.assertRedirects(r, reverse("guidelines"))
        g = Guideline.objects.get()
        self.assertEqual((g.title, g.chunks), ("Py guide", 4))
        self.assertEqual(up.call_args.args[:4], (str(g.remote_id), self.user.id, "Py guide", "py.md"))
        with mock.patch("reviews.views.delete_guideline") as dele:
            self.client.post(reverse("delete_guideline", args=[g.pk]))
        dele.assert_called_once_with(str(g.remote_id))
        self.assertEqual(Guideline.objects.count(), 0)

    def test_rejects_bad_type(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        r = self.client.post(reverse("guidelines"), {"file": SimpleUploadedFile("x.exe", b"zz")})
        self.assertContains(r, "Upload a PDF")
        self.assertEqual(Guideline.objects.count(), 0)


class ProfileTests(Base):
    def test_update(self):
        r = self.client.post(reverse("profile"), {"first_name": "Divya", "last_name": "G", "email": "new@example.com"})
        self.assertRedirects(r, reverse("profile"))
        self.user.refresh_from_db()
        self.assertEqual((self.user.first_name, self.user.email), ("Divya", "new@example.com"))
