import json
import re
from unittest import mock

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from reviews.models import Analysis

User = get_user_model()


class UsernameOrEmailLoginTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="member",
            email="member@example.com",
            password="Strong-test-password-123!",
        )

    def test_login_with_username(self):
        response = self.client.post(reverse("login"), {
            "username": "member",
            "password": "Strong-test-password-123!",
        })

        self.assertRedirects(response, reverse("dashboard"))
        self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.pk)

    def test_login_with_email_case_insensitively(self):
        response = self.client.post(reverse("login"), {
            "username": "MEMBER@EXAMPLE.COM",
            "password": "Strong-test-password-123!",
        })

        self.assertRedirects(response, reverse("dashboard"))
        self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.pk)

    def test_invalid_email_login_is_rejected(self):
        response = self.client.post(reverse("login"), {
            "username": "unknown@example.com",
            "password": "Strong-test-password-123!",
        })

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class EmailOTPTests(TestCase):
    def setUp(self):
        mail.outbox.clear()

    @staticmethod
    def last_code():
        return re.search(r"\b(\d{6})\b", mail.outbox[-1].body).group(1)

    @override_settings(
        EMAIL_BACKEND="django.core.mail.backends.smtp.EmailBackend",
        EMAIL_HOST="",
        DEFAULT_FROM_EMAIL="",
    )
    def test_registration_reports_missing_smtp_configuration(self):
        response = self.client.post(reverse("register"), {
            "username": "no-smtp",
            "email": "no-smtp@example.com",
            "password1": "Strong-test-password-123!",
            "password2": "Strong-test-password-123!",
        })

        self.assertContains(response, "Set EMAIL_HOST and DEFAULT_FROM_EMAIL")
        self.assertFalse(User.objects.filter(username="no-smtp").exists())
        self.assertNotIn("pending_registration", self.client.session)

    def test_registration_requires_correct_email_code_before_creating_user(self):
        response = self.client.post(reverse("register"), {
            "username": "verified",
            "email": "verified@example.com",
            "password1": "Strong-test-password-123!",
            "password2": "Strong-test-password-123!",
        })
        self.assertRedirects(response, reverse("verify_registration"))
        self.assertFalse(User.objects.filter(username="verified").exists())

        response = self.client.post(reverse("verify_registration"), {"code": "000000"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="verified").exists())

        response = self.client.post(reverse("verify_registration"), {"code": self.last_code()})
        self.assertRedirects(response, reverse("dashboard"))
        self.assertTrue(User.objects.filter(username="verified", email="verified@example.com").exists())

    def test_fresh_registration_form_csrf_token_can_be_submitted(self):
        client = Client(enforce_csrf_checks=True)
        response = client.get(reverse("register"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response["Cache-Control"])
        self.assertIn("csrftoken", client.cookies)

        response = client.post(reverse("register"), {
            "csrfmiddlewaretoken": re.search(
                r'name="csrfmiddlewaretoken" value="([^"]+)"',
                response.content.decode(),
            ).group(1),
            "username": "csrf-user",
            "email": "csrf-user@example.com",
            "password1": "Strong-test-password-123!",
            "password2": "Strong-test-password-123!",
        })

        self.assertRedirects(response, reverse("verify_registration"))

    def test_registration_code_expiry_prevents_account_creation(self):
        self.client.post(reverse("register"), {
            "username": "expired",
            "email": "expired@example.com",
            "password1": "Strong-test-password-123!",
            "password2": "Strong-test-password-123!",
        })
        session = self.client.session
        pending = session["pending_registration"]
        pending["expires_at"] = 0
        session["pending_registration"] = pending
        session.save()

        response = self.client.post(reverse("verify_registration"), {"code": self.last_code()})

        self.assertRedirects(response, reverse("register"))
        self.assertFalse(User.objects.filter(username="expired").exists())

    def test_password_reset_requires_code_then_updates_password(self):
        user = User.objects.create_user("reset-user", "reset@example.com", "Original-password-123!")
        response = self.client.post(reverse("password_reset"), {"email": "reset@example.com"})
        self.assertRedirects(response, reverse("password_reset_verify"))
        self.assertEqual(len(mail.outbox), 1)

        response = self.client.post(reverse("password_reset_verify"), {"code": "000000"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.client.session.get("verified_password_reset"))

        response = self.client.post(reverse("password_reset_verify"), {"code": self.last_code()})
        self.assertRedirects(response, reverse("password_reset_set"))
        self.assertContains(self.client.get(reverse("password_reset_set")), "Set a new password")

        response = self.client.post(reverse("password_reset_set"), {
            "new_password1": "A-new-unique-password-88!",
            "new_password2": "A-new-unique-password-88!",
        })
        self.assertRedirects(response, reverse("login"))
        user.refresh_from_db()
        self.assertTrue(user.check_password("A-new-unique-password-88!"))
        self.assertFalse(self.client.session.get("verified_password_reset"))

    def test_password_reset_response_does_not_disclose_unknown_email(self):
        response = self.client.post(reverse("password_reset"), {"email": "missing@example.com"})

        self.assertRedirects(response, reverse("password_reset_verify"))
        self.assertEqual(len(mail.outbox), 0)
        response = self.client.post(reverse("password_reset_verify"), {"code": "123456"})
        self.assertContains(response, "incorrect or expired")


@override_settings(
    GITHUB_OAUTH_CLIENT_ID="github-client",
    GITHUB_OAUTH_CLIENT_SECRET="github-secret",
    GITHUB_OAUTH_REDIRECT_URI="http://testserver/settings/github/callback/",
)
class GitHubOAuthTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("member", password="strong-test-password")
        self.client.force_login(self.user)
        session = self.client.session
        session["pending_private_repo_url"] = "https://github.com/github-member/private-repo"
        session.save()

    @staticmethod
    def response(status_code, data):
        from requests import Response

        response = Response()
        response.status_code = status_code
        response._content = json.dumps(data).encode()
        return response

    def test_connect_requires_profile_analysis_and_redirects_with_repo_scope(self):
        session = self.client.session
        session.pop("pending_private_repo_url")
        session.save()
        response = self.client.get(reverse("github_connect"))
        self.assertRedirects(response, reverse("analyze"))

        session = self.client.session
        session["pending_private_repo_url"] = "https://github.com/github-member/private-repo"
        session.save()
        response = self.client.get(reverse("github_connect"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("scope=repo", response["Location"])
        self.assertIn("state=", response["Location"])
        self.assertIn("github_oauth_state", self.client.session)

    @mock.patch("accounts.views.start_analysis", return_value="job-123")
    @mock.patch("accounts.views.requests.get")
    @mock.patch("accounts.views.requests.post")
    def test_callback_starts_profile_job_without_persisting_authorization(self, post, get, start):
        session = self.client.session
        session["github_oauth_state"] = "valid-state"
        session["pending_private_repo_url"] = "https://github.com/github-member/private-repo"
        session.save()
        post.return_value = self.response(200, {"access_token": "oauth-access-secret", "scope": "repo"})
        get.return_value = self.response(200, {"login": "github-member"})

        response = self.client.get(reverse("github_callback"), {
            "code": "temporary-code", "state": "valid-state",
        })

        analysis = Analysis.objects.get()
        self.assertRedirects(response, reverse("report", args=[analysis.pk]))
        self.assertTrue(analysis.includes_private)
        self.assertEqual(analysis.owner, "github-member")
        self.assertEqual(analysis.repo_name, "private-repo")
        self.assertFalse(analysis.is_profile)
        self.assertTrue(get.call_args.kwargs["headers"]["Authorization"].startswith("Bearer "))
        start.assert_called_once_with(
            "https://github.com/github-member/private-repo", "", "oauth-access-secret", self.user.id,
        )
        self.assertEqual(Analysis.objects.count(), 1)

    @mock.patch("accounts.views.requests.post")
    def test_callback_rejects_invalid_state_without_exchanging_code(self, post):
        session = self.client.session
        session["github_oauth_state"] = "expected-state"
        session.save()

        response = self.client.get(reverse("github_callback"), {
            "code": "temporary-code", "state": "wrong-state",
        })

        self.assertRedirects(response, reverse("analyze"))
        post.assert_not_called()
        self.assertFalse(Analysis.objects.exists())

    @mock.patch("accounts.views.start_analysis", return_value="job-123")
    @mock.patch("accounts.views.requests.get")
    @mock.patch("accounts.views.requests.post")
    def test_private_repository_can_be_authorized_by_a_collaborator(self, post, get, start):
        session = self.client.session
        session["github_oauth_state"] = "valid-state"
        session["pending_private_repo_url"] = "https://github.com/org/shared-private-repo"
        session.save()
        post.return_value = self.response(200, {"access_token": "oauth-access-secret"})
        get.return_value = self.response(200, {"login": "collaborator"})

        response = self.client.get(reverse("github_callback"), {
            "code": "temporary-code", "state": "valid-state",
        })

        analysis = Analysis.objects.get()
        self.assertRedirects(response, reverse("report", args=[analysis.pk]))
        self.assertEqual((analysis.owner, analysis.repo_name), ("org", "shared-private-repo"))
        start.assert_called_once_with(
            "https://github.com/org/shared-private-repo", "", "oauth-access-secret", self.user.id,
        )
