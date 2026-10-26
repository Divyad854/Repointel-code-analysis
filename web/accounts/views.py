import hmac
import logging
import secrets
import smtplib
import time
from urllib.parse import urlencode

import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model, login, logout
from django.contrib.auth.hashers import check_password, make_password
from django.contrib.auth.views import PasswordChangeView
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.db.models import Avg, Count
from django.shortcuts import redirect, render
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import ensure_csrf_cookie
from reviews.models import Analysis, Guideline

from .forms import EmailForm, OTPForm, ProfileForm, RegisterForm, ResetPasswordForm
from reviews.services import AIServiceError, parse_repo_url, start_analysis

logger = logging.getLogger(__name__)
User = get_user_model()
OTP_TTL_SECONDS = 10 * 60
OTP_MAX_ATTEMPTS = 5
REGISTRATION_SESSION_KEY = "pending_registration"
RESET_SESSION_KEY = "pending_password_reset"
VERIFIED_RESET_SESSION_KEY = "verified_password_reset"

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_URL = "https://api.github.com"


def _github_callback_uri(request):
    return settings.GITHUB_OAUTH_REDIRECT_URI or request.build_absolute_uri(
        reverse("github_callback")
    )


class AccountPasswordChangeView(PasswordChangeView):
    template_name = "accounts/password_change.html"
    success_url = reverse_lazy("profile")

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, "Your password has been changed.")
        return response


@never_cache
@ensure_csrf_cookie
def register(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        email = form.cleaned_data["email"]
        code = f"{secrets.randbelow(1_000_000):06d}"
        request.session[REGISTRATION_SESSION_KEY] = {
            "username": form.cleaned_data["username"],
            "email": email,
            "password": make_password(form.cleaned_data["password1"]),
            "code_hash": make_password(code),
            "expires_at": time.time() + OTP_TTL_SECONDS,
            "attempts": 0,
        }
        try:
            _send_otp(email, code, "Verify your Repo Intel account")
        except ValueError as exc:
            logger.warning("Email configuration error: %s", exc)
            request.session.pop(REGISTRATION_SESSION_KEY, None)
            form.add_error(
                None,
                "Email verification is not configured. Set EMAIL_HOST and "
                "DEFAULT_FROM_EMAIL in the root .env file, then restart Django.",
            )
        except (OSError, smtplib.SMTPException) as exc:
            logger.warning("Could not send account verification email: %s", exc)
            request.session.pop(REGISTRATION_SESSION_KEY, None)
            form.add_error(None, "We could not send the verification email. Check the SMTP host, port, and credentials.")
        else:
            messages.info(request, "We sent a 6-digit verification code to your email.")
            return redirect("verify_registration")
    return render(request, "accounts/register.html", {"form": form})


def _send_otp(email, code, subject):
    if settings.EMAIL_BACKEND == "django.core.mail.backends.smtp.EmailBackend":
        if not settings.EMAIL_HOST or not settings.DEFAULT_FROM_EMAIL:
            raise ValueError(
                "SMTP is selected, but EMAIL_HOST or DEFAULT_FROM_EMAIL is not configured."
            )
        if settings.EMAIL_USE_TLS and settings.EMAIL_USE_SSL:
            raise ValueError("Use either EMAIL_USE_TLS or EMAIL_USE_SSL, not both.")

    sent = send_mail(
        subject,
        f"Your Repo Intel verification code is {code}. It expires in 10 minutes.",
        settings.DEFAULT_FROM_EMAIL,
        [email],
        fail_silently=False,
    )
    if sent != 1:
        raise OSError("Email backend did not send the verification message.")


def _check_otp(data, code):
    if not data or data.get("expires_at", 0) <= time.time():
        return "expired"
    if data.get("attempts", 0) >= OTP_MAX_ATTEMPTS:
        return "locked"
    if not check_password(code, data.get("code_hash", "")):
        data["attempts"] = data.get("attempts", 0) + 1
        return "invalid"
    return "valid"


def verify_registration(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    pending = request.session.get(REGISTRATION_SESSION_KEY)
    if not pending:
        messages.info(request, "Start by entering your registration details.")
        return redirect("register")

    form = OTPForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        outcome = _check_otp(pending, form.cleaned_data["code"])
        if outcome == "valid":
            if User.objects.filter(username=pending["username"]).exists():
                request.session.pop(REGISTRATION_SESSION_KEY, None)
                messages.error(request, "That username is now in use. Please register again.")
                return redirect("register")
            if User.objects.filter(email__iexact=pending["email"]).exists():
                request.session.pop(REGISTRATION_SESSION_KEY, None)
                messages.error(request, "An account with this email already exists.")
                return redirect("register")
            user = User(
                username=pending["username"],
                email=pending["email"],
                password=pending["password"],
            )
            user.save()
            request.session.pop(REGISTRATION_SESSION_KEY, None)
            login(request, user)
            messages.success(request, "Your email is verified and your account is ready.")
            return redirect("dashboard")
        request.session[REGISTRATION_SESSION_KEY] = pending
        if outcome in {"expired", "locked"}:
            request.session.pop(REGISTRATION_SESSION_KEY, None)
            messages.error(request, "That verification code expired or exceeded the attempt limit. Please register again.")
            return redirect("register")
        form.add_error("code", "That verification code is incorrect.")

    return render(request, "accounts/otp_verify.html", {
        "form": form,
        "title": "Verify your email",
        "description": f"Enter the 6-digit code sent to {pending['email']}.",
        "back_url": reverse("register"),
        "back_label": "Start registration again",
    })


def forgot_password(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    form = EmailForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        email = form.cleaned_data["email"]
        request.session.pop(RESET_SESSION_KEY, None)
        request.session.pop(VERIFIED_RESET_SESSION_KEY, None)
        code = f"{secrets.randbelow(1_000_000):06d}"
        reset_data = {
            "email": email,
            "code_hash": make_password(code),
            "expires_at": time.time() + OTP_TTL_SECONDS,
            "attempts": 0,
        }
        user_exists = User.objects.filter(email__iexact=email).exists()
        request.session[RESET_SESSION_KEY] = reset_data
        if user_exists:
            try:
                _send_otp(email, code, "Reset your Repo Intel password")
            except ValueError as exc:
                logger.warning("Email configuration error: %s", exc)
                request.session.pop(RESET_SESSION_KEY, None)
                messages.error(
                    request,
                    "Email verification is not configured. Set EMAIL_HOST and "
                    "DEFAULT_FROM_EMAIL in the root .env file, then restart Django.",
                )
                return render(request, "accounts/password_reset_email.html", {"form": form})
            except (OSError, smtplib.SMTPException) as exc:
                logger.warning("Could not send password reset email: %s", exc)
                request.session.pop(RESET_SESSION_KEY, None)
                messages.error(request, "We could not send the verification email. Check the SMTP host, port, and credentials.")
                return render(request, "accounts/password_reset_email.html", {"form": form})
        messages.info(request, "If an account exists for that email, a verification code has been sent.")
        return redirect("password_reset_verify")
    return render(request, "accounts/password_reset_email.html", {"form": form})


def verify_password_reset(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    pending = request.session.get(RESET_SESSION_KEY)
    if not pending:
        messages.info(request, "Enter your email address to request a password reset code.")
        return redirect("password_reset")

    form = OTPForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        outcome = _check_otp(pending, form.cleaned_data["code"])
        if outcome == "valid" and User.objects.filter(email__iexact=pending["email"]).exists():
            request.session.pop(RESET_SESSION_KEY, None)
            request.session[VERIFIED_RESET_SESSION_KEY] = {
                "email": pending["email"],
                "expires_at": time.time() + OTP_TTL_SECONDS,
            }
            return redirect("password_reset_set")
        if outcome in {"expired", "locked"}:
            request.session.pop(RESET_SESSION_KEY, None)
            messages.error(request, "That code expired or exceeded the attempt limit. Request a new code.")
            return redirect("password_reset")
        request.session[RESET_SESSION_KEY] = pending
        form.add_error("code", "That code is incorrect or expired.")

    return render(request, "accounts/otp_verify.html", {
        "form": form,
        "title": "Verify your email",
        "description": "Enter the 6-digit password reset code sent to your email.",
        "back_url": reverse("password_reset"),
        "back_label": "Request another code",
    })


def set_reset_password(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    verified = request.session.get(VERIFIED_RESET_SESSION_KEY)
    if not verified or verified.get("expires_at", 0) <= time.time():
        request.session.pop(VERIFIED_RESET_SESSION_KEY, None)
        messages.info(request, "Verify a password reset code before choosing a new password.")
        return redirect("password_reset")
    user = User.objects.filter(email__iexact=verified["email"]).first()
    if not user:
        request.session.pop(VERIFIED_RESET_SESSION_KEY, None)
        messages.error(request, "The account could not be found. Request a new password reset code.")
        return redirect("password_reset")

    form = ResetPasswordForm(user, request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        request.session.pop(VERIFIED_RESET_SESSION_KEY, None)
        messages.success(request, "Your password has been reset. You can now log in.")
        return redirect("login")
    return render(request, "accounts/password_reset_new.html", {"form": form})


@login_required
def connect_github(request):
    repo_url = request.session.get("pending_private_repo_url")
    if not repo_url:
        messages.error(request, "Submit a private repository analysis before connecting GitHub.")
        return redirect("analyze")
    if not settings.GITHUB_OAUTH_CLIENT_ID or not settings.GITHUB_OAUTH_CLIENT_SECRET:
        request.session.pop("pending_private_repo_url", None)
        messages.error(request, "GitHub authentication is not configured yet.")
        return redirect("analyze")

    state = secrets.token_urlsafe(32)
    request.session["github_oauth_state"] = state
    params = urlencode({
        "client_id": settings.GITHUB_OAUTH_CLIENT_ID,
        "redirect_uri": _github_callback_uri(request),
        "scope": "repo",
        "state": state,
    })
    return redirect(f"{GITHUB_AUTHORIZE_URL}?{params}")


def _revoke_github_grant(access_token):
    for attempt in range(3):
        try:
            response = requests.delete(
                f"{GITHUB_API_URL}/applications/{settings.GITHUB_OAUTH_CLIENT_ID}/grant",
                auth=(settings.GITHUB_OAUTH_CLIENT_ID, settings.GITHUB_OAUTH_CLIENT_SECRET),
                json={"access_token": access_token},
                headers={"Accept": "application/vnd.github+json"},
                timeout=15,
            )
            if response.status_code == 404:
                return
            response.raise_for_status()
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))


@login_required
def github_callback(request):
    expected_state = request.session.pop("github_oauth_state", None)
    repo_url = request.session.pop("pending_private_repo_url", None)
    received_state = request.GET.get("state", "")
    if not expected_state or not hmac.compare_digest(expected_state, received_state):
        messages.error(request, "GitHub authorization could not be verified. Please try again.")
        return redirect("analyze")

    if request.GET.get("error"):
        messages.error(request, "GitHub authorization was cancelled.")
        return redirect("analyze")

    code = request.GET.get("code", "")
    if not code or not repo_url:
        messages.error(request, "GitHub did not return an authorization code. Please try again.")
        return redirect("analyze")
    try:
        owner, repo_name, branch = parse_repo_url(repo_url)
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect("analyze")

    try:
        token_response = requests.post(
            GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            json={
                "client_id": settings.GITHUB_OAUTH_CLIENT_ID,
                "client_secret": settings.GITHUB_OAUTH_CLIENT_SECRET,
                "code": code,
                "redirect_uri": _github_callback_uri(request),
            },
            timeout=15,
        )
        token_response.raise_for_status()
        token_data = token_response.json()
    except requests.RequestException:
        messages.error(request, "Could not contact GitHub to complete authorization. Please try again.")
        return redirect("analyze")
    except ValueError:
        messages.error(request, "GitHub returned an invalid authorization response.")
        return redirect("analyze")

    access_token = token_data.get("access_token")
    if not access_token:
        messages.error(request, "GitHub did not grant repository access. Please authorize and try again.")
        return redirect("analyze")

    try:
        user_response = requests.get(
            f"{GITHUB_API_URL}/user",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "repo-intel/1.0",
            },
            timeout=15,
        )
        user_response.raise_for_status()
        github_login = user_response.json().get("login")
    except requests.RequestException:
        try:
            _revoke_github_grant(access_token)
        except requests.RequestException:
            messages.error(request, "Could not verify the GitHub account or revoke access. Revoke Repo Intel in GitHub Settings → Applications.")
            return redirect("analyze")
        messages.error(request, "Could not verify your GitHub account. Please try again.")
        return redirect("analyze")
    except ValueError:
        try:
            _revoke_github_grant(access_token)
        except requests.RequestException:
            messages.error(request, "GitHub returned an invalid account response and access could not be revoked. Revoke Repo Intel in GitHub Settings → Applications.")
            return redirect("analyze")
        messages.error(request, "GitHub returned an invalid account response.")
        return redirect("analyze")

    if not github_login:
        try:
            _revoke_github_grant(access_token)
        except requests.RequestException:
            messages.error(request, "GitHub account verification failed and access could not be revoked. Revoke Repo Intel in GitHub Settings → Applications.")
            return redirect("analyze")
        messages.error(request, "GitHub did not return an account name. Please try again.")
        return redirect("analyze")

    try:
        job_id = start_analysis(repo_url, branch, access_token, request.user.id)
    except (AIServiceError, requests.RequestException, ValueError) as exc:
        try:
            _revoke_github_grant(access_token)
        except requests.RequestException:
            messages.error(request, "Could not start the analysis or revoke GitHub access. Revoke Repo Intel in GitHub Settings → Applications.")
            return redirect("analyze")
        messages.error(request, str(exc))
        return redirect("analyze")

    analysis = Analysis.objects.create(
        user=request.user,
        repo_url=repo_url,
        owner=owner,
        repo_name=repo_name,
        branch=branch,
        includes_private=True,
        job_id=job_id,
        status=Analysis.Status.QUEUED,
    )
    return redirect("report", pk=analysis.pk)


@login_required
def profile(request):
    form = ProfileForm(request.POST or None, instance=request.user)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Profile updated.")
        return redirect("profile")
    done = Analysis.objects.filter(user=request.user, status=Analysis.Status.DONE)
    stats = done.aggregate(n=Count("id"), avg=Avg("overall_score"))
    return render(request, "accounts/profile.html", {
        "form": form,
        "analyses": stats["n"],
        "avg_score": round(stats["avg"] or 0),
        "guidelines": Guideline.objects.filter(user=request.user).count(),
    })
