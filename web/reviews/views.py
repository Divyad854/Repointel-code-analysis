"""
Django views for the Repo Intel application.

This module connects browser requests to:
- Django models (PostgreSQL database records)
- Django templates (HTML pages)
- The AI service (repository analysis and guideline indexing)

Comments explain the important logic without changing the application's behavior.
"""

# Django project settings, such as MAX_ACTIVE_ANALYSES_PER_USER.
from django.conf import settings

# Used to show success, warning, and error messages on the next rendered page.
from django.contrib import messages

# Blocks access to a view unless the user is authenticated.
from django.contrib.auth.decorators import login_required

# Splits long repository-history lists into multiple pages.
from django.core.paginator import Paginator

# Database aggregation and complex query helpers:
# Avg = average, Count = count rows, Q = combine conditions with OR/AND.
from django.db.models import Avg, Count, Q

# Returns data in JSON format, commonly consumed by frontend JavaScript.
from django.http import JsonResponse

# Common shortcuts for loading templates, redirecting, and fetching records.
from django.shortcuts import get_object_or_404, redirect, render

# Restrict an endpoint to a particular HTTP method.
from django.views.decorators.http import require_GET, require_POST

# Forms used to validate repository-analysis and guideline-upload input.
from .forms import AnalyzeForm, GuidelineForm

# Database models used by these views.
from .models import Analysis, Finding, GitHubConnection, Guideline

# Service functions communicate with the AI service.
# These functions keep API/business logic outside the Django view layer.
from .services import (
    AIServiceError,
    PrivateRepositoryAuthRequired,
    delete_guideline,
    start_analysis,
    sync_analysis,
    upload_guideline,
)


# Ordered stages displayed on the report page.
# The first value is the internal stage key; the second is its human-readable label.
# This list describes progress for the UI; it does not itself execute the AI stages.
PIPELINE = [
    ("fetch_repository", "Fetch repository"),
    ("detect_technology", "Detect technologies"),
    ("collect_files", "Select & split source files"),
    ("quality", "Code quality review"),
    ("security", "Security review"),
    ("performance", "Performance review"),
    ("documentation", "Documentation review"),
    ("best_practices", "Best practices (+ your guidelines)"),
    ("architecture", "Architecture analysis"),
    ("final_reviewer", "Final reviewer & scoring"),
    ("generate_report", "Generate report"),
]


def _get_github_access_token(user):
    """Return this Repo Intel user's saved GitHub token, if connected."""
    connection = GitHubConnection.objects.filter(user=user).first()
    if not connection:
        return ""

    try:
        return connection.get_access_token()
    except ValueError:
        # Keep the connection record. A fresh OAuth callback can replace the token.
        return ""


def _dashboard_context(user):
    """Build the statistics and chart data needed by one user's dashboard."""

    # Only completed analyses belonging to this user contribute to final statistics.
    done = Analysis.objects.filter(user=user, status=Analysis.Status.DONE)

    # Include LLM and RAG findings. Include rule-based findings only when the
    # report says rule checks or rule fallback checks were actually used.
    findings = Finding.objects.filter(analysis__in=done).filter(
        Q(source__in=["llm", "rag"])
        | Q(
            source="rule",
            analysis__report__stats__rule_checks_used=True,
        )
        | Q(
            source="rule",
            analysis__report__stats__rule_fallback_used=True,
        )
    )

    # Group findings by category and count them, e.g. security: 10, quality: 5.
    by_category = {
        r["category"]: r["n"]
        for r in findings.values("category").annotate(n=Count("id"))
    }

    # Group findings by severity, e.g. critical, high, medium, and low.
    by_severity = {
        r["severity"]: r["n"]
        for r in findings.values("severity").annotate(n=Count("id"))
    }

    # Collect numeric performance and documentation scores from completed reports.
    category_scores: dict[str, list[float]] = {
        "performance": [],
        "documentation": [],
    }

    # values_list(..., flat=True) retrieves only the scores field from each record.
    for scores in done.values_list("scores", flat=True):
        for category, values in category_scores.items():
            score = (scores or {}).get(category)

            # Ignore missing values, booleans, and non-numeric values.
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                values.append(score)

    # Calculate an average for each category. Use None when there are no scores.
    average_category_scores = {
        category: round(sum(values) / len(values)) if values else None
        for category, values in category_scores.items()
    }

    # Take the eight most recent completed analyses, then reverse them so chart
    # points appear from older to newer.
    recent = list(done.order_by("-created_at")[:8])[::-1]

    # Prepare one context dictionary for the dashboard template.
    context = {
        "analyzed": done.count(),
        "avg_score": round(done.aggregate(a=Avg("overall_score"))["a"] or 0),
        "security_issues": by_category.get("security", 0),
        "performance_issues": by_category.get("performance", 0),
        "code_issues": sum(
            by_category.get(c, 0)
            for c in ("quality", "performance", "best_practices")
        ),
        "doc_issues": by_category.get("documentation", 0),
        "avg_performance_score": average_category_scores["performance"],
        "avg_documentation_score": average_category_scores["documentation"],

        # Count this user's analyses that are queued or currently running.
        "active": Analysis.objects.filter(
            user=user,
            status__in=[Analysis.Status.QUEUED, Analysis.Status.RUNNING],
        ).count(),

        "has_data": done.exists(),

        # Latest five analyses for the dashboard's recent-activity section.
        "latest": Analysis.objects.filter(user=user)[:5],

        # Chart labels and values are passed to the frontend chart code.
        "chart_data": {
            "scores": {
                "labels": [
                    f"@{a.owner}" if a.is_profile else f"{a.owner}/{a.repo_name}"
                    for a in recent
                ],
                "values": [a.overall_score for a in recent],
            },
            "categories": {
                "labels": ["Quality", "Security", "Best practices"],
                "values": [
                    by_category.get(k, 0)
                    for k in ("quality", "security", "best_practices")
                ],
            },
            "severity": {
                "labels": ["Critical", "High", "Medium", "Low"],
                "values": [
                    by_severity.get(k, 0)
                    for k in ("critical", "high", "medium", "low")
                ],
            },
        },
    }

    return context


def dashboard(request):
    """Show the public home page or the authenticated user's dashboard."""

    # Anonymous visitors see the public landing page.
    if not request.user.is_authenticated:
        return render(request, "reviews/home.html")

    # Logged-in users see dashboard statistics belonging to their account.
    return render(
        request,
        "reviews/dashboard.html",
        _dashboard_context(request.user),
    )


@login_required
@require_GET
def dashboard_data(request):
    """
    Return fresh dashboard statistics as JSON.

    The frontend can request this endpoint without reloading the full dashboard.
    """

    # Find this user's analyses that have not finished yet.
    active = Analysis.objects.filter(
        user=request.user,
        status__in=[Analysis.Status.QUEUED, Analysis.Status.RUNNING],
    )

    try:
        # Ask the AI service for the latest state of each active analysis and
        # synchronize that state into the Django database.
        for analysis in active:
            sync_analysis(analysis)
    except AIServiceError as exc:
        # HTTP 503 indicates that the required AI service is temporarily unavailable.
        return JsonResponse({"error": str(exc)}, status=503)

    # Rebuild statistics after synchronization so the response uses updated states.
    context = _dashboard_context(request.user)

    response = JsonResponse({
        "analyzed": context["analyzed"],
        "avg_score": context["avg_score"],
        "security_issues": context["security_issues"],
        "code_issues": context["code_issues"],
        "performance_issues": context["performance_issues"],
        "doc_issues": context["doc_issues"],
        "avg_performance_score": context["avg_performance_score"],
        "avg_documentation_score": context["avg_documentation_score"],
        "active": context["active"],
        "has_data": context["has_data"],
        "chart_data": context["chart_data"],
    })

    # Prevent browsers and intermediate caches from reusing stale dashboard data.
    response["Cache-Control"] = "no-store"
    return response


@login_required
def analyze(request):
    """Validate a repository URL and submit a new analysis job."""

    # Bind submitted POST data to the form; on GET, show an empty form.
    form = AnalyzeForm(request.POST or None)

    if request.method == "POST" and form.is_valid():

        # Enforce the configured maximum number of queued/running analyses
        # for this individual user. This is not the global AI semaphore limit.
        active = Analysis.objects.filter(
            user=request.user,
            status__in=[Analysis.Status.QUEUED, Analysis.Status.RUNNING],
        ).count()

        if active >= settings.MAX_ACTIVE_ANALYSES_PER_USER:
            form.add_error(
                None,
                f"You already have {active} analyses running. Wait for one to finish.",
            )
        else:
            repo_url = form.cleaned_data["repo_url"]

            is_private = form.cleaned_data["analysis_type"] == "private"
            github_token = (
                _get_github_access_token(request.user) if is_private else ""
            )

            # The token belongs to the Repo Intel account, not this browser session.
            # It can therefore be reused on other devices after the user signs in.
            if is_private and not github_token:
                request.session["pending_private_repo_url"] = repo_url
                return redirect("github_connect")

            # Clear any old pending private-repository URL before starting analysis.
            request.session.pop("pending_private_repo_url", None)

            try:
                job_id = start_analysis(
                    repo_url,
                    "",
                    github_token,
                    request.user.id,
                )

            except PrivateRepositoryAuthRequired:
                # Keep the database connection. OAuth callback can replace its token.
                request.session["pending_private_repo_url"] = repo_url
                return redirect("github_connect")

            except AIServiceError as exc:
                # Show an AI-service error on the submitted form.
                form.add_error(None, str(exc))

            else:
                # The AI service accepted the job and returned its identifier.
                # Save a corresponding record in Django's database.
                analysis = Analysis.objects.create(
                    user=request.user,
                    repo_url=repo_url,
                    owner=form.owner,
                    repo_name=form.name,
                    job_id=job_id,
                    status=Analysis.Status.QUEUED,
                    includes_private=is_private,
                )

                # Open the report/progress page for the newly created analysis.
                return redirect("report", pk=analysis.pk)

    # Let the template know whether this user has any ready-to-use guidelines.
    has_guidelines = Guideline.objects.filter(
        user=request.user,
        status=Guideline.Status.READY,
    ).exists()

    return render(request, "reviews/analyze.html", {
        "form": form,
        "has_guidelines": has_guidelines,
    })


@login_required
def history(request):
    """Show one row per repository, with its latest analysis and report count."""

    # Retrieve only this user's analyses, newest first.
    analyses = Analysis.objects.filter(
        user=request.user
    ).order_by("-created_at", "-pk")

    # Group records by repository URL. Each Analysis remains a separate database row.
    repositories = {}

    for analysis in analyses:
        repo_url = analysis.repo_url

        # The first record encountered is the latest because the queryset is newest-first.
        if repo_url not in repositories:
            repositories[repo_url] = {
                "latest": analysis,
                "report_count": 0,
            }

        # Count every analysis/report for this repository.
        repositories[repo_url]["report_count"] += 1

    # Show 15 repository groups per page. get_page() handles invalid page values safely.
    page = Paginator(list(repositories.values()), 15).get_page(
        request.GET.get("page")
    )

    return render(request, "reviews/history.html", {"page": page})


@login_required
def repository_history(request, pk):
    """Show every analysis report for the selected repository."""

    # Fetch the selected analysis only if it belongs to the logged-in user.
    # Otherwise, return a 404 page (also prevents cross-user access by changing pk).
    selected = get_object_or_404(
        Analysis,
        pk=pk,
        user=request.user,
    )

    # Use the selected record's repository URL to find all its previous reports.
    reports = Analysis.objects.filter(
        user=request.user,
        repo_url=selected.repo_url,
    ).order_by("-created_at", "-pk")

    return render(
        request,
        "reviews/repository_history.html",
        {
            "repository": selected,
            "reports": reports,
        },
    )


@login_required
def report(request, pk):
    """Display analysis progress or the completed report."""

    # Retrieve the requested analysis while ensuring it belongs to this user.
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)

    # If the job is unfinished, synchronize its current state with the AI service.
    if not analysis.is_finished:
        try:
            sync_analysis(analysis)
        except AIServiceError:
            # The report page can still render; show a warning if the AI service
            # cannot currently provide an update.
            messages.warning(
                request,
                "Waiting for the AI service... the page will keep retrying.",
            )

    # These values are always sent to the report template, including while running.
    ctx = {
        "analysis": analysis,
        "pipeline": PIPELINE,
    }

    # Add detailed report information only after analysis has completed successfully.
    if analysis.status == Analysis.Status.DONE:
        rep = analysis.report

        # Determine whether rule-based findings should be included in the UI.
        rule_checks_used = bool(
            rep.get("stats", {}).get("rule_checks_used")
            or rep.get("stats", {}).get("rule_fallback_used")
        )

        # Keep LLM/RAG findings. Include rule findings only when rule checks were used.
        findings = [
            finding for finding in analysis.findings.all()
            if finding.source in {"llm", "rag"}
            or (rule_checks_used and finding.source == "rule")
        ]

        # Separate RAG findings so the UI can identify guideline-related findings.
        guideline_findings = [
            finding for finding in findings if finding.source == "rag"
        ]

        # Group AI-generated and RAG findings together.
        ai_findings = [
            finding for finding in findings if finding.source in {"llm", "rag"}
        ]

        # Display warnings returned by the analysis service.
        displayed_warnings = rep.get("warnings", [])

        # Add all completed-report data required by the report template.
        ctx.update({
            "rep": rep,
            "displayed_warnings": displayed_warnings,
            "findings": findings,
            "guideline_findings": guideline_findings,
            "ai_findings": ai_findings,
            "rule_checks_used": rule_checks_used,

            # Scores displayed as individual report cards.
            "score_cards": [
                ("Code quality", analysis.scores.get("quality")),
                ("Security", analysis.scores.get("security")),
                ("Performance", analysis.scores.get("performance")),
                ("Best practices", analysis.scores.get("best_practices")),
                ("Documentation", analysis.scores.get("documentation")),
            ],

            # Human-readable labels mapped to keys in the report's documentation section.
            "doc_items": [
                ("README present", "readme"),
                ("Project description", "description"),
                ("Installation steps", "installation"),
                ("Usage instructions", "usage"),
                ("Environment variables", "environment_variables"),
                ("API documentation", "api_docs"),
                ("Testing notes", "tests"),
                ("License", "license"),
            ],

            # Convert detected technologies into key/value pairs for template iteration.
            "tech_groups": list(
                (rep.get("technology") or {}).get("technologies", {}).items()
            ),

            # Category choices defined by the Finding model.
            "categories": Finding.Category.choices,
        })

    return render(request, "reviews/report.html", ctx)


@login_required
@require_GET
def analysis_status(request, pk):
    """Return one analysis's latest progress as JSON for frontend polling."""

    # Restrict the lookup to the current user.
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)
    warning = ""

    try:
        # Refresh the local Analysis record using the AI service's latest job status.
        sync_analysis(analysis)
    except AIServiceError as exc:
        # Return the warning in JSON while still returning the known local status.
        warning = str(exc)

    # Convert completed step keys to a set for efficient membership checks.
    done_steps = set(analysis.steps)

    return JsonResponse({
        "status": analysis.status,
        "stage": analysis.stage,
        "error": analysis.error,
        "warning": warning,

        # Preserve PIPELINE order when returning completed stages to the frontend.
        "done_steps": [
            key for key, _ in PIPELINE if key in done_steps
        ],
        "finished": analysis.is_finished,
    })


@login_required
@require_POST
def delete_analysis(request, pk):
    """Delete one analysis belonging to the logged-in user."""

    # The user filter prevents deleting another user's analysis by guessing its ID.
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)

    # Delete the database record (related-object behavior depends on model relationships).
    analysis.delete()

    messages.success(request, "Analysis deleted.")
    return redirect("history")


@login_required
@require_POST
def rerun_analysis(request, pk):
    """Submit a fresh analysis for a repository from the user's history."""

    # Find the old report and verify ownership.
    old = get_object_or_404(Analysis, pk=pk, user=request.user)

    # Profile-wide reruns are intentionally not supported by this view.
    if old.is_profile:
        messages.info(
            request,
            "Profile-wide reruns are no longer available. Enter a specific repository URL to analyze one repository.",
        )
        return redirect("analyze")

    # Retrieve the token from the user's database connection, not the session.
    github_token = (
        _get_github_access_token(request.user) if old.includes_private else ""
    )
    if old.includes_private and not github_token:
        request.session["pending_private_repo_url"] = old.repo_url
        return redirect("github_connect")

    try:
        # Submit a new job using the old repository URL, branch, and saved token.
        job_id = start_analysis(
            old.repo_url,
            old.branch,
            github_token,
            request.user.id,
        )
    except PrivateRepositoryAuthRequired:
        # Keep the saved connection; request fresh OAuth authorization.
        request.session["pending_private_repo_url"] = old.repo_url
        return redirect("github_connect")
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect("report", pk=old.pk)
    except AIServiceError as exc:
        messages.error(request, str(exc))
        return redirect("report", pk=old.pk)

    # Create a NEW Analysis record. The previous report remains in history.
    new = Analysis.objects.create(
        user=request.user,
        repo_url=old.repo_url,
        owner=old.owner,
        repo_name=old.repo_name,
        branch=old.branch,
        job_id=job_id,
        includes_private=old.includes_private,
    )

    return redirect("report", pk=new.pk)


@login_required
def guidelines(request):
    """Upload a user's coding guideline and index it in the AI service."""

    # request.FILES makes uploaded file data available to the Django form.
    form = GuidelineForm(request.POST or None, request.FILES or None)

    if request.method == "POST" and form.is_valid():
        f = form.cleaned_data["file"]

        # Use the supplied title, or derive a default title from the filename.
        title = (
            form.cleaned_data["title"].strip()
            or f.name.rsplit(".", 1)[0]
        )

        # Build the guideline model object. It is saved after successful indexing.
        g = Guideline(
            user=request.user,
            title=title,
            original_name=f.name,
        )

        try:
            # Send the guideline bytes to the AI service for processing/indexing.
            # remote_id identifies the guideline in the remote search index;
            # the returned chunk count is stored on the Django Guideline record.
            g.chunks = upload_guideline(
                str(g.remote_id),
                request.user.id,
                title,
                f.name,
                f.read(),
            )
        except AIServiceError as exc:
            # Do not save the local guideline record if remote indexing failed.
            form.add_error(None, str(exc))
        else:
            # Save metadata only after the AI service reports successful indexing.
            g.save()
            messages.success(
                request,
                f"“{title}” indexed ({g.chunks} passages). It will be used in your next analysis.",
            )
            return redirect("guidelines")

    # Display the upload form and this user's existing guideline records.
    return render(
        request,
        "reviews/guidelines.html",
        {
            "form": form,
            "items": Guideline.objects.filter(user=request.user),
        },
    )


@login_required
@require_POST
def delete_guideline_view(request, pk):
    """Remove a guideline from the AI search index and the Django database."""

    # Only allow the owner of the guideline to delete it.
    g = get_object_or_404(Guideline, pk=pk, user=request.user)

    try:
        # Remove indexed chunks from the AI service first.
        delete_guideline(str(g.remote_id))
    except AIServiceError as exc:
        # Keep the Django record if remote deletion failed, so the two systems
        # do not silently become inconsistent.
        messages.error(
            request,
            f"Could not remove it from the search index: {exc}",
        )
        return redirect("guidelines")

    # Delete the local metadata only after remote deletion succeeds.
    g.delete()
    messages.success(request, "Guideline removed.")
    return redirect("guidelines")
