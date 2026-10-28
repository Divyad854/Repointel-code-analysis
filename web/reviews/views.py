from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from .forms import AnalyzeForm, GuidelineForm
from .models import Analysis, Finding, Guideline
from .services import (
    AIServiceError, PrivateRepositoryAuthRequired, delete_guideline, start_analysis,
    sync_analysis, upload_guideline,
)

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


def _dashboard_context(user):
    done = Analysis.objects.filter(user=user, status=Analysis.Status.DONE)
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
    by_category = {r["category"]: r["n"] for r in findings.values("category").annotate(n=Count("id"))}
    by_severity = {r["severity"]: r["n"] for r in findings.values("severity").annotate(n=Count("id"))}

    category_scores: dict[str, list[float]] = {"performance": [], "documentation": []}
    for scores in done.values_list("scores", flat=True):
        for category, values in category_scores.items():
            score = (scores or {}).get(category)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                values.append(score)

    average_category_scores = {
        category: round(sum(values) / len(values)) if values else None
        for category, values in category_scores.items()
    }

    recent = list(done.order_by("-created_at")[:8])[::-1]
    context = {
        "analyzed": done.count(),
        "avg_score": round(done.aggregate(a=Avg("overall_score"))["a"] or 0),
        "security_issues": by_category.get("security", 0),
        "performance_issues": by_category.get("performance", 0),
        "code_issues": sum(by_category.get(c, 0) for c in ("quality", "performance", "best_practices")),
        "doc_issues": by_category.get("documentation", 0),
        "avg_performance_score": average_category_scores["performance"],
        "avg_documentation_score": average_category_scores["documentation"],
        "active": Analysis.objects.filter(user=user, status__in=[Analysis.Status.QUEUED, Analysis.Status.RUNNING]).count(),
        "has_data": done.exists(),
        "latest": Analysis.objects.filter(user=user)[:5],
        "chart_data": {
            "scores": {
                "labels": [f"@{a.owner}" if a.is_profile else f"{a.owner}/{a.repo_name}" for a in recent],
                "values": [a.overall_score for a in recent],
            },
            "categories": {
                "labels": ["Quality", "Security", "Performance", "Best practices", "Documentation"],
                "values": [by_category.get(k, 0) for k in ("quality", "security", "performance", "best_practices", "documentation")],
            },
            "severity": {"labels": ["Critical", "High", "Medium", "Low"],
                         "values": [by_severity.get(k, 0) for k in ("critical", "high", "medium", "low")]},
        },
    }
    return context


def dashboard(request):
    if not request.user.is_authenticated:
        return render(request, "reviews/home.html")
    return render(request, "reviews/dashboard.html", _dashboard_context(request.user))


@login_required
@require_GET
def dashboard_data(request):
    active = Analysis.objects.filter(
        user=request.user,
        status__in=[Analysis.Status.QUEUED, Analysis.Status.RUNNING],
    )
    try:
        for analysis in active:
            sync_analysis(analysis)
    except AIServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=503)

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
    response["Cache-Control"] = "no-store"
    return response


@login_required
def analyze(request):
    form = AnalyzeForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        active = Analysis.objects.filter(user=request.user, status__in=[Analysis.Status.QUEUED, Analysis.Status.RUNNING]).count()
        if active >= settings.MAX_ACTIVE_ANALYSES_PER_USER:
            form.add_error(None, f"You already have {active} analyses running. Wait for one to finish.")
        else:
            repo_url = form.cleaned_data["repo_url"]
            if form.cleaned_data["analysis_type"] == "private":
                request.session["pending_private_repo_url"] = repo_url
                return redirect("github_connect")
            request.session.pop("pending_private_repo_url", None)
            try:
                job_id = start_analysis(repo_url, "", "", request.user.id)
            except PrivateRepositoryAuthRequired:
                request.session["pending_private_repo_url"] = repo_url
                return redirect("github_connect")
            except AIServiceError as exc:
                form.add_error(None, str(exc))
            else:
                analysis = Analysis.objects.create(
                    user=request.user,
                    repo_url=repo_url,
                    owner=form.owner,
                    repo_name=form.name,
                    job_id=job_id,
                    status=Analysis.Status.QUEUED,
                )
                return redirect("report", pk=analysis.pk)
    has_guidelines = Guideline.objects.filter(user=request.user, status=Guideline.Status.READY).exists()
    return render(request, "reviews/analyze.html", {
        "form": form,
        "has_guidelines": has_guidelines,
    })


@login_required
def history(request):
    page = Paginator(Analysis.objects.filter(user=request.user), 15).get_page(request.GET.get("page"))
    return render(request, "reviews/history.html", {"page": page})


@login_required
def report(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)
    if (
        not analysis.is_finished
        or (
            analysis.status == Analysis.Status.FAILED
            and "GitHub access could not be revoked automatically" in analysis.error
        )
    ):
        try:
            sync_analysis(analysis)
        except AIServiceError:
            messages.warning(request, "Waiting for the AI service... the page will keep retrying.")
    ctx = {
        "analysis": analysis,
        "pipeline": PIPELINE,
    }
    if analysis.status == Analysis.Status.DONE:
        rep = analysis.report
        rule_checks_used = bool(
            rep.get("stats", {}).get("rule_checks_used")
            or rep.get("stats", {}).get("rule_fallback_used")
        )
        findings = [
            finding for finding in analysis.findings.all()
            if finding.source in {"llm", "rag"}
            or (rule_checks_used and finding.source == "rule")
        ]
        guideline_findings = [finding for finding in findings if finding.source == "rag"]
        ai_findings = [finding for finding in findings if finding.source in {"llm", "rag"}]
        displayed_warnings = [
            warning for warning in rep.get("warnings", [])
            if "GitHub access could not be revoked automatically" not in warning
        ]
        ctx.update({
            "rep": rep,
            "displayed_warnings": displayed_warnings,
            "findings": findings,
            "guideline_findings": guideline_findings,
            "ai_findings": ai_findings,
            "rule_checks_used": rule_checks_used,
            "score_cards": [
                ("Code quality", analysis.scores.get("quality")),
                ("Security", analysis.scores.get("security")),
                ("Performance", analysis.scores.get("performance")),
                ("Best practices", analysis.scores.get("best_practices")),
                ("Documentation", analysis.scores.get("documentation")),
            ],
            "doc_items": [
                ("README present", "readme"), ("Project description", "description"), ("Installation steps", "installation"),
                ("Usage instructions", "usage"), ("Environment variables", "environment_variables"),
                ("API documentation", "api_docs"), ("Testing notes", "tests"), ("License", "license"),
            ],
            "tech_groups": list((rep.get("technology") or {}).get("technologies", {}).items()),
            "categories": Finding.Category.choices,
        })
    return render(request, "reviews/report.html", ctx)


@login_required
@require_GET
def analysis_status(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)
    warning = ""
    try:
        sync_analysis(analysis)
    except AIServiceError as exc:
        warning = str(exc)
    done_steps = set(analysis.steps)
    return JsonResponse({
        "status": analysis.status, "stage": analysis.stage, "error": analysis.error, "warning": warning,
        "done_steps": [key for key, _ in PIPELINE if key in done_steps],
        "finished": analysis.is_finished,
    })


@login_required
@require_POST
def delete_analysis(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)
    analysis.delete()
    messages.success(request, "Analysis deleted.")
    return redirect("history")


@login_required
@require_POST
def rerun_analysis(request, pk):
    old = get_object_or_404(Analysis, pk=pk, user=request.user)
    if old.includes_private and not old.is_profile:
        request.session["pending_private_repo_url"] = old.repo_url
        return redirect("github_connect")
    if old.is_profile:
        messages.info(request, "Profile-wide reruns are no longer available. Enter a specific repository URL to analyze one repository.")
        return redirect("analyze")
    try:
        job_id = start_analysis(old.repo_url, old.branch, "", request.user.id)
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect("report", pk=old.pk)
    except AIServiceError as exc:
        messages.error(request, str(exc))
        return redirect("report", pk=old.pk)
    new = Analysis.objects.create(
        user=request.user, repo_url=old.repo_url, owner=old.owner, repo_name=old.repo_name,
        branch=old.branch, job_id=job_id,
    )
    return redirect("report", pk=new.pk)


@login_required
def guidelines(request):
    form = GuidelineForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        f = form.cleaned_data["file"]
        title = form.cleaned_data["title"].strip() or f.name.rsplit(".", 1)[0]
        g = Guideline(user=request.user, title=title, original_name=f.name)
        try:
            g.chunks = upload_guideline(str(g.remote_id), request.user.id, title, f.name, f.read())
        except AIServiceError as exc:
            form.add_error(None, str(exc))
        else:
            g.save()
            messages.success(request, f"“{title}” indexed ({g.chunks} passages). It will be used in your next analysis.")
            return redirect("guidelines")
    return render(request, "reviews/guidelines.html", {"form": form, "items": Guideline.objects.filter(user=request.user)})


@login_required
@require_POST
def delete_guideline_view(request, pk):
    g = get_object_or_404(Guideline, pk=pk, user=request.user)
    try:
        delete_guideline(str(g.remote_id))
    except AIServiceError as exc:
        messages.error(request, f"Could not remove it from the search index: {exc}")
        return redirect("guidelines")
    g.delete()
    messages.success(request, "Guideline removed.")
    return redirect("guidelines")
