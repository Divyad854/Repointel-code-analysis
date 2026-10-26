import uuid

from django.conf import settings
from django.db import models

SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class Analysis(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analyses")
    repo_url = models.CharField(max_length=300)
    owner = models.CharField(max_length=100)
    repo_name = models.CharField(max_length=100)
    branch = models.CharField(max_length=200, blank=True)
    is_profile = models.BooleanField(default=False)
    includes_private = models.BooleanField(default=False)
    target_repo_name = models.CharField(max_length=100, blank=True)

    status = models.CharField(max_length=10, choices=Status.choices, default=Status.QUEUED, db_index=True)
    stage = models.CharField(max_length=40, blank=True)
    steps = models.JSONField(default=list, blank=True)
    job_id = models.CharField(max_length=64, blank=True)
    error = models.TextField(blank=True)

    overall_score = models.IntegerField(null=True, blank=True)
    scores = models.JSONField(default=dict, blank=True)
    report = models.JSONField(default=dict, blank=True)  # everything except the findings (stored in Finding)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "analyses"

    def __str__(self) -> str:
        target = f"@{self.owner} profile" if self.is_profile else f"{self.owner}/{self.repo_name}"
        return f"{target} ({self.status})"

    @property
    def full_name(self) -> str:
        return f"@{self.owner} profile" if self.is_profile else f"{self.owner}/{self.repo_name}"

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.DONE, self.Status.FAILED)


class Finding(models.Model):
    class Category(models.TextChoices):
        QUALITY = "quality", "Code quality"
        SECURITY = "security", "Security"
        PERFORMANCE = "performance", "Performance"
        BEST_PRACTICES = "best_practices", "Best practices"
        DOCUMENTATION = "documentation", "Documentation"

    class Severity(models.TextChoices):
        CRITICAL = "critical", "Critical"
        HIGH = "high", "High"
        MEDIUM = "medium", "Medium"
        LOW = "low", "Low"

    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="findings")
    file = models.CharField(max_length=500)
    line = models.IntegerField(null=True, blank=True)
    category = models.CharField(max_length=20, choices=Category.choices, db_index=True)
    severity = models.CharField(max_length=10, choices=Severity.choices)
    rank = models.PositiveSmallIntegerField(default=3, editable=False)  # 0 = critical ... 3 = low, for ordering
    issue = models.CharField(max_length=300)
    evidence = models.TextField(blank=True)
    reason = models.TextField(blank=True)
    recommendation = models.TextField(blank=True)
    confidence = models.FloatField(default=0.7)
    source = models.CharField(max_length=10, default="llm")  # rule | llm | rag
    guideline = models.CharField(max_length=300, blank=True)

    class Meta:
        ordering = ["rank", "-confidence", "file", "line"]

    def save(self, *args, **kwargs):
        self.rank = SEVERITY_RANK.get(self.severity, 3)
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.file}:{self.line} {self.issue}"


class Guideline(models.Model):
    class Status(models.TextChoices):
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="guidelines")
    remote_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    title = models.CharField(max_length=200)
    original_name = models.CharField(max_length=255)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.READY)
    chunks = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title
