from django.contrib import admin

from .models import Analysis, Finding, Guideline


class FindingInline(admin.TabularInline):
    model = Finding
    extra = 0
    fields = ("severity", "category", "file", "line", "issue", "source")
    readonly_fields = fields
    can_delete = False
    max_num = 0


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "owner", "repo_name", "user", "status", "overall_score", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("owner", "repo_name", "user__username")
    inlines = [FindingInline]


@admin.register(Finding)
class FindingAdmin(admin.ModelAdmin):
    list_display = ("issue", "severity", "category", "file", "line", "source")
    list_filter = ("severity", "category", "source")
    search_fields = ("issue", "file")


@admin.register(Guideline)
class GuidelineAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "chunks", "status", "created_at")
