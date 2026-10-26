from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("analyze/", views.analyze, name="analyze"),
    path("history/", views.history, name="history"),
    path("reports/<int:pk>/", views.report, name="report"),
    path("reports/<int:pk>/status/", views.analysis_status, name="analysis_status"),
    path("reports/<int:pk>/delete/", views.delete_analysis, name="delete_analysis"),
    path("reports/<int:pk>/rerun/", views.rerun_analysis, name="rerun_analysis"),
    path("guidelines/", views.guidelines, name="guidelines"),
    path("guidelines/<int:pk>/delete/", views.delete_guideline_view, name="delete_guideline"),
]
