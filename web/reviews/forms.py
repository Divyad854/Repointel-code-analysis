import re

from django import forms

from .services import parse_repo_url

MAX_GUIDELINE_MB = 10


class AnalyzeForm(forms.Form):
    analysis_type = forms.ChoiceField(
        label="Analysis type",
        choices=(("public", "Public repository"), ("private", "Private repository")),
        widget=forms.RadioSelect,
        initial="public",
    )
    repo_url = forms.CharField(
        label="GitHub repository URL",
        max_length=300,
        widget=forms.URLInput(attrs={
            "placeholder": "https://github.com/username/repository",
            "autofocus": True,
        }),
    )

    def clean_repo_url(self):
        value = self.cleaned_data["repo_url"].strip().rstrip("/")
        if not re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
            value,
            flags=re.IGNORECASE,
        ):
            raise forms.ValidationError(
                "Enter a repository URL like https://github.com/username/repository."
            )
        try:
            self.owner, self.name, branch = parse_repo_url(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc
        if branch:
            raise forms.ValidationError("Enter the repository URL without a branch or file path.")
        return f"https://github.com/{self.owner}/{self.name}"


class GuidelineForm(forms.Form):
    title = forms.CharField(label="Title (optional)", max_length=200, required=False,
                            widget=forms.TextInput(attrs={"placeholder": "e.g. Company Python style guide"}))
    file = forms.FileField(label="Guideline file (PDF, .txt or .md)")

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith((".pdf", ".txt", ".md", ".markdown", ".rst")):
            raise forms.ValidationError("Upload a PDF, .txt or .md file.")
        if f.size > MAX_GUIDELINE_MB * 1024 * 1024:
            raise forms.ValidationError(f"File is too large (max {MAX_GUIDELINE_MB} MB).")
        return f
