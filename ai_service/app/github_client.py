"""Fetch a repository from GitHub (one zipball download, a few API calls)."""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field

import httpx

from .config import get_settings

API = "https://api.github.com"

_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?"
    r"(?:/tree/(?P<branch>[^/\s]+))?(?:/.*)?$"
)
_SHORT_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?$")


class GitHubError(Exception):
    """A problem the user can understand (bad URL, private repo, rate limit...)."""


class GitHubAuthorizationRequired(GitHubError):
    """GitHub does not expose this repository without an authorized account."""


@dataclass
class RepoRef:
    owner: str
    name: str
    branch: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


def parse_repo(text: str, branch: str | None = None) -> RepoRef:
    """Accepts https://github.com/owner/repo, github.com/owner/repo/tree/branch or owner/repo."""
    text = (text or "").strip().rstrip("/")
    m = _URL_RE.match(text) or _SHORT_RE.match(text)
    if not m:
        raise GitHubError("That doesn't look like a GitHub repository. Use https://github.com/owner/repo or owner/repo.")
    owner, name = m.group("owner"), m.group("name")
    if name.lower().endswith(".git"):
        name = name[:-4]
    found_branch = m.groupdict().get("branch")
    return RepoRef(owner=owner, name=name, branch=(branch or found_branch or None))


@dataclass
class RepoSnapshot:
    ref: RepoRef
    default_branch: str = "main"
    description: str = ""
    stars: int = 0
    is_private: bool = False
    languages_bytes: dict[str, int] = field(default_factory=dict)
    all_paths: list[str] = field(default_factory=list)  # every non-ignored path in the repo
    files: dict[str, str] = field(default_factory=dict)  # path -> text, readable text files only
    skipped_large: int = 0


# --------------------------------------------------------------- filtering
IGNORED_DIRS = {
    "node_modules", ".git", "dist", "build", "__pycache__", ".venv", "venv", "env", ".env.d",
    "vendor", ".next", ".nuxt", ".idea", ".vscode", "coverage", "target", "obj", "site-packages",
    "staticfiles", ".tox", ".mypy_cache", ".pytest_cache", ".gradle", "bower_components",
    ".terraform", "Pods", ".dart_tool",
}
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
    "composer.lock", "go.sum", "Gemfile.lock", "cargo.lock", "Cargo.lock", ".DS_Store",
}
TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".go", ".rb", ".php", ".cs", ".rs",
    ".kt", ".swift", ".c", ".h", ".cpp", ".hpp", ".cc", ".vue", ".svelte", ".sh", ".bash", ".sql",
    ".html", ".css", ".scss", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".xml", ".gradle", ".tf", ".md", ".rst", ".txt", ".env", ".example", ".sample", ".template", ".dockerfile",
}
SPECIAL_NAMES = {
    "dockerfile", "makefile", "procfile", "gemfile", ".gitignore", ".dockerignore", ".env.example",
    ".env.sample", "requirements.txt", "license", "licence", "readme",
}


def is_ignored(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    if any(p in IGNORED_DIRS for p in parts[:-1]):
        return True
    if name in IGNORED_FILES or name.endswith((".min.js", ".min.css", ".map", ".lock")):
        return True
    return False


def is_text_candidate(path: str) -> bool:
    name = path.split("/")[-1].lower()
    if name in SPECIAL_NAMES or name.startswith("dockerfile") or name.startswith("readme"):
        return True
    if name.startswith(".env") and not name.endswith((".example", ".sample", ".template")):
        return False  # real .env files are never read; they are reported by path only
    dot = name.rfind(".")
    return dot != -1 and name[dot:] in TEXT_EXTS


# ------------------------------------------------------------ zip handling
def snapshot_from_zip(
    data: bytes, ref: RepoRef, *, default_branch: str = "main", description: str = "",
    stars: int = 0, languages: dict[str, int] | None = None, is_private: bool = False,
) -> RepoSnapshot:
    s = get_settings()
    snap = RepoSnapshot(ref=ref, default_branch=default_branch, description=description,
                        stars=stars, is_private=is_private, languages_bytes=languages or {})
    total_read = 0
    max_total = 25 * 1024 * 1024
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # GitHub zipballs wrap everything in "<owner>-<repo>-<sha>/"
            _, _, path = info.filename.partition("/")
            if not path or is_ignored(path) or ".." in path.split("/"):
                continue
            snap.all_paths.append(path)
            if not is_text_candidate(path):
                continue
            if info.file_size > s.max_file_bytes:
                snap.skipped_large += 1
                continue
            if total_read + info.file_size > max_total or len(snap.files) >= 3000:
                continue
            raw = zf.read(info)
            if b"\x00" in raw[:2048]:  # binary
                continue
            total_read += len(raw)
            snap.files[path] = raw.decode("utf-8", errors="ignore")
    return snap


# -------------------------------------------------------------- HTTP fetch
def _explain(resp: httpx.Response, ref: RepoRef) -> GitHubError:
    if resp.status_code == 404:
        hint = (
            " GitHub returns 404 both when a repository does not exist and when a private repository is not "
            "accessible. Choose Private repository and authorize the GitHub account that can access it."
        )
        return GitHubError(f"Repository {ref.full_name} was not found.{hint}")
    if resp.status_code in (401,):
        return GitHubError("GitHub authorization was rejected (401). Reconnect your GitHub account.")
    if resp.status_code in (403, 429):
        remaining = resp.headers.get("x-ratelimit-remaining")
        if remaining == "0" or resp.status_code == 429:
            return GitHubError("GitHub API rate limit reached. Try again later or connect your GitHub account.")
        return GitHubError(
            "GitHub denied access to this repository (403). Check that your connected GitHub account can access it."
        )
    return GitHubError(f"GitHub returned HTTP {resp.status_code} for {ref.full_name}.")


async def ensure_public_repository(ref: RepoRef) -> None:
    """Reject hidden/private repositories before an unauthenticated analysis is queued."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repo-intel/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(headers=headers, timeout=15) as client:
        try:
            response = await client.get(f"{API}/repos/{ref.owner}/{ref.name}")
        except httpx.HTTPError as exc:
            raise GitHubError(f"Could not verify repository visibility with GitHub: {exc.__class__.__name__}.") from exc

    if response.status_code == 404:
        raise GitHubAuthorizationRequired(
            f"Repository {ref.full_name} is private or unavailable without GitHub authorization."
        )
    if response.status_code != 200:
        raise _explain(response, ref)
    try:
        is_private = response.json().get("private", False)
    except ValueError as exc:
        raise GitHubError("GitHub returned an invalid repository response.") from exc
    if is_private:
        raise GitHubAuthorizationRequired(
            f"Repository {ref.full_name} is private and requires GitHub authorization."
        )


async def fetch_snapshot(ref: RepoRef, access_token: str | None = None) -> RepoSnapshot:
    s = get_settings()
    access_token = (access_token or "").strip()
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "repo-intel/1.0",
               "X-GitHub-Api-Version": "2022-11-28"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    async with httpx.AsyncClient(headers=headers, timeout=60, follow_redirects=True) as client:
        try:
            meta = await client.get(f"{API}/repos/{ref.owner}/{ref.name}")
            if meta.status_code != 200:
                raise _explain(meta, ref)
            info = meta.json()
            branch = ref.branch or info.get("default_branch") or "main"

            languages: dict[str, int] = {}
            lang = await client.get(f"{API}/repos/{ref.owner}/{ref.name}/languages")
            if lang.status_code == 200:
                languages = lang.json()

            buf = bytearray()
            limit = s.max_zip_mb * 1024 * 1024
            async with client.stream("GET", f"{API}/repos/{ref.owner}/{ref.name}/zipball/{branch}") as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise _explain(resp, ref)
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > limit:
                        raise GitHubError(f"Repository is larger than {s.max_zip_mb} MB; it is too big to analyze here.")
        except httpx.HTTPError as exc:
            raise GitHubError(f"Could not reach GitHub: {exc.__class__.__name__}") from exc

    try:
        return snapshot_from_zip(
            bytes(buf), ref, default_branch=branch, description=info.get("description") or "",
            stars=info.get("stargazers_count") or 0, languages=languages,
            is_private=bool(info.get("private")),
        )
    except zipfile.BadZipFile as exc:
        raise GitHubError("GitHub returned an unreadable archive.") from exc


async def list_profile_repositories(
    username: str,
    *,
    include_private: bool = False,
    access_token: str | None = None,
    repository_name: str | None = None,
) -> list[RepoRef]:
    """List repositories owned by a profile, using OAuth only for private mode."""
    access_token = (access_token or "").strip()
    if include_private and not access_token:
        raise GitHubError("GitHub authorization is required to include private repositories.")

    endpoint = f"{API}/user/repos" if include_private else f"{API}/users/{username}/repos"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "repo-intel/1.0",
               "X-GitHub-Api-Version": "2022-11-28"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    repositories: list[RepoRef] = []
    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        page = 1
        while True:
            params = {"per_page": 100, "page": page, "sort": "updated"}
            if include_private:
                params.update({"visibility": "all", "affiliation": "owner"})
            else:
                params["type"] = "owner"
            try:
                response = await client.get(endpoint, params=params)
            except httpx.HTTPError as exc:
                raise GitHubError(f"Could not reach GitHub: {exc.__class__.__name__}") from exc
            if response.status_code == 404:
                raise GitHubError(f"GitHub profile @{username} was not found.")
            if response.status_code in (401, 403, 429):
                raise GitHubError("GitHub denied repository access or the API rate limit was reached.")
            if response.status_code != 200:
                raise GitHubError(f"GitHub returned HTTP {response.status_code} while listing repositories.")
            data = response.json()
            if not isinstance(data, list):
                raise GitHubError("GitHub returned an invalid repository list.")
            for repo in data:
                owner = (repo.get("owner") or {}).get("login", "")
                name = repo.get("name", "")
                if (
                    owner.casefold() == username.casefold()
                    and name
                    and (not repository_name or name.casefold() == repository_name.casefold())
                ):
                    repositories.append(RepoRef(owner=owner, name=name))
            if len(data) < 100:
                break
            page += 1
    return repositories
