"""Technology detection: languages, frameworks, databases, DevOps and cloud."""
from __future__ import annotations

import json
import re
from collections import Counter

from .github_client import RepoSnapshot

# dependency name -> (display name, group)
PY_PACKAGES = {
    "django": ("Django", "frameworks"), "fastapi": ("FastAPI", "frameworks"), "flask": ("Flask", "frameworks"),
    "starlette": ("Starlette", "frameworks"), "streamlit": ("Streamlit", "frameworks"), "tornado": ("Tornado", "frameworks"),
    "langchain": ("LangChain", "ai"), "langgraph": ("LangGraph", "ai"), "openai": ("OpenAI API", "ai"),
    "anthropic": ("Anthropic API", "ai"), "groq": ("Groq", "ai"), "chromadb": ("ChromaDB", "ai"),
    "transformers": ("Transformers", "ai"), "torch": ("PyTorch", "ai"), "tensorflow": ("TensorFlow", "ai"),
    "scikit-learn": ("scikit-learn", "ai"), "pandas": ("Pandas", "libraries"), "numpy": ("NumPy", "libraries"),
    "sqlalchemy": ("SQLAlchemy", "libraries"), "pydantic": ("Pydantic", "libraries"), "celery": ("Celery", "libraries"),
    "requests": ("Requests", "libraries"), "httpx": ("HTTPX", "libraries"),
    "psycopg": ("PostgreSQL", "databases"), "psycopg2": ("PostgreSQL", "databases"), "asyncpg": ("PostgreSQL", "databases"),
    "pymongo": ("MongoDB", "databases"), "motor": ("MongoDB", "databases"), "redis": ("Redis", "databases"),
    "mysqlclient": ("MySQL", "databases"), "pymysql": ("MySQL", "databases"),
    "boto3": ("AWS SDK", "cloud"), "google-cloud-storage": ("Google Cloud", "cloud"), "azure-storage-blob": ("Azure", "cloud"),
    "pytest": ("pytest", "testing"), "gunicorn": ("Gunicorn", "libraries"), "uvicorn": ("Uvicorn", "libraries"),
}
JS_PACKAGES = {
    "react": ("React", "frameworks"), "next": ("Next.js", "frameworks"), "vue": ("Vue", "frameworks"),
    "@angular/core": ("Angular", "frameworks"), "svelte": ("Svelte", "frameworks"), "express": ("Express", "frameworks"),
    "@nestjs/core": ("NestJS", "frameworks"), "fastify": ("Fastify", "frameworks"), "koa": ("Koa", "frameworks"),
    "react-native": ("React Native", "frameworks"), "electron": ("Electron", "frameworks"),
    "mongoose": ("MongoDB", "databases"), "mongodb": ("MongoDB", "databases"), "pg": ("PostgreSQL", "databases"),
    "mysql2": ("MySQL", "databases"), "sqlite3": ("SQLite", "databases"), "redis": ("Redis", "databases"),
    "prisma": ("Prisma", "libraries"), "@prisma/client": ("Prisma", "libraries"), "sequelize": ("Sequelize", "libraries"),
    "typeorm": ("TypeORM", "libraries"), "axios": ("Axios", "libraries"), "redux": ("Redux", "libraries"),
    "tailwindcss": ("Tailwind CSS", "libraries"), "typescript": ("TypeScript", "libraries"),
    "socket.io": ("Socket.IO", "libraries"), "jsonwebtoken": ("JWT", "libraries"), "passport": ("Passport", "libraries"),
    "firebase": ("Firebase", "cloud"), "aws-sdk": ("AWS SDK", "cloud"),
    "jest": ("Jest", "testing"), "vitest": ("Vitest", "testing"), "mocha": ("Mocha", "testing"), "cypress": ("Cypress", "testing"),
    "vite": ("Vite", "devops"), "webpack": ("Webpack", "devops"),
}
IMAGE_HINTS = {"postgres": "PostgreSQL", "mysql": "MySQL", "mariadb": "MariaDB", "mongo": "MongoDB",
               "redis": "Redis", "rabbitmq": "RabbitMQ", "elasticsearch": "Elasticsearch", "nginx": "Nginx"}
GROUP_ORDER = ["frameworks", "ai", "databases", "libraries", "cloud", "devops", "testing"]


def _py_dep_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m and not line.startswith("-"):
            names.add(m.group(1).lower().replace("_", "-"))
    return names


def _pyproject_names(text: str) -> set[str]:
    return {m.lower().replace("_", "-") for m in re.findall(r"""["']\s*([A-Za-z0-9_.\-]+)\s*(?:[<>=!~\[;]|["'])""", text)}


def _js_dep_names(text: str) -> set[str]:
    try:
        pkg = json.loads(text)
    except (ValueError, TypeError):
        return set()
    names: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        deps = pkg.get(key)
        if isinstance(deps, dict):
            names.update(deps.keys())
    return names


def detect_technology(snap: RepoSnapshot) -> dict:
    groups: dict[str, set[str]] = {g: set() for g in GROUP_ORDER}

    def add(name: str, group: str) -> None:
        groups[group].add(name)

    py_names: set[str] = set()
    js_names: set[str] = set()
    for path, content in snap.files.items():
        base = path.rsplit("/", 1)[-1].lower()
        if re.match(r"requirements.*\.txt$", base):
            py_names |= _py_dep_names(content)
        elif base == "pyproject.toml":
            py_names |= _pyproject_names(content)
        elif base == "pipfile":
            py_names |= _pyproject_names(content) | _py_dep_names(content)
        elif base == "package.json" and "node_modules" not in path:
            js_names |= _js_dep_names(content)

    for dep in py_names:
        base_dep = re.split(r"[\[<>=!~]", dep)[0]
        if base_dep in PY_PACKAGES:
            add(*PY_PACKAGES[base_dep])
    for dep in js_names:
        if dep in JS_PACKAGES:
            add(*JS_PACKAGES[dep])
        elif dep.startswith("@aws-sdk/"):
            add("AWS SDK", "cloud")

    lowered = {p.lower() for p in snap.all_paths}
    names = {p.rsplit("/", 1)[-1] for p in lowered}
    if any(n.startswith("dockerfile") for n in names):
        add("Docker", "devops")
    compose = [p for p in snap.files if p.rsplit("/", 1)[-1].lower().startswith(("docker-compose", "compose."))]
    if compose:
        add("Docker Compose", "devops")
        for p in compose:
            for key, label in IMAGE_HINTS.items():
                if re.search(rf"image:\s*['\"]?[\w./-]*{key}", snap.files[p], re.I):
                    add(label, "databases" if label in {"PostgreSQL", "MySQL", "MariaDB", "MongoDB", "Redis", "Elasticsearch"} else "devops")
    if any(p.startswith(".github/workflows/") for p in lowered):
        add("GitHub Actions", "devops")
    if ".gitlab-ci.yml" in lowered:
        add("GitLab CI", "devops")
    if any(p.endswith(".tf") for p in lowered):
        add("Terraform", "devops")
    if "serverless.yml" in names or "serverless.yaml" in names:
        add("Serverless Framework", "cloud")
    if "vercel.json" in names:
        add("Vercel", "cloud")
    if "netlify.toml" in names:
        add("Netlify", "cloud")
    if "procfile" in names:
        add("Heroku", "cloud")
    if any(p.endswith(("k8s.yaml", "deployment.yaml")) or "/k8s/" in p or "/kubernetes/" in p for p in lowered):
        add("Kubernetes", "devops")
    if "manage.py" in names:
        add("Django", "frameworks")
    if any(n.endswith(".sqlite3") or n.endswith(".db") for n in names):
        add("SQLite", "databases")
    if any(re.search(r"\b(?:boto3|aws-sdk|amazonaws\.com)\b", c) for p, c in snap.files.items() if p.endswith((".py", ".js", ".ts", ".yml", ".yaml", ".tf"))):
        add("AWS", "cloud")
    if "AWS SDK" in groups["cloud"]:
        add("AWS", "cloud")

    return {
        "languages": _languages(snap),
        "technologies": {g: sorted(v) for g, v in groups.items() if v},
        "all": sorted({t for v in groups.values() for t in v}),
    }


def _languages(snap: RepoSnapshot) -> list[dict]:
    data = dict(snap.languages_bytes)
    if not data:  # fall back to counting bytes of files by extension
        ext_lang = {".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
                    ".java": "Java", ".go": "Go", ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".rs": "Rust",
                    ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".kt": "Kotlin", ".swift": "Swift", ".vue": "Vue",
                    ".c": "C", ".cpp": "C++", ".sh": "Shell"}
        counter: Counter[str] = Counter()
        for path, content in snap.files.items():
            dot = path.rfind(".")
            lang = ext_lang.get(path[dot:].lower()) if dot != -1 else None
            if lang:
                counter[lang] += len(content)
        data = dict(counter)
    total = sum(data.values())
    if not total:
        return []
    ranked = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    return [{"name": n, "percent": round(v * 100 / total, 1)} for n, v in ranked[:8]]
