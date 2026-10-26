import io
import os
import tempfile
import zipfile

import pytest

from app.config import reset_settings_cache

os.environ["LLM_MODE"] = "mock"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["AI_SERVICE_TOKEN"] = "test-token"
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="chroma-test-")
reset_settings_cache()

SAMPLE_FILES = {
    "README.md": "# Demo\nA tiny demo project.\n",
    ".gitignore": "__pycache__/\n",
    ".env": "SECRET=abc\n",
    "requirements.txt": "fastapi==0.110\nrequests\npsycopg[binary]\npytest\n",
    "Dockerfile": "FROM python:3.12\nCOPY . .\n",
    "app/main.py": (
        "import requests\n"
        "import subprocess\n\n"
        'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"\n'
        'DB = "postgresql://admin:hunter2pass@db:5432/app"\n\n'
        "def temp_fetch(url):\n"
        "    r = requests.get(url)\n"
        "    try:\n"
        "        return r.json()\n"
        "    except:\n"
        "        pass\n\n"
        "def run(cmd):\n"
        "    return subprocess.run(cmd, shell=True)\n"
    ),
    "app/utils.py": "def add(a, b):\n    return a + b\n",
    "node_modules/leftpad/index.js": "module.exports = 1;\n",
}


def build_zip(files: dict[str, str], top: str = "demo-user-demo-abc123") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"{top}/{path}", content)
    return buf.getvalue()


@pytest.fixture
def sample_zip() -> bytes:
    return build_zip(SAMPLE_FILES)


@pytest.fixture
def snapshot(sample_zip):
    from app.github_client import RepoRef, snapshot_from_zip

    return snapshot_from_zip(sample_zip, RepoRef("demo-user", "demo"), default_branch="main", languages={"Python": 900, "Dockerfile": 100})
