"""Deterministic security scanning (no LLM): secret patterns and risky code patterns.

The LLM is good at judgement, bad at being certain. These regex scanners are fast,
repeatable and cheap, and their findings carry source="rule".
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .schemas import Finding

# --------------------------------------------------------------------- secrets
KNOWN_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("AWS access key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Stripe live secret key", re.compile(r"\b[sr]k_live_[0-9a-zA-Z]{20,}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----")),
]

_GENERIC_ASSIGN = re.compile(
    r"""(?ix)
    (?P<key>[\w.-]*(?:api[_-]?key|secret|passwd|password|pwd|token|auth[_-]?key|access[_-]?key|private[_-]?key)[\w.-]*)
    \s*[:=]\s*
    (?P<q>["'])(?P<val>[^"'\s]{6,})(?P=q)
    """
)
_DB_URL = re.compile(r"""(?ix)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s/@"']+:(?P<pw>[^@\s"']+)@""")

_PLACEHOLDER = re.compile(
    r"""(?ix)^(?:
        your[_-].*|.*changeme.*|.*change[_-]me.*|.*example.*|.*placeholder.*|.*xxx+.*|\*+|<.*>|\$\{.*\}|\{\{.*\}\}|
        test.*|dummy.*|fake.*|sample.*|secret|password|token|none|null|true|false|todo.*|replace.*|insert.*|
        os\.environ.*|process\.env.*|env\(.*|getenv.*|.*\.\.\..*|%s|%\(.*
    )$"""
)


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER.match(value.strip()))


def _mask(value: str) -> str:
    return (value[:4] + "****") if len(value) > 4 else "****"


def redact_secrets(text: str) -> str:
    """Mask secrets so they are neither sent to the LLM nor repeated in reports."""
    for _, pat in KNOWN_SECRET_PATTERNS:
        text = pat.sub(lambda m: _mask(m.group(0)), text)

    def _generic(m: re.Match) -> str:
        val = m.group("val")
        if _is_placeholder(val):
            return m.group(0)
        return m.group(0).replace(val, _mask(val))

    text = _GENERIC_ASSIGN.sub(_generic, text)
    text = _DB_URL.sub(lambda m: m.group(0).replace(m.group("pw"), "****") if not _is_placeholder(m.group("pw")) else m.group(0), text)
    return text


# ------------------------------------------------------------- risky patterns
@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    exts: tuple[str, ...] | None
    severity: str
    issue: str
    reason: str
    recommendation: str
    confidence: float = 0.75


RULES: list[Rule] = [
    Rule("py-eval", re.compile(r"(?<![\w.])eval\s*\("), (".py",), "high",
         "Use of eval()", "eval() executes arbitrary code; if any part of the input is attacker-controlled this is remote code execution.",
         "Replace eval() with ast.literal_eval() or an explicit parser.", 0.8),
    Rule("py-exec", re.compile(r"(?<![\w.])exec\s*\("), (".py",), "high",
         "Use of exec()", "exec() runs arbitrary code strings and is a common code-injection vector.",
         "Avoid exec(); use explicit dispatch tables or safe parsers.", 0.75),
    Rule("py-shell-true", re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True"), (".py",), "high",
         "subprocess called with shell=True", "shell=True passes the command through a shell and enables command injection when any argument is user-controlled.",
         "Pass the command as a list and drop shell=True.", 0.8),
    Rule("py-os-system", re.compile(r"\bos\.system\s*\("), (".py",), "medium",
         "Use of os.system()", "os.system() runs through a shell and is vulnerable to command injection.",
         "Use subprocess.run([...]) with an argument list.", 0.7),
    Rule("py-pickle", re.compile(r"\bpickle\.loads?\s*\("), (".py",), "high",
         "Unsafe pickle deserialization", "Unpickling untrusted data can execute arbitrary code.",
         "Use JSON or another safe format for untrusted data.", 0.7),
    Rule("py-verify-false", re.compile(r"verify\s*=\s*False"), (".py",), "medium",
         "TLS certificate verification disabled", "Disabling verification allows man-in-the-middle attacks.",
         "Remove verify=False and fix the certificate chain instead.", 0.85),
    Rule("py-sql-format", re.compile(r"""\.execute\(\s*(?:f["']|["'][^"']*["']\s*(?:%|\+|\.format))"""), (".py",), "high",
         "SQL built with string formatting", "Interpolating values into SQL strings enables SQL injection.",
         "Use parameterised queries: cursor.execute(sql, params).", 0.8),
    Rule("py-md5", re.compile(r"hashlib\.(?:md5|sha1)\s*\("), (".py",), "low",
         "Weak hash function (MD5/SHA-1)", "MD5 and SHA-1 are broken for security purposes such as passwords or signatures.",
         "Use SHA-256+ for integrity and bcrypt/argon2 for passwords.", 0.5),
    Rule("django-debug", re.compile(r"^\s*DEBUG\s*=\s*True\b", re.M), (".py",), "medium",
         "DEBUG enabled in settings", "DEBUG=True in production leaks stack traces and settings.",
         "Read DEBUG from an environment variable and default it to False.", 0.6),
    Rule("django-allowed-hosts", re.compile(r"ALLOWED_HOSTS\s*=\s*\[\s*['\"]\*['\"]\s*\]"), (".py",), "medium",
         "ALLOWED_HOSTS allows every host", "A wildcard host list disables Host-header protection.",
         "List the real hostnames in ALLOWED_HOSTS.", 0.7),
    Rule("cors-wildcard", re.compile(r"""(?:allow_origins\s*=\s*\[\s*["']\*["']|CORS_ALLOW_ALL_ORIGINS\s*=\s*True|Access-Control-Allow-Origin["']?\s*[:,]\s*["']?\*)"""), None, "medium",
         "CORS allows every origin", "A wildcard CORS policy lets any website call the API from a visitor's browser.",
         "Restrict allowed origins to the front-end domains you control.", 0.7),
    Rule("js-eval", re.compile(r"(?<![\w.])eval\s*\("), (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"), "high",
         "Use of eval()", "eval() executes arbitrary code and enables injection attacks.",
         "Use JSON.parse or a proper parser instead.", 0.8),
    Rule("js-inner-html", re.compile(r"\.innerHTML\s*=(?!=)|dangerouslySetInnerHTML|document\.write\s*\("), (".js", ".jsx", ".ts", ".tsx", ".vue", ".html"), "medium",
         "Raw HTML injection (XSS risk)", "Writing unsanitised strings as HTML allows cross-site scripting.",
         "Use textContent or framework bindings, or sanitise with DOMPurify.", 0.6),
    Rule("js-child-process", re.compile(r"child_process\.exec(?:Sync)?\s*\(|\bexecSync\s*\("), (".js", ".ts", ".mjs", ".cjs"), "high",
         "Shell command execution", "exec() runs through a shell and enables command injection with unsanitised input.",
         "Use execFile/spawn with an argument array.", 0.7),
    Rule("js-tls-off", re.compile(r"rejectUnauthorized\s*:\s*false"), (".js", ".ts", ".mjs", ".cjs"), "medium",
         "TLS certificate verification disabled", "Disabling verification allows man-in-the-middle attacks.",
         "Remove rejectUnauthorized: false and fix the certificate chain.", 0.85),
    Rule("docker-root", re.compile(r"^\s*USER\s+root\b", re.M | re.I), (".dockerfile",), "low",
         "Container runs as root", "Running as root increases the impact of a container escape.",
         "Create and switch to an unprivileged user.", 0.5),
]

_SENSITIVE_FILES = re.compile(r"(?:^|/)(?:\.env(?:\.(?!example$|sample$|template$)[\w.-]+)?|id_rsa|id_dsa|id_ed25519|[^/]+\.pem|[^/]+\.p12|[^/]+\.pfx|[^/]+\.keystore)$", re.I)
_MAX_PER_RULE = 5


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _line_text(lines: list[str], line: int) -> str:
    return lines[line - 1].strip()[:240] if 0 < line <= len(lines) else ""


def _is_test_path(path: str) -> bool:
    p = path.lower()
    return any(x in p for x in ("/test/", "/tests/", "test_", "_test.", ".test.", ".spec.", "/__tests__/", "/fixtures/", "/examples/", "/docs/"))


def scan_repository(files: dict[str, str], all_paths: list[str]) -> list[Finding]:
    """Run every deterministic security check over the repository."""
    findings: list[Finding] = []
    per_rule: dict[str, int] = defaultdict(int)

    # 1. Sensitive files committed to the repository (reported by path; never read)
    for path in all_paths:
        if _SENSITIVE_FILES.search(path):
            findings.append(Finding(
                file=path, category="security", severity="high", source="rule", confidence=0.9,
                issue="Sensitive file committed to the repository",
                evidence=path,
                reason="Environment, key and certificate files usually contain secrets and stay in git history forever.",
                recommendation="Remove it from git, add it to .gitignore, rotate any credentials it contained and commit a .env.example instead.",
            ))

    for path, content in files.items():
        lower = path.lower()
        name = lower.split("/")[-1]
        ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        if name.startswith("dockerfile"):
            ext = ".dockerfile"
        is_test = _is_test_path(path)
        lines = content.splitlines()

        # 2. Secrets
        for label, pat in KNOWN_SECRET_PATTERNS:
            for m in pat.finditer(content):
                if per_rule[label] >= _MAX_PER_RULE:
                    break
                per_rule[label] += 1
                line = _line_of(content, m.start())
                findings.append(Finding(
                    file=path, line=line, category="security", severity="critical", source="rule", confidence=0.95,
                    issue=f"Hardcoded {label}",
                    evidence=redact_secrets(_line_text(lines, line)),
                    reason="Anyone with read access to the repository (or its history) can use this credential.",
                    recommendation="Revoke and rotate the credential, then load it from an environment variable or a secrets manager.",
                ))

        if not name.endswith((".md", ".rst", ".txt")) and not name.startswith(".env."):
            for m in _GENERIC_ASSIGN.finditer(content):
                val = m.group("val")
                if _is_placeholder(val) or per_rule["generic-secret"] >= _MAX_PER_RULE:
                    continue
                # skip file names and URLs that merely sit next to a "token"/"key" word
                if re.search(r"(?i)(?:\.|_)(?:html|css|js|py|png|svg|json|txt)$", val) or val.startswith(("http://", "https://", "/")):
                    continue
                per_rule["generic-secret"] += 1
                line = _line_of(content, m.start())
                sev = "medium" if is_test else "high"
                findings.append(Finding(
                    file=path, line=line, category="security", severity=sev, source="rule", confidence=0.55 if is_test else 0.65,
                    issue="Possible hardcoded credential",
                    evidence=redact_secrets(_line_text(lines, line)),
                    reason=f"The variable '{m.group('key')}' is assigned a literal string value, which looks like a secret.",
                    recommendation="Read the value from an environment variable or secrets manager and keep it out of source control.",
                ))

        for m in _DB_URL.finditer(content):
            if _is_placeholder(m.group("pw")) or per_rule["db-url"] >= _MAX_PER_RULE:
                continue
            per_rule["db-url"] += 1
            line = _line_of(content, m.start())
            findings.append(Finding(
                file=path, line=line, category="security", severity="high", source="rule", confidence=0.8,
                issue="Database URL with embedded password",
                evidence=redact_secrets(_line_text(lines, line)),
                reason="Connection strings with credentials in source control expose the database.",
                recommendation="Move the connection string to an environment variable.",
            ))

        # 3. Risky code patterns
        if is_test:
            continue
        for rule in RULES:
            if rule.exts is not None and ext not in rule.exts:
                continue
            for m in rule.pattern.finditer(content):
                if per_rule[rule.id] >= _MAX_PER_RULE:
                    break
                per_rule[rule.id] += 1
                line = _line_of(content, m.start())
                findings.append(Finding(
                    file=path, line=line, category="security", severity=rule.severity,  # type: ignore[arg-type]
                    source="rule", confidence=rule.confidence, issue=rule.issue,
                    evidence=redact_secrets(_line_text(lines, line)), reason=rule.reason,
                    recommendation=rule.recommendation,
                ))
    return findings
