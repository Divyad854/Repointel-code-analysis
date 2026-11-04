"""Deterministic code-quality checks (no LLM).

Detects long functions, excessive parameters, deep nesting, ignored exceptions,
large files, TODO/FIXME markers, and missing repository-level practices.
All findings produced here use source="rule".
"""
from __future__ import annotations
# AST understands that this code contains a function and nested if statements
import ast
import re

from .schemas import Finding

# File extensions whose contents are treated as source code by this scanner.
SOURCE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".go", ".rb", ".php", ".cs", ".rs",
    ".kt", ".swift", ".c", ".h", ".cpp", ".hpp", ".cc", ".vue", ".svelte",
}

# Python AST node types that increase the measured control-flow nesting depth.
_NESTING_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)

# Regex for an empty JavaScript/TypeScript-style catch block.
_EMPTY_CATCH = re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*\}")

# Recognizes common test directory and test filename conventions.
_TEST_FILE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec)(?:/|$)|(?:^|/)test_[^/]+$|[^/]+_test\.\w+$|\.(?:test|spec)\.\w+$", re.I)


def is_source_file(path: str) -> bool:
    """Return True when the path has one of the supported source extensions."""
    name = path.rsplit("/", 1)[-1]  # Ignore parent directories and inspect the filename.
    return "." in name and "." + name.rsplit(".", 1)[-1].lower() in SOURCE_EXTS


def is_test_file(path: str) -> bool:
    """Identify test files so their intentional examples are not scanned as production code."""
    return bool(_TEST_FILE.search(path))


def _max_depth(node: ast.AST, depth: int = 0) -> int:
    """Recursively calculate the deepest nested control structure below an AST node."""
    best = depth
    for child in ast.iter_child_nodes(node):  # Visit each child node in the syntax tree.
        # Increase depth only for control-flow/context nodes listed in _NESTING_NODES.
        d = depth + 1 if isinstance(child, _NESTING_NODES) else depth
        best = max(best, _max_depth(child, d))  # Keep the deepest value found in any branch.
    return best


def _python_checks(path: str, content: str) -> list[Finding]:
    """Run Python AST checks for long functions, parameters, nesting and exception handling."""
    out: list[Finding] = []
    try:
        tree = ast.parse(content)  # Parse source into an AST so functions/handlers can be inspected.
    except (SyntaxError, ValueError, RecursionError):
        return out  # Skip files that cannot be parsed; do not fail the whole repository scan.
    lines = content.splitlines()

    def text(n: int) -> str:
        # Return a short source snippet for evidence, using Python's 1-based line numbers.
        return lines[n - 1].strip()[:200] if 0 < n <= len(lines) else ""

    for node in ast.walk(tree):  # Inspect every node in the parsed Python file.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function length includes its definition line and final body line.
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > 60:
                out.append(Finding(
                    file=path, line=node.lineno, category="quality", source="rule", confidence=0.9,
                    severity="high" if length > 150 else "medium",
                    issue=f"Function '{node.name}' is {length} lines long", evidence=text(node.lineno),
                    reason="Very long functions do several jobs, are hard to test and hide bugs.",
                    recommendation=f"Split '{node.name}' into smaller functions that each do one thing.",
                ))

            # Count positional and keyword-only parameters; *args/**kwargs are not included here.
            n_args = len(node.args.args) + len(node.args.kwonlyargs)
            if n_args > 7:
                out.append(Finding(
                    file=path, line=node.lineno, category="quality", source="rule", confidence=0.85, severity="low",
                    issue=f"Function '{node.name}' takes {n_args} parameters", evidence=text(node.lineno),
                    reason="Long parameter lists make call sites error-prone and hint at a missing abstraction.",
                    recommendation="Group related parameters into a dataclass or settings object.",
                ))

            # Use the recursive AST helper to detect deeply nested control structures.
            depth = _max_depth(node)
            if depth > 4:
                out.append(Finding(
                    file=path, line=node.lineno, category="quality", source="rule", confidence=0.85, severity="medium",
                    issue=f"Function '{node.name}' has {depth} levels of nesting", evidence=text(node.lineno),
                    reason="Deeply nested branches are hard to read and reason about.",
                    recommendation="Use early returns (guard clauses) and extract helper functions.",
                ))

        elif isinstance(node, ast.ExceptHandler):
            # An exception handler is considered silently ignored when its only statement is pass/continue.
            only_pass = len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue))
            if node.type is None:  # No exception type means a bare 'except:'.
                out.append(Finding(
                    file=path, line=node.lineno, category="quality", source="rule", confidence=0.9, severity="medium",
                    issue="Bare 'except:' clause", evidence=text(node.lineno),
                    reason="A bare except also catches KeyboardInterrupt/SystemExit and hides real bugs.",
                    recommendation="Catch the specific exception types you expect and log or handle them.",
                ))
            elif only_pass:
                out.append(Finding(
                    file=path, line=node.lineno, category="quality", source="rule", confidence=0.85, severity="medium",
                    issue="Exception is silently swallowed", evidence=text(node.lineno),
                    reason="Ignoring an exception makes failures invisible and debugging very hard.",
                    recommendation="Log the exception or handle it explicitly instead of passing.",
                ))
    return out


def _generic_checks(path: str, content: str) -> list[Finding]:
    """Run line-count, empty-catch and TODO/FIXME checks that do not require Python AST."""
    out: list[Finding] = []
    lines = content.splitlines()
    n = len(lines)

    # Python files use an 800-line threshold; other files use 600 lines.
    if n > 600 and not path.endswith(".py") or n > 800:
        out.append(Finding(
            file=path, line=1, category="quality", source="rule", confidence=0.8, severity="low",
            issue=f"Very large file ({n} lines)", evidence=lines[0].strip()[:200] if lines else "",
            reason="Large files accumulate unrelated responsibilities and are hard to navigate.",
            recommendation="Split the file into cohesive modules.",
        ))

    # The empty-catch regex is applied to non-Python files (for example JS/TS).
    if not path.endswith(".py"):
        for m in _EMPTY_CATCH.finditer(content):
            line = content.count("\n", 0, m.start()) + 1  # Convert match offset to a 1-based line number.
            out.append(Finding(
                file=path, line=line, category="quality", source="rule", confidence=0.85, severity="medium",
                issue="Empty catch block swallows errors", evidence=lines[line - 1].strip()[:200],
                reason="Swallowing exceptions hides failures.",
                recommendation="Log the error or handle it explicitly.",
            ))

    # Count lines containing TODO/FIXME/HACK/XXX; report when one file has more than ten.
    todos = [i + 1 for i, l in enumerate(lines) if re.search(r"\b(?:TODO|FIXME|HACK|XXX)\b", l)]
    if len(todos) > 10:
        out.append(Finding(
            file=path, line=todos[0], category="quality", source="rule", confidence=0.7, severity="low",
            issue=f"{len(todos)} TODO/FIXME markers in one file", evidence=lines[todos[0] - 1].strip()[:200],
            reason="Many unresolved markers indicate unfinished or fragile code.",
            recommendation="Turn the markers into tracked issues and resolve or remove them.",
        ))
    return out


def run_static_checks(files: dict[str, str], all_paths: list[str]) -> list[Finding]:
    """Run file-level and repository-level deterministic quality checks."""
    findings: list[Finding] = []

    # Scan available file contents, excluding unsupported files and recognized test files.
    for path, content in files.items():
        if not is_source_file(path) or is_test_file(path):
            continue
        if path.endswith(".py"):
            findings.extend(_python_checks(path, content))  # Python-only AST checks.
        findings.extend(_generic_checks(path, content))  # Checks shared across source languages.

    # Repository-level checks use the complete path list, not only files with loaded contents.
    source_paths = [p for p in all_paths if is_source_file(p)]

    # Report missing tests only when the repository contains source code but no recognized test paths.
    if source_paths and not any(is_test_file(p) for p in all_paths):
        findings.append(Finding(
            file="(repository)", category="quality", severity="medium", source="rule", confidence=0.8,
            issue="No automated tests found", evidence="No test directories or test files detected",
            reason="Without tests, changes cannot be verified and regressions go unnoticed.",
            recommendation="Add unit tests for the core logic and run them in CI.",
        ))

    # Recognize a few common CI configuration locations/names.
    if source_paths and not any(p.startswith(".github/workflows/") or p in (".gitlab-ci.yml", "Jenkinsfile", ".circleci/config.yml") for p in all_paths):
        findings.append(Finding(
            file="(repository)", category="quality", severity="low", source="rule", confidence=0.6,
            issue="No CI configuration found", evidence="No .github/workflows, .gitlab-ci.yml or similar",
            reason="Automated checks catch problems before they are merged.",
            recommendation="Add a CI workflow that runs tests and linters on every push.",
        ))

    # Check for a root-level .gitignore to help prevent accidental commits of generated files/secrets.
    if source_paths and ".gitignore" not in all_paths:
        findings.append(Finding(
            file="(repository)", category="quality", severity="low", source="rule", confidence=0.9,
            issue="Missing .gitignore", evidence="No .gitignore at the repository root",
            reason="Without it, build artefacts, virtual environments and secrets are easily committed by accident.",
            recommendation="Add a .gitignore for your language/framework, including .env.",
        ))
    return findings
