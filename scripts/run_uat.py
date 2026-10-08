from pathlib import Path
import subprocess
import sys
import json
import hashlib
from datetime import datetime, timezone

TEST_GROUPS = [
    "tests/unit",
    "tests/integration",
    "tests/strategy",
    "tests/backtest",
    "tests/broker",
    "tests/security",
    "tests/uat",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_python() -> str:
    """Use the repository-local virtual environment, never a parent-shell venv."""
    if sys.platform == "win32":
        candidates = [
            REPO_ROOT / ".venv" / "Scripts" / "python.exe",
            REPO_ROOT / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            REPO_ROOT / ".venv" / "bin" / "python",
            REPO_ROOT / "venv" / "bin" / "python",
        ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # CI environments already provide the intended interpreter. Falling back
    # to it keeps the UAT harness portable while still preferring the repo venv
    # for local Windows/macOS/Linux runs.
    return sys.executable


PYTHON = resolve_python()


def run_group(group: str) -> str:
    print()
    print("=" * 60)
    print(f"Running {group}")
    print("=" * 60)

    result = subprocess.run(
        [PYTHON, "-m", "pytest", group, "-v", "--tb=short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output)

    if result.returncode == 5:
        return "EMPTY"
    if result.returncode != 0:
        return "FAIL"
    return "PASS"


def suite_fingerprint() -> str:
    digest = hashlib.sha256()
    for group in TEST_GROUPS:
        path = REPO_ROOT / group
        if path.exists():
            for item in sorted(path.rglob('*.py')):
                digest.update(str(item.relative_to(REPO_ROOT)).encode())
                digest.update(item.read_bytes())
    return digest.hexdigest()

def git_head() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ''

def main():
    passed, empty, failed = [], [], []
    head = git_head()
    fingerprint = suite_fingerprint()

    print(f"UAT Python: {PYTHON}")
    print(
        "Python version: "
        + subprocess.check_output([PYTHON, "--version"], text=True).strip()
    )

    for group in TEST_GROUPS:
        status = run_group(group)
        {"PASS": passed, "EMPTY": empty, "FAIL": failed}[status].append(group)

    print()
    print("=" * 60)
    print("UAT SUMMARY")
    print("=" * 60)
    print()

    for group in passed:
        print(f"PASS   {group}")
    for group in empty:
        print(f"EMPTY  {group}")
    for group in failed:
        print(f"FAIL   {group}")

    print()

    evidence_path = REPO_ROOT / "logs" / "uat_evidence.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    overall_pass = not failed and not empty
    evidence_path.write_text(json.dumps({
        "status": "PASS" if overall_pass else "FAIL",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": PYTHON,
        "git_head": head,
        "suite_fingerprint": fingerprint,
        "passed": passed,
        "empty": empty,
        "failed": failed,
    }, indent=2), encoding="utf-8")

    if not overall_pass:
        print("UAT STATUS: FAILED")
        if empty:
            print("Required suites were empty; empty suites are not production evidence.")
        for group in failed:
            print(f"  - {group}")
        raise SystemExit(1)

    print("UAT STATUS: PASSED")


if __name__ == "__main__":
    main()
