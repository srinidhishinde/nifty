from pathlib import Path
import subprocess
import sys

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

    raise SystemExit(
        "Repository-local UAT environment not found. "
        "Create it with: python -m venv .venv"
    )


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


def main():
    passed, empty, failed = [], [], []

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

    if failed:
        print("UAT STATUS: FAILED")
        print()
        print("Failed test suites:")
        for group in failed:
            print(f"  - {group}")
        raise SystemExit(1)

    print("UAT STATUS: PASSED")
    if empty:
        print("WARNING: Empty test suites are not production evidence.")


if __name__ == "__main__":
    main()
