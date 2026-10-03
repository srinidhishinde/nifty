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


def resolve_python() -> str:
    """Resolve the repository-local Python interpreter.

    UAT must not silently use an unrelated virtual environment that happens
    to be activated in the parent shell.
    """
    repo_root = Path(__file__).resolve().parents[1]

    if sys.platform == "win32":
        candidates = [
            repo_root / ".venv" / "Scripts" / "python.exe",
            repo_root / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            repo_root / ".venv" / "bin" / "python",
            repo_root / "venv" / "bin" / "python",
        ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise SystemExit(
        "UAT environment not found. Create the repository-local virtual "
        "environment first with: python -m venv .venv"
    )


PYTHON = resolve_python()


def run_group(group: str) -> tuple[str, int, int]:
    """
    Returns:
        (status, collected_tests, exit_code)

    status:
        PASS       -> tests existed and passed
        EMPTY      -> no tests currently exist
        FAIL       -> one or more tests failed
    """

    print()
    print("=" * 60)
    print(f"Running {group}")
    print("=" * 60)

    result = subprocess.run(
        [
            PYTHON,
            "-m",
            "pytest",
            group,
            "-v",
            "--tb=short",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    print(output)

    # Pytest returns exit code 5 when no tests are collected.
    if result.returncode == 5:
        return "EMPTY", 0, result.returncode

    if result.returncode != 0:
        return "FAIL", 0, result.returncode

    # Basic collection detection.
    collected = 0

    for line in output.splitlines():
        if "collected" in line and "item" in line:
            try:
                collected = int(
                    line.split("collected")[1]
                    .split("item")[0]
                    .strip()
                )
            except (ValueError, IndexError):
                pass

    return "PASS", collected, result.returncode


def main():
    passed = []
    empty = []
    failed = []

    print(f"UAT Python: {PYTHON}")
    print(f"Python version: {subprocess.check_output([PYTHON, '--version'], text=True).strip()}")

    for group in TEST_GROUPS:
        status, collected, _ = run_group(group)

        if status == "PASS":
            passed.append(group)
        elif status == "EMPTY":
            empty.append(group)
        else:
            failed.append(group)

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

    print("UAT STATUS: CURRENT TESTS PASSED")
    print()
    print(
        "Note: EMPTY suites are not considered passed. "
        "They require implementation before production release."
    )


if __name__ == "__main__":
    main()
