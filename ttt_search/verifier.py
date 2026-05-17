"""Verifies a generated solution by running HumanEval tests in a subprocess."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile


def verify(solution: str, test_code: str, entry_point: str, timeout: int = 10) -> tuple[bool, str | None]:
    """
    Returns (passed, error_message).
    Never executes arbitrary code in the main process.
    """
    code = solution.strip()

    # Pre-flight: must compile
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Pre-flight: entry_point must be defined
    if not re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE):
        return False, f"entry_point '{entry_point}' not defined"

    # Build script without any added indentation — code is already well-formed
    script = code + "\n\n" + test_code + "\n\ncheck(" + entry_point + ")\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name

    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, None
        err = (result.stderr or result.stdout).strip()
        return False, err
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    finally:
        import os
        os.unlink(fname)
