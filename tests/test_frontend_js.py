"""Drives the Node frontend unit tests (tests/test_frontend.mjs) from
pytest. Skips cleanly when Node isn't available."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_js_units():
    proc = subprocess.run(
        ["node", "tests/test_frontend.mjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
