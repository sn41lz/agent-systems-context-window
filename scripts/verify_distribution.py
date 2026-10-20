"""Install the built wheel away from the checkout and exercise its bundled tokenizer."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
wheels, archives = list((root / "dist").glob("*.whl")), list((root / "dist").glob("*.tar.gz"))
if len(wheels) != 1 or len(archives) != 1:
    raise SystemExit("Expected exactly one wheel and one source archive; clean dist first")
with zipfile.ZipFile(wheels[0]) as wheel:
    names = wheel.namelist()
    for suffix in (
        "data/o200k_base.tiktoken.gz",
        "data/encoding.json",
        "data/TIKTOKEN_LICENSE.txt",
        "py.typed",
    ):
        assert any(name.endswith(suffix) for name in names), suffix
with tarfile.open(archives[0]) as archive:
    names = archive.getnames()
    for suffix in (
        "requirements.lock",
        "docker-compose.yml",
        "tests/test_modern.py",
        "scripts/verify_distribution.py",
    ):
        assert any(name.endswith(suffix) for name in names), suffix
with tempfile.TemporaryDirectory(prefix="context-wheel-") as directory:
    target = Path(directory) / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            str(wheels[0]),
        ],
        check=True,
        cwd=directory,
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(target),
        "PYTHONNOUSERSITE": "1",
        "TIKTOKEN_CACHE_DIR": str(Path(directory) / "empty-cache"),
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import context_window; from context_window.tokenizer import TokenCounter; assert 'installed' in str(Path(context_window.__file__)); assert TokenCounter().count('offline packaging check') > 0",
        ],
        check=True,
        cwd=directory,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "context_window",
            "demo",
            "--turns",
            "2",
            "--output",
            str(Path(directory) / "artifacts"),
        ],
        check=True,
        cwd=directory,
        env=env,
    )
print("Wheel imports, offline assets and CLI verified outside the source checkout.")
