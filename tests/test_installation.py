import os
from pathlib import Path
import subprocess
import sys


def test_build_reports_missing_cudss_development_files(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["CUDSS_ROOT"] = os.fspath(tmp_path)

    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "cuDSS development files were not found" in output
    assert os.fspath(tmp_path / "include" / "cudss.h") in output
    assert os.fspath(tmp_path / "lib64" / "libcudss.so") in output
    assert "set CUDSS_ROOT to its installation prefix" in output
