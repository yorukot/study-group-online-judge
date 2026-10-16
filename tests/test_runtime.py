import os
import subprocess
import sys

from judge.check_runtime import check_runtime


def test_actual_evaluator_runtime_passes_dependency_and_import_checks(capsys, tmp_path):
    # This deliberately checks installed distributions and real imports rather
    # than mocked package installers. CUDA execution has its own GPU tests.
    subprocess.run(
        ["uv", "pip", "check", "--python", sys.executable],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
    )
    check_runtime()
    output = capsys.readouterr().out
    assert "runtime transformers==5." in output
    assert "runtime huggingface-hub==1." in output
    assert "runtime torch==2." in output
    assert "runtime datasets==4." in output
    assert "evaluator runtime imports passed" in output
