import json
import subprocess
import sys
from pathlib import Path


def test_reporting_redirects_cached_native_and_child_output_and_restores_on_error(
    tmp_path,
):
    log = tmp_path / "reporting.log"
    script = """
import json
import os
import subprocess
import sys
from pathlib import Path
from judge.remote_store import reporting_output

cached_stdout = sys.stdout
cached_stderr = sys.stderr
try:
    with reporting_output(Path(sys.argv[1])):
        print('python console')
        cached_stdout.write('cached stdout\\n')
        cached_stderr.write('cached stderr\\n')
        os.write(1, b'native stdout\\n')
        os.write(2, b'native stderr\\n')
        subprocess.run([sys.executable, '-c', "print('child console')"], check=True)
        raise RuntimeError('SDK failure')
except RuntimeError:
    pass
print(json.dumps({'report_pending': True}))
print('stderr restored', file=sys.stderr)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(log)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"report_pending": True}
    assert result.stderr == "stderr restored\n"
    contents = log.read_text()
    for message in (
        "python console",
        "cached stdout",
        "cached stderr",
        "native stdout",
        "native stderr",
        "child console",
    ):
        assert message in contents


def test_real_wandb_capture_keeps_console_logs_and_ssh_json_separate(tmp_path):
    script = """
import json
import sys
from pathlib import Path
import wandb
from judge.remote_store import reporting_output

root = Path(sys.argv[1])
stdout = sys.stdout
stderr = sys.stderr
with reporting_output(root / 'reporting.log'):
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    run = wandb.init(
        project='judge-console-regression',
        dir=str(root),
        settings=wandb.Settings(mode='offline', console='wrap'),
    )
    print('[judge] repository ready', flush=True)
    print('[lab4] corpus perplexity=35.6410', flush=True)
    print('[judge] verdict: PASS', flush=True)
    directory = run.settings.files_dir
    run.finish()
print(json.dumps({'files_dir': directory}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    files = Path(json.loads(result.stdout)["files_dir"])
    # The core stores raw console records in the offline sync file; output.log
    # is materialized when those records are uploaded to W&B.
    console = next(files.parent.glob("run-*.wandb")).read_bytes()
    assert b"[judge] repository ready" in console
    assert b"[lab4] corpus perplexity=35.6410" in console
    assert b"[judge] verdict: PASS" in console
    assert result.stderr == ""
