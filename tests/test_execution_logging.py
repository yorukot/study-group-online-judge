import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from judge.execution_logging import OUTPUT_LIMIT, log_event, safe_text


class ExecutionLoggingTests(unittest.TestCase):
    def test_output_limit_counts_utf8_bytes(self):
        text = safe_text("模" * OUTPUT_LIMIT)
        self.assertTrue(text.endswith(" [truncated]"))
        self.assertLessEqual(
            len(text.removesuffix(" [truncated]").encode()), OUTPUT_LIMIT
        )

    def test_redacts_credentials_before_truncating_and_escapes_newlines(self):
        secret = "configured-wandb-credential"
        with patch.dict(os.environ, {"WANDB_API_KEY": secret}, clear=True):
            text = safe_text("x" * (OUTPUT_LIMIT - 5) + secret)
            self.assertNotIn(secret[:5], text)
            self.assertTrue(text.endswith(" [truncated]"))
            with self.assertLogs("judge.execution", level="INFO") as logs:
                log_event("test.response", stderr=f"first\n{secret}\nhf_hidden123")
        self.assertNotIn(secret, logs.output[0])
        self.assertNotIn("hf_hidden123", logs.output[0])
        self.assertIn(r"first\n[REDACTED]", logs.output[0])
        self.assertNotIn("\n", logs.output[0])

    def test_remote_entrypoint_keeps_stdout_json_and_logs_to_stderr(self):
        request = json.dumps(
            {
                "job_id": "1" * 32,
                "submission": {
                    "repo_url": "https://github.com/example/repo.git",
                    "commit_sha": "a" * 40,
                    "task_id": "lab2",
                    "github_actor": "student",
                },
                "resources": {"gpus": 1},
                "config": {"host": "nano4", "user": "judge", "work_root": "/work/oj"},
            }
        )
        script = """
from unittest.mock import patch
from judge.execution_logging import configure_logging, log_event
from judge.remote_job import main
from judge.ssh import RemoteSnapshot

def handle(request):
    log_event('remote.request.received', job_id=request.job_id)
    return RemoteSnapshot(slurm_job_id='123', slurm_state='PENDING')

configure_logging()
configure_logging()
with patch('judge.remote_job.handle', side_effect=handle):
    main()
"""
        process = subprocess.run(
            [sys.executable, "-c", script],
            input=request,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        self.assertEqual(json.loads(process.stdout)["slurm_job_id"], "123")
        self.assertEqual(process.stderr.count("remote.request.received"), 1)
        self.assertIn("1" * 32, process.stderr)
