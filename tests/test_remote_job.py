import json
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from judge.models import Resources, Submission
from judge.remote_job import handle, read_log
from judge.slurm_executor import SlurmState
from judge.ssh import RemoteConfig, RemoteRequest


class RemoteJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = RemoteRequest(
            job_id="1" * 32,
            submission=Submission(
                repo_url="https://github.com/example/repo.git",
                commit_sha="a" * 40,
                task_id="lab2",
                github_actor="student",
            ),
            resources=Resources(gpus=1),
            config=RemoteConfig(host="nano4", user="judge", work_root=str(self.root)),
        )
        self.workspace = self.root / "jobs" / self.request.job_id
        self.output = self.workspace / "output"
        setup = patch(
            "judge.remote_job.subprocess.run", side_effect=self.setup_repository
        )
        self.setup_process = setup.start()
        self.addCleanup(setup.stop)
        submit = patch("judge.remote_job.SlurmExecutor.submit", return_value="12345")
        self.submit = submit.start()
        self.addCleanup(submit.stop)
        status = patch(
            "judge.remote_job.SlurmExecutor.status", return_value=SlurmState("PENDING")
        )
        self.status = status.start()
        self.addCleanup(status.stop)
        reporting = patch("judge.remote_job.publish_report", side_effect=self.publish)
        self.reporting = reporting.start()
        self.addCleanup(reporting.stop)

    def publish(self, request, snapshot, workspace, record):
        snapshot.wandb_url = "https://wandb.ai/entity/project/runs/job"
        record["report"] = {
            "url": snapshot.wandb_url,
            "complete": snapshot.error is not None or snapshot.result is not None,
        }

    def setup_repository(self, command, **kwargs):
        script = Path(command[-1]) / "src/labs/lab2.sbatch"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/bash\n")
        kwargs["stdout"].write("repository ready\n")
        return Mock(returncode=0)

    def test_repeated_polls_reuse_slurm_id_without_setup_or_submission(self):
        first = handle(self.request)
        again = handle(
            self.request.model_copy(update={"setup_offset": first.setup_offset})
        )
        self.assertEqual(first.slurm_job_id, "12345")
        self.assertEqual(again.slurm_job_id, "12345")
        self.assertEqual(again.setup_log, "")
        self.setup_process.assert_called_once()
        self.submit.assert_called_once()
        self.assertEqual(
            self.submit.call_args.kwargs["job_name"], f"judge-{self.request.job_id}"
        )

    def test_concurrent_requests_submit_once(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(handle, [self.request, self.request]))
        self.assertEqual(
            [result.slurm_job_id for result in results], ["12345", "12345"]
        )
        self.submit.assert_called_once()

    def test_rejects_changed_job_identity(self):
        handle(self.request)
        changed = self.request.model_copy(update={"resources": Resources(gpus=2)})
        with self.assertRaisesRegex(ValueError, "identity"):
            handle(changed)
        self.submit.assert_called_once()

    def test_old_record_revision_is_ignored_without_resubmitting(self):
        handle(self.request)
        path = self.root / "job-records" / f"{self.request.job_id}.json"
        record = json.loads(path.read_text())
        record["request"]["config"]["revision"] = "b" * 40
        path.write_text(json.dumps(record))
        self.assertEqual(handle(self.request).slurm_job_id, "12345")
        self.submit.assert_called_once()
        self.setup_process.assert_called_once()

    def test_setup_failure_is_durable_and_never_submits(self):
        self.setup_process.side_effect = subprocess.CalledProcessError(1, "uv sync")
        self.assertIn("CalledProcessError", handle(self.request).error or "")
        self.assertIn("CalledProcessError", handle(self.request).error or "")
        self.setup_process.assert_called_once()
        self.submit.assert_not_called()

    def test_slow_setup_keeps_reporting_before_sbatch(self):
        waiting = Event()
        resumed = Event()

        def slow_setup(command, **kwargs):
            waiting.set()
            if not resumed.wait(timeout=5):
                raise RuntimeError("heartbeat never arrived")
            return self.setup_repository(command, **kwargs)

        def heartbeat(request, snapshot, workspace, record):
            self.publish(request, snapshot, workspace, record)
            if waiting.is_set() and not resumed.is_set():
                self.submit.assert_not_called()
                self.assertEqual(record["state"], "preparing")
                resumed.set()

        self.setup_process.side_effect = slow_setup
        self.reporting.side_effect = heartbeat
        with patch("judge.remote_job.REPORT_INTERVAL", 0.01):
            self.assertEqual(handle(self.request).slurm_job_id, "12345")
        self.assertTrue(resumed.is_set())
        self.assertGreaterEqual(self.reporting.call_count, 3)
        self.submit.assert_called_once()

    def test_setup_finishing_at_heartbeat_deadline_is_not_a_failure(self):
        future = Mock()
        future.result.side_effect = [TimeoutError, None]
        future.done.return_value = True

        def already_finished(function, *args, **kwargs):
            function(*args, **kwargs)
            return future

        with patch("judge.remote_job.ThreadPoolExecutor") as preparation:
            pool = preparation.return_value.__enter__.return_value
            pool.submit.side_effect = already_finished
            snapshot = handle(self.request)
        self.assertIsNone(snapshot.error)
        self.assertEqual(snapshot.slurm_job_id, "12345")
        self.submit.assert_called_once()

    def test_missing_script_and_invalid_resources_fail_before_sbatch(self):
        self.setup_process.side_effect = lambda *args, **kwargs: Mock(returncode=0)
        self.assertIn("FileNotFoundError", handle(self.request).error or "")
        self.submit.assert_not_called()
        changed = self.request.model_copy(
            update={"job_id": "2" * 32, "resources": Resources(gpus=9)}
        )
        self.setup_process.side_effect = self.setup_repository
        self.assertIn("1 to 8 GPUs", handle(changed).error or "")
        self.submit.assert_not_called()

    def test_sbatch_rejection_is_a_definite_failure(self):
        self.submit.side_effect = subprocess.CalledProcessError(
            1, "sbatch", stderr="quota exceeded"
        )
        self.assertEqual(handle(self.request).error, "quota exceeded")
        self.assertEqual(handle(self.request).error, "quota exceeded")
        self.submit.assert_called_once()

    def test_timeout_after_submission_never_retries_sbatch(self):
        self.submit.side_effect = subprocess.TimeoutExpired("sbatch", 30)
        with self.assertRaises(subprocess.TimeoutExpired):
            handle(self.request)
        snapshot = handle(self.request)
        self.assertIn("ambiguous", snapshot.error or "")
        self.assertFalse((self.workspace / ".finished").exists())
        self.submit.assert_called_once()
        record = json.loads((self.workspace / "submission.json").read_text())
        self.assertEqual(record["state"], "submitting")

    def test_unrecognized_sbatch_response_is_ambiguous(self):
        self.submit.side_effect = RuntimeError("Unrecognized sbatch response")
        with self.assertRaises(RuntimeError):
            handle(self.request)
        self.assertIn("ambiguous", handle(self.request).error or "")
        self.submit.assert_called_once()

    def test_valid_result_and_complete_log_drain(self):
        handle(self.request)
        (self.output / "result.json").write_text('{"passed": true}')
        (self.output / "slurm.log").write_text("x" * 9000)
        self.status.return_value = SlurmState("COMPLETED", "0:0")
        first = handle(self.request)
        self.assertTrue(first.result and first.result.passed)
        self.assertFalse(first.logs_remaining)
        self.assertEqual(first.slurm_log, "")
        again = handle(
            self.request.model_copy(
                update={
                    "log_offset": first.log_offset,
                    "setup_offset": first.setup_offset,
                }
            )
        )
        self.assertFalse(again.logs_remaining)
        self.assertEqual((self.output / "slurm.log").read_text(), "x" * 9000)

    def test_terminal_failure_and_missing_or_malformed_result(self):
        handle(self.request)
        self.status.return_value = SlurmState("OUT_OF_MEMORY", "1:0")
        self.assertIn("OUT_OF_MEMORY", handle(self.request).error or "")
        changed = self.request.model_copy(update={"job_id": "2" * 32})
        self.status.return_value = SlurmState("COMPLETED", "0:0")
        self.assertIn("Missing or invalid", handle(changed).error or "")
        malformed = self.request.model_copy(update={"job_id": "3" * 32})
        output = self.root / "jobs" / malformed.job_id / "output"
        output.mkdir(parents=True)
        (output / "result.json").write_text("bad json")
        self.assertIn("Missing or invalid", handle(malformed).error or "")

    def test_report_failure_retries_after_terminal_execution_without_resubmitting(self):
        self.reporting.side_effect = RuntimeError("W&B unavailable")
        handle(self.request)
        self.status.return_value = SlurmState("FAILED", "1:0")
        first = handle(self.request)
        self.assertTrue(first.report_pending)
        self.assertFalse((self.workspace / ".finished").exists())
        self.reporting.side_effect = self.publish
        again = handle(self.request)
        self.assertFalse(again.report_pending)
        self.assertTrue((self.workspace / ".finished").exists())
        self.submit.assert_called_once()

    def test_retired_workspace_keeps_terminal_record_and_never_resubmits(self):
        import shutil

        handle(self.request)
        self.status.return_value = SlurmState("FAILED", "1:0")
        handle(self.request)
        shutil.rmtree(self.workspace)
        again = handle(self.request)
        self.assertIn("FAILED", again.error or "")
        self.assertFalse(self.workspace.exists())
        self.submit.assert_called_once()

    def test_reporter_console_output_never_corrupts_ssh_json(self):
        import sys

        script = """
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from judge.remote_job import report
from judge.ssh import RemoteRequest, RemoteSnapshot

request = RemoteRequest.model_validate_json(sys.argv[1])
workspace = Path(request.config.work_root) / 'jobs' / request.job_id
(workspace / 'output').mkdir(parents=True)
snapshot = RemoteSnapshot(slurm_job_id='12345')
def publish(*args):
    print('W&B console output')
    os.write(1, b'cached SDK output\\n')
with patch('judge.remote_job.publish_report', side_effect=publish):
    report(request, snapshot, workspace, {'state': 'preparing'})
print(snapshot.model_dump_json())
"""
        with subprocess.Popen(
            [sys.executable, "-c", script, self.request.model_dump_json()],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(json.loads(stdout)["slurm_job_id"], "12345")
        self.assertEqual(stderr, "")
        self.assertIn("W&B console output", (self.output / "reporting.log").read_text())

    def test_accounting_delay_is_retryable(self):
        handle(self.request)
        self.status.return_value = None
        self.assertIsNone(handle(self.request).error)

    def test_log_offsets_count_bytes(self):
        path = self.root / "log"
        path.write_text("模型完成\n")
        text, offset, remaining = read_log(path, 0)
        self.assertEqual(text, "模型完成\n")
        self.assertEqual(offset, len(path.read_bytes()))
        self.assertFalse(remaining)
