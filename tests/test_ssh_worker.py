import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from judge.database import (
    append_remote_event,
    create_remote_job,
    get_job,
    migrate_database,
    next_unreported_event,
    pending_ssh_jobs,
    record_ssh_snapshot,
)
from judge.models import (
    JudgeResult,
    RemoteEvent,
    RemoteEventKind,
    Resources,
    Submission,
)
from judge.ssh import RemoteConfig, RemoteSetupError, RemoteSnapshot, TailscaleSSH
from judge.ssh_worker import poll_jobs


class SSHWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "judge.db"
        migrate_database(self.path)
        self.config = RemoteConfig(
            host="nano4", user="judge", work_root="/work/oj spaces"
        )
        self.job = create_remote_job(
            self.path,
            Submission(
                repo_url="https://github.com/example/repo.git",
                commit_sha="a" * 40,
                task_id="lab2",
                github_actor="student",
            ),
            config=self.config,
            resources=Resources(gpus=1),
            request_key="run-1",
        )
        self.transport = Mock(spec=TailscaleSSH)

    def test_disconnect_and_restart_preserve_job_id(self):
        self.transport.poll.side_effect = ConnectionError("disconnected")
        poll_jobs(self.path, self.transport)
        job = get_job(self.path, self.job.id)
        assert job is not None
        self.assertEqual(job.status.value, "dispatching")
        self.transport.poll.side_effect = None
        self.transport.poll.return_value = RemoteSnapshot(
            slurm_job_id="123", slurm_state="PENDING"
        )
        poll_jobs(self.path, self.transport)
        job = get_job(self.path, self.job.id)
        assert job is not None
        self.assertEqual(job.status.value, "queued")
        self.assertEqual(job.slurm_job_id, "123")
        self.assertEqual(self.transport.poll.call_args.args[0].job_id, self.job.id)

    def test_remote_reporting_retry_survives_finished_job_and_worker_restart(self):
        request = pending_ssh_jobs(self.path)[0]
        job = record_ssh_snapshot(
            self.path,
            request,
            RemoteSnapshot(
                slurm_job_id="123", error="Slurm failed", report_pending=True
            ),
        )
        self.assertEqual(job.status.value, "error")
        self.assertEqual(len(pending_ssh_jobs(self.path)), 1)
        url = "https://wandb.ai/entity/project/runs/job"
        job = record_ssh_snapshot(
            self.path,
            pending_ssh_jobs(self.path)[0],
            RemoteSnapshot(slurm_job_id="123", error="Slurm failed", wandb_url=url),
        )
        self.assertEqual(job.wandb_url, url)
        self.assertEqual(job.wandb_run_id, self.job.id)
        self.assertFalse(pending_ssh_jobs(self.path))

    def test_running_result_waits_for_remaining_logs(self):
        request = pending_ssh_jobs(self.path)[0]
        snapshot = RemoteSnapshot(
            slurm_job_id="123",
            slurm_state="COMPLETED",
            result=JudgeResult(passed=True),
            slurm_log="part one",
            log_offset=8,
            logs_remaining=True,
        )
        job = record_ssh_snapshot(self.path, request, snapshot)
        self.assertEqual(job.status.value, "running")
        recovered = pending_ssh_jobs(self.path)[0]
        self.assertEqual(recovered.log_offset, 8)
        finished = record_ssh_snapshot(
            self.path,
            recovered,
            RemoteSnapshot(
                slurm_job_id="123",
                result=JudgeResult(passed=True),
                slurm_log="part two",
                log_offset=16,
            ),
        )
        self.assertEqual(finished.status.value, "completed")
        self.assertFalse(pending_ssh_jobs(self.path))
        with self.assertRaisesRegex(ValueError, "Stale"):
            record_ssh_snapshot(self.path, request, snapshot)

    def test_bootstrap_failure_is_reportable_without_slurm_id(self):
        self.transport.poll.side_effect = RemoteSetupError("Git checkout failed")
        poll_jobs(self.path, self.transport)
        job = get_job(self.path, self.job.id)
        assert job is not None
        self.assertEqual(job.status.value, "error")
        self.assertIsNone(job.slurm_job_id)
        self.assertIn("Git checkout failed", job.error or "")
        self.assertIsNotNone(next_unreported_event(self.path))

    def test_offsets_roll_back_when_a_snapshot_changes_slurm_id(self):
        request = pending_ssh_jobs(self.path)[0]
        record_ssh_snapshot(self.path, request, RemoteSnapshot(slurm_job_id="123"))
        with self.assertRaisesRegex(ValueError, "ID changed"):
            record_ssh_snapshot(
                self.path,
                request,
                RemoteSnapshot(slurm_job_id="456", slurm_log="bad", log_offset=3),
            )
        self.assertEqual(pending_ssh_jobs(self.path)[0].log_offset, 0)

    def test_preparation_log_and_failure_events_are_idempotent(self):
        event = RemoteEvent(sequence=1, kind=RemoteEventKind.LOG, line="uv syncing\n")
        first = append_remote_event(self.path, self.job.id, event)
        self.assertEqual(first, append_remote_event(self.path, self.job.id, event))
        failed = append_remote_event(
            self.path,
            self.job.id,
            RemoteEvent(sequence=2, kind=RemoteEventKind.FAILED, error="uv failed"),
        )
        self.assertEqual(failed.status.value, "error")

    def test_transport_quotes_remote_paths_and_never_sends_credentials(self):
        transport = TailscaleSSH()
        request = pending_ssh_jobs(self.path)[0]
        processes = [
            Mock(returncode=0, stdout="ready", stderr=""),
            Mock(returncode=0, stdout='{"slurm_job_id":"123"}', stderr=""),
        ]
        with patch("judge.ssh.subprocess.run", side_effect=processes) as run:
            self.assertEqual(transport.poll(request).slurm_job_id, "123")
        bootstrap, polling = run.call_args_list
        self.assertEqual(
            bootstrap.args[0][:4],
            [
                "tailscale",
                "--socket=/var/run/tailscale/tailscaled.sock",
                "ssh",
                "judge@nano4",
            ],
        )
        script = shlex.split(bootstrap.args[0][-1])[-1]
        self.assertIn(f"latest {shlex.quote(request.trusted_root)}", script)
        self.assertIn(
            f"PYTHONPATH={request.trusted_root}/src",
            shlex.split(polling.args[0][-1]),
        )
        for secret in ("WANDB_API_KEY", "TS_AUTHKEY", "JUDGE_API_TOKEN"):
            self.assertNotIn(secret, polling.kwargs["input"])

    def test_successful_ssh_forwards_remote_diagnostics_without_logging_payload(self):
        request = pending_ssh_jobs(self.path)[0]
        processes = [
            subprocess.CompletedProcess("ssh", 0, "ready", ""),
            subprocess.CompletedProcess(
                "ssh",
                0,
                '{"slurm_job_id":"123","slurm_state":"PENDING"}',
                "setup "
                + "x" * 9000
                + "\nslurm.command.response stdout=123|PENDING|0:0\n",
            ),
        ]
        with (
            patch("judge.ssh.subprocess.run", side_effect=processes),
            self.assertLogs("judge.execution", level="INFO") as logs,
        ):
            snapshot = TailscaleSSH().poll(request)
        self.assertEqual(snapshot.slurm_job_id, "123")
        text = "\n".join(logs.output)
        self.assertIn("ssh.started", text)
        self.assertIn("ssh.finished", text)
        self.assertIn("slurm.command.response", text)
        self.assertIn("123|PENDING|0:0", text)
        self.assertIn('host="nano4"', text)
        self.assertIn(request.job_id, text)
        self.assertNotIn(request.model_dump_json(), text)
        self.assertNotIn("PYTHONPATH=", text)

    def test_worker_retry_redacts_exception_credentials(self):
        self.transport.poll.side_effect = ConnectionError("lost hf_hidden123")
        with self.assertLogs("judge.execution", level="INFO") as logs:
            poll_jobs(self.path, self.transport)
        text = "\n".join(logs.output)
        self.assertIn("worker.poll.retry", text)
        self.assertIn(self.job.id, text)
        self.assertNotIn("hf_hidden123", text)
        self.assertEqual(len(pending_ssh_jobs(self.path)), 1)

    def test_preparation_is_per_job_instead_of_cached_for_the_whole_worker(self):
        transport = TailscaleSSH()
        request = pending_ssh_jobs(self.path)[0]
        following = request.model_copy(update={"job_id": "2" * 32})
        with patch(
            "judge.ssh.subprocess.run", return_value=Mock(returncode=0, stdout="ready")
        ) as execute:
            transport.prepare(request)
            transport.prepare(request)
            transport.prepare(following)
        self.assertEqual(execute.call_count, 2)
        self.assertNotEqual(request.trusted_root, following.trusted_root)

    def test_retired_jobs_return_durable_results_without_recreating_checkout(self):
        request = pending_ssh_jobs(self.path)[0]
        root = self.path.parent / "retired workspace with spaces"
        request = request.model_copy(
            update={"config": self.config.model_copy(update={"work_root": str(root)})}
        )
        cached = {
            "request": request.model_dump(mode="json"),
            "report": {"complete": True},
            "terminal": {"slurm_job_id": "123", "result": {"passed": True}},
        }
        record = root / "job-records" / f"{request.job_id}.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps(cached))

        def local_run(config, command, payload, timeout):
            # Run only the generated local shell wrapper, never Tailscale/SSH.
            # If its cache check fails, abort instead of running repository setup.
            return subprocess.run(
                shlex.split(command),
                input="exit 42",
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )

        with patch(
            "judge.ssh.TailscaleSSH._run",
            side_effect=local_run,
        ) as execute:
            result = TailscaleSSH().poll(request)
        self.assertTrue(result.result and result.result.passed)
        execute.assert_called_once()
        self.assertFalse((root / "jobs").exists())

    def test_old_persisted_revision_is_ignored(self):
        config = RemoteConfig.model_validate(
            self.config.model_dump() | {"revision": "b" * 40}
        )
        self.assertEqual(config, self.config)
        self.assertNotIn("revision", config.model_dump())

    def test_transport_retries_ssh_failure_and_classifies_setup_failure(self):
        transport = TailscaleSSH()
        with (
            patch(
                "judge.ssh.subprocess.run",
                return_value=Mock(returncode=255, stdout="", stderr="disconnected"),
            ),
            self.assertRaises(ConnectionError),
        ):
            transport.prepare(pending_ssh_jobs(self.path)[0])
        with (
            patch(
                "judge.ssh.subprocess.run",
                return_value=Mock(
                    returncode=2, stdout="preparing", stderr="bad checkout"
                ),
            ),
            self.assertRaises(RemoteSetupError),
        ):
            transport.prepare(pending_ssh_jobs(self.path)[0])
        with (
            patch(
                "judge.ssh.subprocess.run",
                side_effect=subprocess.TimeoutExpired("tailscale", 30),
            ),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            transport.prepare(pending_ssh_jobs(self.path)[0])

    def test_rejects_invalid_configuration_and_execution_events(self):
        for changes in (
            {"user": "user;touch /tmp/x"},
            {"host": "-oProxyCommand=x"},
            {"work_root": "/"},
            {"work_root": "relative"},
        ):
            with self.assertRaises(ValidationError):
                RemoteConfig.model_validate(self.config.model_dump() | changes)
        with self.assertRaises(ValidationError):
            RemoteEvent(sequence=1, kind=RemoteEventKind.STARTED)
