import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judge.bootstrap_report import report_failure
from judge.models import Resources, Submission
from judge.ssh import RemoteConfig, RemoteRequest, TailscaleSSH


class BootstrapReportTests(unittest.TestCase):
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
        ).model_dump(mode="json")
        self.run_mock = Mock(url="https://wandb.ai/entity/project/runs/job")
        self.run_mock.summary = {}
        init = patch("judge.bootstrap_report.wandb.init", return_value=self.run_mock)
        self.init = init.start()
        self.addCleanup(init.stop)
        self.path = self.root / "job-records" / f"{'1' * 32}.json"

    def test_missing_key_retries_and_retains_original_failure(self):
        failed = report_failure(self.request, "Trusted setup failed")
        self.assertTrue(failed["report_pending"])
        original_time = json.loads(self.path.read_text())["finished_at"]
        (self.root / ".netrc").write_text("machine api.wandb.ai password test-key")
        success = report_failure(self.request, "a later setup failure")
        self.assertFalse(success["report_pending"])
        self.assertEqual(success["error"], "Trusted setup failed")
        self.assertEqual(
            json.loads(self.path.read_text())["finished_at"], original_time
        )
        self.assertEqual(self.run_mock.summary["error"], "Trusted setup failed")
        self.run_mock.finish.assert_called_once_with(exit_code=1)
        self.init.reset_mock()
        self.assertEqual(report_failure(self.request, "repeated"), success)
        self.init.assert_not_called()

    def test_bootstrap_failure_does_not_invalidate_an_accepted_slurm_job(self):
        self.path.parent.mkdir()
        self.path.write_text(
            json.dumps(
                {"request": self.request, "state": "submitted", "slurm_job_id": "123"}
            )
        )
        snapshot = report_failure(self.request, "trusted checkout became unavailable")
        self.assertNotIn("error", snapshot)
        self.assertEqual(json.loads(self.path.read_text())["slurm_job_id"], "123")
        self.init.assert_not_called()

    def test_transport_streams_packaged_reporter_without_credentials(self):
        transport = TailscaleSSH()
        process = Mock(
            returncode=0, stdout='{"error":"setup failed","report_pending":true}'
        )
        request = RemoteRequest.model_validate(self.request)
        with patch("judge.ssh.subprocess.run", return_value=process) as execute:
            snapshot = transport.report_bootstrap_failure(request, "git failed")
        self.assertTrue(snapshot.report_pending)
        self.assertNotIn("WANDB_API_KEY", execute.call_args.kwargs["input"])
        self.assertIn("--no-project", execute.call_args.args[0][-1])
