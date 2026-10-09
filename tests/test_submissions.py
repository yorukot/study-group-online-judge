import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from judge.main import app
from judge.models import JudgeResult, Resources
from judge.tasks import TASKS
from judge.tasks.base import Task


class ExampleTask(Task):
    id = "example"

    def evaluate(self, submission: Path) -> JudgeResult:
        return JudgeResult(passed=submission.is_dir())


class GpuTask(Task):
    id = "gpu-example"
    resources = Resources(gpus=1)

    def evaluate(self, submission: Path) -> JudgeResult:
        return JudgeResult(passed=submission.is_dir())


class SubmissionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_tasks = TASKS.copy()
        TASKS.clear()
        TASKS[ExampleTask.id] = ExampleTask()
        TASKS[GpuTask.id] = GpuTask()
        self.addCleanup(self.restore_tasks)

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "judge.db"

        self.environment = patch.dict(
            os.environ,
            {
                "JUDGE_API_TOKEN": "secret",
                "JUDGE_SSH_HOST": "nano4",
                "JUDGE_SSH_USER": "judge",
                "JUDGE_REMOTE_WORK_ROOT": "/work/oj",
                "JUDGE_DATABASE_PATH": str(self.database_path),
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

        refresh = patch("judge.main.LeaderboardService.refresh")
        refresh.start()
        self.addCleanup(refresh.stop)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)
        self.headers = {"Authorization": "Bearer secret"}

    def restore_tasks(self) -> None:
        TASKS.clear()
        TASKS.update(self.original_tasks)

    def submission(self, task_id: str = "example") -> dict[str, str]:
        return {
            "repo_url": "https://github.com/cerulean-works/example.git",
            "commit_sha": "a" * 40,
            "task_id": task_id,
            "github_actor": "student",
        }

    def test_creates_and_reads_a_submission_job(self) -> None:
        response = self.client.post(
            "/submissions",
            headers=self.headers,
            json=self.submission(),
        )

        self.assertEqual(response.status_code, 201)
        created = response.json()
        self.assertEqual(created["status"], "queued")

        response = self.client.get(
            f"/jobs/{created['id']}",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), created)

    def test_requires_the_api_token(self) -> None:
        response = self.client.post("/submissions", json=self.submission())

        self.assertEqual(response.status_code, 401)

    def test_rejects_an_unknown_task(self) -> None:
        response = self.client.post(
            "/submissions",
            headers=self.headers,
            json=self.submission(task_id="missing"),
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Unknown task: missing")

    def test_returns_not_found_for_an_unknown_job(self) -> None:
        response = self.client.get("/jobs/missing", headers=self.headers)

        self.assertEqual(response.status_code, 404)

    def test_gpu_submission_requires_an_idempotency_key(self) -> None:
        response = self.client.post(
            "/submissions", headers=self.headers, json=self.submission("gpu-example")
        )

        self.assertEqual(response.status_code, 422)

    def test_gpu_submission_queues_without_connecting_remotely(self):
        with (
            patch("judge.ssh.TailscaleSSH.poll") as poll,
            self.assertLogs("judge.execution", level="INFO") as logs,
        ):
            response = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )
            retry = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(response.json()["execution_backend"], "ssh_slurm")
        self.assertIsNone(response.json()["slurm_job_id"])
        self.assertEqual(response.json()["id"], retry.json()["id"])
        poll.assert_not_called()
        text = "\n".join(logs.output)
        self.assertIn("submission.received", text)
        self.assertIn(response.json()["id"], text)
        self.assertIn('host="nano4"', text)
        self.assertNotIn("Bearer secret", text)

    def test_gpu_submission_requires_remote_configuration(self):
        app.state.remote_config = None
        response = self.client.post(
            "/submissions",
            headers=self.gpu_headers(),
            json=self.submission("gpu-example"),
        )
        self.assertEqual(response.status_code, 503)

    def test_rejects_reused_key_for_different_submission(self):
        self.client.post(
            "/submissions",
            headers=self.gpu_headers(),
            json=self.submission("gpu-example"),
        )
        changed = self.submission("gpu-example") | {"commit_sha": "c" * 40}
        response = self.client.post(
            "/submissions", headers=self.gpu_headers(), json=changed
        )
        self.assertEqual(response.status_code, 409)

    def test_agents_are_no_longer_exposed(self):
        self.assertNotIn(
            "/agents/register", self.client.get("/openapi.json").json()["paths"]
        )

    def gpu_headers(self) -> dict[str, str]:
        return self.headers | {"Idempotency-Key": "run-1"}


class StartupTests(unittest.TestCase):
    def test_requires_an_api_token_at_startup(self) -> None:
        with (
            patch.dict(os.environ, {"JUDGE_API_TOKEN": ""}),
            self.assertRaisesRegex(RuntimeError, "JUDGE_API_TOKEN"),
            TestClient(app),
        ):
            pass


if __name__ == "__main__":
    unittest.main()
