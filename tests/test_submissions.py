import os
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from judge.agent_channel import AgentUnavailable
from judge.database import get_job_by_request_key
from judge.main import app
from judge.models import JobReceipt, JudgeResult, Resources
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
                "JUDGE_AGENT_TOKEN": "agent-secret",
                "JUDGE_DATABASE_PATH": str(self.database_path),
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

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

    def test_gpu_submission_fails_when_no_agent_is_connected(self) -> None:
        self.register_gpu_agent()

        response = self.client.post(
            "/submissions",
            headers=self.gpu_headers(),
            json=self.submission("gpu-example"),
        )

        self.assertEqual(response.status_code, 503)
        self.assertIsNone(
            get_job_by_request_key(
                self.database_path, self.submission()["repo_url"], "run-1"
            )
        )

    def test_gpu_submission_skips_agents_with_a_different_judge_revision(self) -> None:
        self.register_gpu_agent()
        app.state.judge_revision = "new-master-commit"
        with patch.object(
            app.state.agent_channel, "available_ids", new_callable=AsyncMock
        ) as available_ids:
            available_ids.return_value = {"nano4"}
            response = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )
        self.assertEqual(response.status_code, 503)

    def test_gpu_submission_succeeds_after_slurm_acknowledges_it(self) -> None:
        self.register_gpu_agent()
        with (
            patch.object(
                app.state.agent_channel, "available_ids", new_callable=AsyncMock
            ) as available_ids,
            patch.object(
                app.state.agent_channel, "offer", new_callable=AsyncMock
            ) as offer,
        ):
            available_ids.return_value = {"nano4"}
            offer.return_value = JobReceipt(accepted=True, slurm_job_id="12345")

            response = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(response.json()["assigned_judge_id"], "nano4")
        self.assertEqual(response.json()["slurm_job_id"], "12345")
        self.assertEqual(offer.call_args.args[1].job_id, response.json()["id"])

        retry = self.client.post(
            "/submissions",
            headers=self.gpu_headers(),
            json=self.submission("gpu-example"),
        )
        self.assertEqual(retry.status_code, 201)
        self.assertEqual(retry.json()["id"], response.json()["id"])

    def test_gpu_submission_rejects_an_expired_agent_even_if_poll_is_visible(
        self,
    ) -> None:
        self.register_gpu_agent()
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE sub_judges SET last_seen_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", "nano4"),
            )
        with patch.object(
            app.state.agent_channel, "available_ids", new_callable=AsyncMock
        ) as available_ids:
            available_ids.return_value = {"nano4"}
            response = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("No available sub-judge", response.json()["detail"])

    def test_gpu_submission_exchanges_a_real_offer_and_receipt(self) -> None:
        self.register_gpu_agent()

        def accept_offer() -> str:
            response = self.client.get(
                "/agents/nano4/next",
                headers={"Authorization": "Bearer agent-secret"},
            )
            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]
            receipt = self.client.post(
                f"/agents/nano4/jobs/{job_id}/receipt",
                headers={"Authorization": "Bearer agent-secret"},
                json={"accepted": True, "slurm_job_id": "12345"},
            )
            self.assertEqual(receipt.status_code, 204)
            return job_id

        with ThreadPoolExecutor(max_workers=1) as executor:
            agent = executor.submit(accept_offer)
            assert self.client.portal is not None
            deadline = time.monotonic() + 2
            while "nano4" not in self.client.portal.call(
                app.state.agent_channel.available_ids
            ):
                if time.monotonic() >= deadline:
                    self.fail("Sub-judge did not begin polling")
                time.sleep(0.01)

            response = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )

            self.assertEqual(response.status_code, 201)
            self.assertEqual(agent.result(timeout=5), response.json()["id"])
            self.assertEqual(response.json()["slurm_job_id"], "12345")

        job_id = response.json()["id"]
        agent_headers = {"Authorization": "Bearer agent-secret"}
        for sequence, payload in enumerate(
            (
                {"kind": "started"},
                {"kind": "log", "line": "[lab] loading model\n"},
                {"kind": "completed", "result": {"passed": True}},
            ),
            start=1,
        ):
            event = self.client.post(
                f"/agents/nano4/jobs/{job_id}/events",
                headers=agent_headers,
                json={"sequence": sequence, "slurm_job_id": "12345", **payload},
            )
            self.assertEqual(event.status_code, 200)
        finished = self.client.get(f"/jobs/{job_id}", headers=self.headers)
        self.assertEqual(finished.json()["status"], "completed")
        self.assertTrue(finished.json()["result"]["passed"])

        duplicate = self.client.post(
            f"/agents/nano4/jobs/{job_id}/events",
            headers=agent_headers,
            json={
                "sequence": 3,
                "slurm_job_id": "12345",
                "kind": "completed",
                "result": {"passed": True},
            },
        )
        self.assertEqual(duplicate.status_code, 200)
        wrong_agent = self.client.post(
            f"/agents/other-agent/jobs/{job_id}/events",
            headers=agent_headers,
            json={
                "sequence": 3,
                "slurm_job_id": "12345",
                "kind": "completed",
                "result": {"passed": True},
            },
        )
        self.assertEqual(wrong_agent.status_code, 409)

    def test_gpu_submission_fails_if_agent_disconnects_during_dispatch(self) -> None:
        self.register_gpu_agent()
        with (
            patch.object(
                app.state.agent_channel, "available_ids", new_callable=AsyncMock
            ) as available_ids,
            patch.object(
                app.state.agent_channel, "offer", new_callable=AsyncMock
            ) as offer,
        ):
            available_ids.return_value = {"nano4"}
            offer.side_effect = AgentUnavailable("not connected")

            response = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )

        self.assertEqual(response.status_code, 503)
        job = get_job_by_request_key(
            self.database_path, self.submission()["repo_url"], "run-1"
        )
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.status.value, "dispatching")

    def test_gpu_submission_retries_a_dispatch_without_creating_another_job(
        self,
    ) -> None:
        self.register_gpu_agent()
        with (
            patch.object(
                app.state.agent_channel, "available_ids", new_callable=AsyncMock
            ) as available_ids,
            patch.object(
                app.state.agent_channel, "offer", new_callable=AsyncMock
            ) as offer,
        ):
            available_ids.return_value = {"nano4"}
            offer.side_effect = [
                AgentUnavailable("not connected"),
                JobReceipt(accepted=True, slurm_job_id="12345"),
            ]

            first = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )
            second = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )

        self.assertEqual(first.status_code, 503)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(offer.call_count, 2)
        self.assertEqual(offer.call_args_list[0].args[1].job_id, second.json()["id"])
        self.assertEqual(offer.call_args_list[1].args[1].job_id, second.json()["id"])

    def test_gpu_submission_reports_slurm_rejection(self) -> None:
        self.register_gpu_agent()
        with (
            patch.object(
                app.state.agent_channel, "available_ids", new_callable=AsyncMock
            ) as available_ids,
            patch.object(
                app.state.agent_channel, "offer", new_callable=AsyncMock
            ) as offer,
        ):
            available_ids.return_value = {"nano4"}
            offer.return_value = JobReceipt(accepted=False, error="quota exceeded")

            response = self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("quota exceeded", response.json()["detail"])
        job = get_job_by_request_key(
            self.database_path, self.submission()["repo_url"], "run-1"
        )
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.status.value, "error")

    def test_rejects_reused_idempotency_key_with_a_different_commit(self) -> None:
        self.register_gpu_agent()
        with (
            patch.object(
                app.state.agent_channel, "available_ids", new_callable=AsyncMock
            ) as available_ids,
            patch.object(
                app.state.agent_channel, "offer", new_callable=AsyncMock
            ) as offer,
        ):
            available_ids.return_value = {"nano4"}
            offer.return_value = JobReceipt(accepted=True, slurm_job_id="12345")
            self.client.post(
                "/submissions",
                headers=self.gpu_headers(),
                json=self.submission("gpu-example"),
            )

        changed = self.submission("gpu-example") | {"commit_sha": "b" * 40}
        response = self.client.post(
            "/submissions", headers=self.gpu_headers(), json=changed
        )

        self.assertEqual(response.status_code, 409)

    def register_gpu_agent(self) -> None:
        response = self.client.post(
            "/agents/register",
            headers={"Authorization": "Bearer agent-secret"},
            json={
                "id": "nano4",
                "backend": "slurm",
                "task_ids": ["gpu-example"],
                "max_gpus": 8,
                "judge_revision": "test-revision",
            },
        )
        self.assertEqual(response.status_code, 200)

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
