import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from judge.main import app
from judge.models import JobOffer, Resources, Submission


class AgentRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        database_path = Path(self.temporary_directory.name) / "judge.db"

        self.environment = patch.dict(
            os.environ,
            {
                "JUDGE_API_TOKEN": "participant-secret",
                "JUDGE_AGENT_TOKEN": "agent-secret",
                "JUDGE_DATABASE_PATH": str(database_path),
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)
        self.headers = {"Authorization": "Bearer agent-secret"}

    @staticmethod
    def registration(task_ids: list[str] | None = None) -> dict[str, object]:
        return {
            "id": "nano4",
            "backend": "slurm",
            "task_ids": ["lab1"] if task_ids is None else task_ids,
            "max_gpus": 8,
            "judge_revision": "test-revision",
        }

    def test_registers_an_agent(self) -> None:
        response = self.client.post(
            "/agents/register", headers=self.headers, json=self.registration()
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "nano4")
        self.assertEqual(response.json()["backend"], "slurm")
        self.assertIn("registered_at", response.json())

    def test_agent_routes_need_the_separate_agent_token(self) -> None:
        for headers in (
            {},
            {"Authorization": "Bearer participant-secret"},
        ):
            with self.subTest(headers=headers):
                response = self.client.post(
                    "/agents/register", headers=headers, json=self.registration()
                )
                self.assertEqual(response.status_code, 401)

    def test_rejects_unknown_task_capabilities(self) -> None:
        response = self.client.post(
            "/agents/register",
            headers=self.headers,
            json=self.registration(["missing"]),
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Unknown tasks: missing")

    def test_requires_registration_before_polling(self) -> None:
        response = self.client.get("/agents/nano4/next", headers=self.headers)

        self.assertEqual(response.status_code, 404)

    def test_heartbeat_requires_a_live_registration_and_agent_token(self) -> None:
        route = "/agents/nano4/heartbeat"
        self.assertEqual(self.client.post(route, headers=self.headers).status_code, 404)
        self.client.post(
            "/agents/register", headers=self.headers, json=self.registration()
        )
        self.assertEqual(self.client.post(route).status_code, 401)
        self.assertEqual(self.client.post(route, headers=self.headers).status_code, 204)

    def test_poll_returns_an_offer_or_no_content(self) -> None:
        self.client.post(
            "/agents/register", headers=self.headers, json=self.registration()
        )
        offer = JobOffer(
            job_id="a" * 32,
            submission=Submission(
                repo_url="https://github.com/cerulean-works/example.git",
                commit_sha="b" * 40,
                task_id="lab1",
                github_actor="participant",
            ),
            resources=Resources(gpus=1),
        )
        with patch.object(
            app.state.agent_channel, "wait_for_offer", new_callable=AsyncMock
        ) as wait_for_offer:
            wait_for_offer.return_value = offer
            response = self.client.get("/agents/nano4/next", headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["job_id"], offer.job_id)

            wait_for_offer.return_value = None
            response = self.client.get("/agents/nano4/next", headers=self.headers)
            self.assertEqual(response.status_code, 204)

    def test_rejects_a_receipt_without_a_pending_offer(self) -> None:
        response = self.client.post(
            "/agents/nano4/jobs/unknown/receipt",
            headers=self.headers,
            json={"accepted": True},
        )

        self.assertEqual(response.status_code, 404)

    def test_disables_agent_routes_without_a_configured_token(self) -> None:
        app.state.agent_token = None

        response = self.client.post(
            "/agents/register", headers=self.headers, json=self.registration()
        )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
