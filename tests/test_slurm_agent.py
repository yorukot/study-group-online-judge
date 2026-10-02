import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judge.models import (
    JobOffer,
    JudgeResult,
    RemoteEventKind,
    Resources,
    Submission,
)
from judge.slurm_agent import AgentLedger, MasterClient, SlurmAgent
from judge.slurm_executor import SlurmState


class SlurmAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger = AgentLedger(self.root / "agent.db")
        self.client = Mock()
        self.executor = Mock()
        self.executor.submit.return_value = "54321"
        self.agent = SlurmAgent(
            client=self.client,
            ledger=self.ledger,
            executor=self.executor,
            work_root=self.root / "work",
            task_ids=["lab2"],
            max_gpus=8,
            revision="test",
            poll_seconds=0,
        )
        self.offer = JobOffer(
            job_id="a" * 32,
            submission=Submission(
                repo_url="https://github.com/example/repo.git",
                commit_sha="b" * 40,
                task_id="lab2",
                github_actor="participant",
            ),
            resources=Resources(gpus=1),
        )

    @staticmethod
    def fake_checkout(repo_url: str, commit_sha: str, destination: Path) -> Path:
        destination.mkdir(parents=True)
        return destination

    def submit(self) -> None:
        with (
            patch("judge.slurm_agent.checkout_repository", self.fake_checkout),
            patch.object(self.agent, "_start_monitor"),
        ):
            receipt = self.agent.accept(self.offer)
        self.assertTrue(receipt.accepted)
        self.assertEqual(receipt.slurm_job_id, "54321")

    def test_repeated_offer_reuses_persisted_slurm_job(self) -> None:
        self.submit()
        with patch.object(self.agent, "_start_monitor"):
            receipt = self.agent.accept(self.offer)
        self.assertEqual(receipt.slurm_job_id, "54321")
        self.executor.submit.assert_called_once()
        restarted_ledger = AgentLedger(self.root / "agent.db")
        self.assertEqual(restarted_ledger.submitted_job_ids(), [self.offer.job_id])

    def test_master_client_sends_heartbeat(self) -> None:
        client = MasterClient("http://127.0.0.1:7000", "secret", "nano4")
        with patch.object(client, "request") as request:
            client.heartbeat()
        request.assert_called_once_with("POST", "/agents/nano4/heartbeat")

    def test_failed_heartbeat_triggers_reregistration(self) -> None:
        self.agent._registered.set()
        self.client.heartbeat.side_effect = RuntimeError("offline")
        with (
            patch("judge.slurm_agent.time.sleep", side_effect=[None, SystemExit]),
            self.assertRaises(SystemExit),
        ):
            self.agent._heartbeat_loop()
        self.client.heartbeat.assert_called_once()
        self.assertFalse(self.agent._registered.is_set())

    def test_ambiguous_submission_is_not_repeated(self) -> None:
        self.ledger.claim(self.offer)
        self.ledger.set_submitting(self.offer.job_id)
        receipt = self.agent.accept(self.offer)
        self.assertFalse(receipt.accepted)
        assert receipt.error is not None
        self.assertIn("manual reconciliation", receipt.error)
        self.executor.submit.assert_not_called()

    def test_monitor_reports_success_and_replays_pending_events(self) -> None:
        self.submit()
        output = self.root / "work" / self.offer.job_id / "output"
        output.mkdir()
        (output / "slurm.log").write_text("loading model\nstep 1/2\rstep 2/2\r")
        (output / "result.json").write_text(JudgeResult(passed=True).model_dump_json())
        self.executor.status.return_value = SlurmState("COMPLETED", "0:0")

        self.agent.monitor(self.offer.job_id)

        events = [call.args[1] for call in self.client.event.call_args_list]
        self.assertEqual(
            [event.kind for event in events],
            [
                RemoteEventKind.STARTED,
                RemoteEventKind.LOG,
                RemoteEventKind.COMPLETED,
            ],
        )
        self.assertTrue(events[-1].result.passed)
        self.assertEqual(self.ledger.state(self.offer.job_id)["state"], "finished")
        self.assertEqual(self.ledger.pending_events(self.offer.job_id), [])

    def test_monitor_reports_missing_result_as_failure(self) -> None:
        self.submit()
        self.executor.status.return_value = SlurmState("COMPLETED", "0:0")
        self.agent.monitor(self.offer.job_id)
        events = [call.args[1] for call in self.client.event.call_args_list]
        self.assertEqual(events[-1].kind, RemoteEventKind.FAILED)
        self.assertIn("Missing or invalid judge result", events[-1].error)

    def test_pending_events_survive_a_network_failure(self) -> None:
        self.submit()
        self.ledger.append_event(self.offer.job_id, RemoteEventKind.STARTED)
        self.client.event.side_effect = RuntimeError("offline")
        with self.assertRaisesRegex(RuntimeError, "offline"):
            self.agent._deliver_pending(self.offer.job_id)
        self.client.event.side_effect = None
        self.agent._deliver_pending(self.offer.job_id)
        self.assertEqual(self.ledger.pending_events(self.offer.job_id), [])


if __name__ == "__main__":
    unittest.main()
