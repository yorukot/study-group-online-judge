import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from judge.database import (
    append_remote_event,
    create_remote_job,
    get_job,
    mark_remote_event_reported,
    migrate_database,
    next_unreported_event,
    register_sub_judge,
)
from judge.models import (
    JobStatus,
    JudgeResult,
    RemoteEvent,
    RemoteEventKind,
    SubJudge,
    SubJudgeBackend,
    Submission,
)
from judge.remote_reporter import publish_event, run_reporter


class RemoteReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "judge.db"
        migrate_database(self.path)
        register_sub_judge(
            self.path,
            SubJudge(
                id="nano4",
                backend=SubJudgeBackend.SLURM,
                task_ids=["lab1"],
                max_gpus=8,
                judge_revision="test",
                registered_at=datetime.now(UTC),
            ),
        )
        self.job = create_remote_job(
            self.path,
            Submission(
                repo_url="https://github.com/example/repo.git",
                commit_sha="a" * 40,
                task_id="lab1",
                github_actor="participant",
            ),
            judge_id="nano4",
            request_key="request-1",
        )

    def event(
        self,
        sequence: int,
        kind: RemoteEventKind,
        *,
        line: str | None = None,
        result: JudgeResult | None = None,
        error: str | None = None,
    ) -> RemoteEvent:
        return RemoteEvent(
            sequence=sequence,
            kind=kind,
            slurm_job_id="12345",
            line=line,
            result=result,
            error=error,
        )

    def test_ordered_events_are_idempotent_and_change_job_state(self) -> None:
        started = self.event(1, RemoteEventKind.STARTED)
        running = append_remote_event(self.path, "nano4", self.job.id, started)
        self.assertEqual(running.status, JobStatus.RUNNING)
        self.assertEqual(running.slurm_job_id, "12345")
        self.assertEqual(
            append_remote_event(self.path, "nano4", self.job.id, started), running
        )
        with self.assertRaisesRegex(ValueError, "Expected event sequence 2"):
            append_remote_event(
                self.path,
                "nano4",
                self.job.id,
                self.event(3, RemoteEventKind.LOG, line="skipped"),
            )
        with self.assertRaisesRegex(ValueError, "different data"):
            append_remote_event(
                self.path,
                "nano4",
                self.job.id,
                self.event(1, RemoteEventKind.LOG, line="changed"),
            )
        append_remote_event(
            self.path,
            "nano4",
            self.job.id,
            self.event(2, RemoteEventKind.LOG, line="training\n"),
        )
        finished = append_remote_event(
            self.path,
            "nano4",
            self.job.id,
            self.event(3, RemoteEventKind.COMPLETED, result=JudgeResult(passed=True)),
        )
        self.assertEqual(finished.status, JobStatus.COMPLETED)
        self.assertTrue(finished.result and finished.result.passed)
        with self.assertRaisesRegex(ValueError, "not running"):
            append_remote_event(
                self.path,
                "nano4",
                self.job.id,
                self.event(4, RemoteEventKind.LOG, line="late"),
            )

    def test_report_cursor_delivers_each_event_in_order(self) -> None:
        append_remote_event(
            self.path, "nano4", self.job.id, self.event(1, RemoteEventKind.STARTED)
        )
        append_remote_event(
            self.path,
            "nano4",
            self.job.id,
            self.event(2, RemoteEventKind.FAILED, error="ValueError: bad model"),
        )
        first = next_unreported_event(self.path)
        assert first is not None
        self.assertEqual(first[1].sequence, 1)
        mark_remote_event_reported(self.path, self.job.id, 1)
        second = next_unreported_event(self.path)
        assert second is not None
        self.assertEqual(second[1].sequence, 2)
        mark_remote_event_reported(self.path, self.job.id, 2)
        self.assertIsNone(next_unreported_event(self.path))
        job = get_job(self.path, self.job.id)
        assert job is not None
        self.assertEqual(job.error, "ValueError: bad model")

    def test_reporter_marks_event_only_after_wandb_succeeds(self) -> None:
        append_remote_event(
            self.path, "nano4", self.job.id, self.event(1, RemoteEventKind.STARTED)
        )
        with (
            patch(
                "judge.remote_reporter.publish_event",
                side_effect=RuntimeError("offline"),
            ),
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            run_reporter(
                database_path=self.path, wandb_project="study-group", once=True
            )
        self.assertIsNotNone(next_unreported_event(self.path))
        with patch("judge.remote_reporter.publish_event", Mock()) as publish:
            run_reporter(
                database_path=self.path, wandb_project="study-group", once=True
            )
        publish.assert_called_once()
        self.assertIsNone(next_unreported_event(self.path))

    def test_log_event_prints_without_creating_a_wandb_property(self) -> None:
        run = Mock()
        run.id = self.job.id
        run.url = "https://wandb.ai/example/run"
        run.summary = {}
        with (
            patch("judge.remote_reporter.wandb.init", return_value=run),
            patch("judge.remote_reporter.set_wandb_run"),
            patch("builtins.print") as output,
        ):
            publish_event(
                self.job,
                self.event(1, RemoteEventKind.LOG, line="training\n"),
                database_path=self.path,
                wandb_project="study-group",
                wandb_entity=None,
                active_runs={},
            )

        output.assert_called_once_with("training\n", end="", flush=True)
        run.log.assert_not_called()
        run.finish.assert_not_called()
        self.assertEqual(run.summary["judge_status"], "running")

    def publish(self, job, event, active_runs) -> None:
        publish_event(
            job,
            event,
            database_path=self.path,
            wandb_project="study-group",
            wandb_entity=None,
            active_runs=active_runs,
        )

    def test_run_stays_running_until_result_is_published(self) -> None:
        run = Mock(id=self.job.id, url="https://wandb.ai/example/run", summary={})
        active_runs = {}
        with (
            patch("judge.remote_reporter.wandb.init", return_value=run) as init,
            patch("judge.remote_reporter.set_wandb_run"),
            patch("judge.remote_reporter._publish_result") as publish_result,
        ):
            self.publish(self.job, self.event(1, RemoteEventKind.STARTED), active_runs)
            self.assertEqual(run.summary["judge_status"], "running")
            run.finish.assert_not_called()
            self.publish(
                self.job,
                self.event(2, RemoteEventKind.LOG, line="training"),
                active_runs,
            )
            run.finish.assert_not_called()
            self.publish(
                self.job,
                self.event(
                    3, RemoteEventKind.COMPLETED, result=JudgeResult(passed=True)
                ),
                active_runs,
            )

        init.assert_called_once()
        self.assertEqual(init.call_args.kwargs["reinit"], "create_new")
        publish_result.assert_called_once()
        run.finish.assert_called_once_with(exit_code=0)
        self.assertEqual(active_runs, {})

    def test_concurrent_jobs_keep_separate_active_runs(self) -> None:
        other_job = self.job.model_copy(update={"id": "other-job"})
        runs = [
            Mock(id=job.id, url="https://wandb.ai/example/run", summary={})
            for job in (self.job, other_job)
        ]
        active_runs = {}
        with (
            patch("judge.remote_reporter.wandb.init", side_effect=runs) as init,
            patch("judge.remote_reporter.set_wandb_run"),
        ):
            for job in (self.job, other_job):
                self.publish(job, self.event(1, RemoteEventKind.STARTED), active_runs)
            self.publish(
                self.job,
                self.event(2, RemoteEventKind.FAILED, error="bad model"),
                active_runs,
            )

        self.assertEqual(init.call_count, 2)
        self.assertEqual(active_runs, {other_job.id: runs[1]})
        runs[0].finish.assert_called_once_with(exit_code=1)
        self.assertEqual(runs[0].summary["judge_status"], "error")
        runs[1].finish.assert_not_called()

    def test_reporting_failure_keeps_run_open_for_retry(self) -> None:
        run = Mock(id=self.job.id, url="https://wandb.ai/example/run", summary={})
        active_runs = {}
        event = self.event(
            2, RemoteEventKind.COMPLETED, result=JudgeResult(passed=True)
        )
        with (
            patch("judge.remote_reporter.wandb.init", return_value=run) as init,
            patch("judge.remote_reporter.set_wandb_run"),
            patch(
                "judge.remote_reporter._publish_result",
                side_effect=[RuntimeError("offline"), None],
            ),
        ):
            self.publish(self.job, self.event(1, RemoteEventKind.STARTED), active_runs)
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.publish(self.job, event, active_runs)
            run.finish.assert_not_called()
            self.assertEqual(active_runs, {self.job.id: run})
            self.publish(self.job, event, active_runs)

        init.assert_called_once()
        run.finish.assert_called_once_with(exit_code=0)
        self.assertEqual(active_runs, {})

    def test_reporter_reuses_run_across_durable_events(self) -> None:
        for event in (
            self.event(1, RemoteEventKind.STARTED),
            self.event(2, RemoteEventKind.LOG, line="training"),
            self.event(3, RemoteEventKind.COMPLETED, result=JudgeResult(passed=True)),
        ):
            append_remote_event(self.path, "nano4", self.job.id, event)
        run = Mock(id=self.job.id, url="https://wandb.ai/example/run", summary={})
        with (
            patch("judge.remote_reporter.wandb.init", return_value=run) as init,
            patch("judge.worker.wandb.Artifact"),
        ):
            run_reporter(
                database_path=self.path, wandb_project="study-group", once=True
            )

        init.assert_called_once()
        self.assertEqual(run.summary["judge_status"], "completed")
        run.finish.assert_called_once_with(exit_code=0)
        self.assertIsNone(next_unreported_event(self.path))


if __name__ == "__main__":
    unittest.main()
