import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judge.database import claim_next_job, create_job, get_job, migrate_database
from judge.executor import DockerExecutor, ExecutionResult
from judge.models import Job, JobStatus, JudgeResult, Submission, TestResult
from judge.repository import CheckoutError
from judge.tasks import TASKS
from judge.tasks.base import Task
from judge.worker import run_job


class ExampleTask(Task):
    id = "example"

    def evaluate(self, submission: Path) -> JudgeResult:
        return JudgeResult(passed=submission.is_dir())


class WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "judge.db"
        self.work_root = self.root / "work"
        migrate_database(self.database_path)

        self.original_tasks = TASKS.copy()
        TASKS.clear()
        TASKS[ExampleTask.id] = ExampleTask()
        self.addCleanup(self.restore_tasks)

        created = create_job(
            self.database_path,
            Submission(
                repo_url="https://github.com/cerulean-works/example.git",
                commit_sha="a" * 40,
                task_id="example",
                github_actor="student",
            ),
        )
        claimed = claim_next_job(self.database_path)
        assert claimed is not None
        self.job: Job = claimed
        self.assertEqual(self.job.id, created.id)

        self.executor = Mock(spec=DockerExecutor)
        self.output: list[str] = []
        self.wandb_run = Mock()
        self.wandb_run.id = "wandb-run"
        self.wandb_run.url = "https://wandb.example/run"
        self.wandb_run.summary = {}
        wandb_init = patch("judge.worker.wandb.init", return_value=self.wandb_run)
        self.mock_wandb_init = wandb_init.start()
        self.addCleanup(wandb_init.stop)

    def restore_tasks(self) -> None:
        TASKS.clear()
        TASKS.update(self.original_tasks)

    @staticmethod
    def fake_checkout(
        repo_url: str,
        commit_sha: str,
        destination: Path,
    ) -> Path:
        destination.mkdir()
        return destination

    def test_persists_a_valid_result(self) -> None:
        def execute(**arguments: object) -> ExecutionResult:
            self.assertEqual(self.wandb_run.summary["judge_status"], "running")
            self.wandb_run.finish.assert_not_called()
            output_directory = arguments["output_directory"]
            assert isinstance(output_directory, Path)
            output_directory.mkdir()
            (output_directory / "result.json").write_text(
                JudgeResult(
                    passed=True,
                    score=0.9,
                    metrics={"accuracy": 0.8},
                    tests=[TestResult(name="loads model", passed=True)],
                ).model_dump_json()
            )
            return ExecutionResult(returncode=0)

        self.executor.run.side_effect = execute

        with (
            patch("judge.worker.checkout_repository", side_effect=self.fake_checkout),
            patch("judge.worker.wandb.Table") as wandb_table,
            patch("judge.worker.wandb.Artifact") as wandb_artifact,
        ):
            completed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
                wandb_project="study-group",
            )

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertTrue(completed.result and completed.result.passed)
        self.assertEqual(completed.wandb_run_id, "wandb-run")
        self.assertEqual(completed.wandb_url, "https://wandb.example/run")
        self.mock_wandb_init.assert_called_once()
        self.assertEqual(
            self.mock_wandb_init.call_args.kwargs["name"],
            f"example-student-{self.job.id[:8]}",
        )
        self.wandb_run.log.assert_any_call(
            {"accuracy": 0.8, "score": 0.9, "passed": True}
        )
        wandb_table.assert_called_once_with(
            columns=["test", "passed", "message"],
            data=[["loads model", True, None]],
        )
        self.wandb_run.log.assert_any_call({"tests": wandb_table.return_value})
        wandb_artifact.assert_called_once_with(
            name=f"judge-result-{self.job.id}", type="judge-result"
        )
        wandb_artifact.return_value.add_file.assert_called_once_with(
            str(self.work_root / self.job.id / "output" / "result.json"),
            name="result.json",
        )
        self.wandb_run.log_artifact.assert_called_once_with(wandb_artifact.return_value)
        self.assertEqual(self.wandb_run.summary["judge_status"], "completed")
        self.wandb_run.finish.assert_called_once_with()
        self.assertIn(f"[judge] completed job {self.job.id}\n", self.output)
        self.assertIn("[judge] verdict: PASS\n", self.output)

    def test_logs_failed_test_reason(self) -> None:
        def execute(**arguments: object) -> ExecutionResult:
            output_directory = arguments["output_directory"]
            assert isinstance(output_directory, Path)
            output_directory.mkdir()
            (output_directory / "result.json").write_text(
                JudgeResult(
                    passed=False,
                    score=0,
                    tests=[
                        TestResult(name="sample_01", passed=False, message="bad logits")
                    ],
                ).model_dump_json()
            )
            return ExecutionResult(returncode=0)

        self.executor.run.side_effect = execute
        with (
            patch("judge.worker.checkout_repository", side_effect=self.fake_checkout),
            patch("judge.worker.wandb.Table"),
            patch("judge.worker.wandb.Artifact"),
        ):
            completed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
                wandb_project="study-group",
            )

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertFalse(completed.result and completed.result.passed)
        self.assertIn("[judge] verdict: FAIL\n", self.output)
        self.assertIn("[judge] failed sample_01: bad logits\n", self.output)

    def test_persists_checkout_failures(self) -> None:
        with patch(
            "judge.worker.checkout_repository",
            side_effect=CheckoutError("clone failed"),
        ):
            failed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
                wandb_project="study-group",
            )

        persisted = get_job(self.database_path, self.job.id)
        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(persisted, failed)
        self.assertEqual(failed.error, "CheckoutError: clone failed")
        self.assertEqual(self.wandb_run.summary["judge_status"], "error")
        self.wandb_run.finish.assert_called_once_with()

    def test_persists_nonzero_runner_exits(self) -> None:
        self.executor.run.return_value = ExecutionResult(returncode=7)

        with patch("judge.worker.checkout_repository", side_effect=self.fake_checkout):
            failed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
                wandb_project="study-group",
            )

        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(failed.error, "RuntimeError: Task runner exited with status 7")

    def test_does_not_complete_a_job_when_reporting_fails(self) -> None:
        def execute(**arguments: object) -> ExecutionResult:
            output_directory = arguments["output_directory"]
            assert isinstance(output_directory, Path)
            output_directory.mkdir()
            (output_directory / "result.json").write_text(
                JudgeResult(passed=True).model_dump_json()
            )
            return ExecutionResult(returncode=0)

        self.executor.run.side_effect = execute
        self.wandb_run.log_artifact.side_effect = RuntimeError("upload failed")

        with (
            patch("judge.worker.checkout_repository", side_effect=self.fake_checkout),
            patch("judge.worker.wandb.Artifact"),
        ):
            failed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
                wandb_project="study-group",
            )

        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(failed.error, "RuntimeError: upload failed")
        self.assertIsNone(failed.result)

    def test_persists_wandb_initialization_failures(self) -> None:
        self.mock_wandb_init.side_effect = RuntimeError("wandb unavailable")

        failed = run_job(
            self.job,
            database_path=self.database_path,
            work_root=self.work_root,
            executor=self.executor,
            on_output=self.output.append,
            wandb_project="study-group",
        )

        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(failed.error, "RuntimeError: wandb unavailable")
        self.executor.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
