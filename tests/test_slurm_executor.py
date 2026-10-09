import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judge.models import Resources
from judge.slurm_executor import SlurmExecutor


class SlurmExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.submission = root / "submission"
        self.script = self.submission / "src" / "labs" / "lab2.sbatch"
        self.script.parent.mkdir(parents=True)
        self.script.write_text("#!/bin/bash\n")
        self.output = root / "output"
        self.executor = SlurmExecutor(trusted_root=root / "trusted")

    def command(self, resources: Resources) -> list[str]:
        return self.executor.build_submit_command(
            task_id="lab2",
            resources=resources,
            submission=self.submission,
            output_directory=self.output,
        )

    def test_uses_dev_partition_and_judge_resource_limits(self) -> None:
        command = self.command(
            Resources(cpus=12, memory_gb=180, gpus=1, timeout_seconds=3600)
        )
        self.assertIn("--account=ACD115198", command)
        self.assertIn("--partition=dev", command)
        self.assertIn("--gres=gpu:1", command)
        self.assertIn("--cpus-per-task=12", command)
        self.assertIn("--mem=180G", command)
        self.assertIn("--time=60", command)
        self.assertEqual(command[-1], str(self.script.resolve()))

    def test_uses_long_partition_for_more_than_four_hours(self) -> None:
        command = self.command(
            Resources(cpus=24, memory_gb=200, gpus=2, timeout_seconds=14401)
        )
        self.assertIn("--partition=8gpus", command)
        self.assertIn("--time=241", command)

    def test_rejects_invalid_resources_and_missing_script(self) -> None:
        for resources in (
            Resources(gpus=0),
            Resources(gpus=9),
            Resources(gpus=1, cpus=13),
            Resources(gpus=1, memory_gb=201),
            Resources(gpus=1, timeout_seconds=48 * 3600 + 1),
        ):
            with self.subTest(resources=resources), self.assertRaises(ValueError):
                self.command(resources)
        self.script.unlink()
        with self.assertRaises(FileNotFoundError):
            self.command(Resources(gpus=1))

    def test_submits_without_credentials_and_parses_job_id(self) -> None:
        process = Mock(stdout="123456;cluster\n")
        with (
            patch("judge.slurm_executor.subprocess.run", return_value=process) as run,
            patch.dict(
                "os.environ",
                {"TS_AUTHKEY": "secret", "WANDB_API_KEY": "secret"},
            ),
        ):
            job_id = self.executor.submit(
                task_id="lab2",
                resources=Resources(gpus=1),
                submission=self.submission,
                output_directory=self.output,
            )
        self.assertEqual(job_id, "123456")
        environment = run.call_args.kwargs["env"]
        self.assertNotIn("TS_AUTHKEY", environment)
        self.assertNotIn("WANDB_API_KEY", environment)
        self.assertEqual(environment["JUDGE_TASK_ID"], "lab2")

    def test_reads_main_sacct_record(self) -> None:
        process = Mock(stdout="123456|COMPLETED|0:0\n123456.batch|COMPLETED|0:0\n")
        with patch("judge.slurm_executor.subprocess.run", return_value=process):
            state = self.executor.status("123456")
        assert state is not None
        self.assertTrue(state.terminal)
        self.assertTrue(state.succeeded)

    def test_scheduler_responses_and_accounting_delays_are_logged(self):
        self.executor.job_id = "1" * 32
        process = subprocess.CompletedProcess("sacct", 0, "", "accounting delayed")
        with (
            patch("judge.slurm_executor.subprocess.run", return_value=process),
            self.assertLogs("judge.execution", level="INFO") as logs,
        ):
            self.assertIsNone(self.executor.status("123456"))
        text = "\n".join(logs.output)
        self.assertIn("sacct --jobs 123456", text)
        self.assertIn("slurm.command.response", text)
        self.assertIn('stdout=""', text)
        self.assertIn("accounting delayed", text)
        self.assertIn("slurm.status.unavailable", text)
        self.assertIn("1" * 32, text)

    def test_scheduler_failures_log_output_and_preserve_retry_exceptions(self):
        for error in (
            subprocess.CalledProcessError(1, "sacct", stderr="accounting unavailable"),
            subprocess.TimeoutExpired("sacct", 30, stderr=b"accounting timeout"),
            FileNotFoundError("sacct missing"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch("judge.slurm_executor.subprocess.run", side_effect=error),
                self.assertLogs("judge.execution", level="WARNING") as logs,
                self.assertRaises(type(error)) as raised,
            ):
                self.executor.status("123456")
            self.assertIs(raised.exception, error)
            text = "\n".join(logs.output)
            self.assertIn("slurm.command.failed", text)
            self.assertIn(type(error).__name__, text)
            if isinstance(error, subprocess.SubprocessError):
                self.assertIn("accounting", text)


if __name__ == "__main__":
    unittest.main()
