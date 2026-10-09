import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from fcntl import LOCK_EX
from fcntl import flock as lock_file
from importlib.resources import files
from pathlib import Path


class SetupRepoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Judge Test")
        self.git("config", "user.email", "judge@example.com")
        (self.source / ".gitignore").write_text(".venv/\n")
        (self.source / "uv.lock").write_text("submitted lock\n")
        (self.source / "solution.txt").write_text("first\n")
        self.git("add", ".")
        self.git("commit", "-m", "first")
        self.revision = self.git("rev-parse", "HEAD").stdout.strip()
        (self.source / "solution.txt").write_text("second\n")
        self.git("commit", "-am", "second")
        self.second_revision = self.git("rev-parse", "HEAD").stdout.strip()
        self.destination = self.root / "checkout with spaces"
        binary = self.root / "bin"
        binary.mkdir()
        uv = binary / "uv"
        uv.write_text(
            f"#!{sys.executable}\n"
            "from pathlib import Path\n"
            "import os, sys\n"
            "assert sys.argv[1:] == ['sync', '--no-sources', '--no-dev']\n"
            "assert not any(os.environ.get(name) for name in ['WANDB_API_KEY', 'TS_AUTHKEY', 'JUDGE_API_TOKEN'])\n"
            "Path('.venv').mkdir(exist_ok=True)\n"
            "Path('.venv/bin').mkdir(exist_ok=True)\n"
            "python = Path('.venv/bin/python')\n"
            "if not python.exists(): python.symlink_to(sys.executable)\n"
            "Path('uv.lock').write_text('CUDA resolved lock')\n"
            "sys.exit(int(os.environ.get('FAKE_UV_EXIT', '0')))\n"
        )
        uv.chmod(0o755)
        if shutil.which("flock") is None:
            # macOS lacks the Linux utility; emulate locking the inherited FD.
            flock = binary / "flock"
            flock.write_text(
                f"#!{sys.executable}\n"
                "import fcntl, sys\n"
                "flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if '-n' in sys.argv else 0)\n"
                "try:\n    fcntl.flock(int(sys.argv[-1]), flags)\n"
                "except BlockingIOError:\n    sys.exit(1)\n"
            )
            flock.chmod(0o755)
        self.environment = dict(
            os.environ,
            PATH=f"{binary}:{os.environ['PATH']}",
            WANDB_API_KEY="test-secret",
            TS_AUTHKEY="test-secret",
            JUDGE_API_TOKEN="test-secret",
        )

    def git(self, *arguments, cwd=None):
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd or self.source,
            text=True,
            capture_output=True,
            check=True,
        )

    def setup(self, revision=None, repo_url=None):
        return subprocess.run(
            [
                "bash",
                str(files("judge").joinpath("setup-repo.sh")),
                repo_url or str(self.source),
                revision or self.revision,
                str(self.destination),
            ],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_clones_exact_commit_and_restores_lock_after_repeated_setup(self):
        first = self.setup()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(
            self.git("rev-parse", "HEAD", cwd=self.destination).stdout.strip(),
            self.revision,
        )
        self.assertEqual((self.destination / "solution.txt").read_text(), "first\n")
        self.assertEqual((self.destination / "uv.lock").read_text(), "submitted lock\n")
        again = self.setup()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(
            self.git("status", "--porcelain", cwd=self.destination).stdout, ""
        )

    def test_fetches_and_checks_out_new_exact_revision(self):
        self.assertEqual(self.setup().returncode, 0)
        updated = self.setup(self.second_revision)
        self.assertEqual(updated.returncode, 0, updated.stderr)
        self.assertEqual((self.destination / "solution.txt").read_text(), "second\n")

    def advance_default_branch(self):
        self.git("branch", "-m", "new-default")
        (self.source / "solution.txt").write_text("latest\n")
        self.git("commit", "-am", "update default branch")
        return self.git("rev-parse", "HEAD").stdout.strip()

    def test_latest_fetches_current_default_branch_for_existing_checkout(self):
        first = self.setup("latest")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual((self.destination / "solution.txt").read_text(), "second\n")
        latest = self.advance_default_branch()
        updated = self.setup("latest")
        self.assertEqual(updated.returncode, 0, updated.stderr)
        self.assertEqual(
            self.git("rev-parse", "HEAD", cwd=self.destination).stdout.strip(), latest
        )
        self.assertEqual((self.destination / "uv.lock").read_text(), "submitted lock\n")

    def test_each_new_job_gets_latest_while_reconnect_reuses_prepared_evaluator(self):
        self.environment["JUDGE_SETUP_ONCE"] = "1"
        self.assertEqual(self.setup("latest").returncode, 0)
        latest = self.advance_default_branch()
        self.assertEqual(self.setup("latest").returncode, 0)
        self.assertEqual(
            self.git("rev-parse", "HEAD", cwd=self.destination).stdout.strip(),
            self.second_revision,
        )
        self.destination = self.root / "next-job-judge"
        self.assertEqual(self.setup("latest").returncode, 0)
        self.assertEqual(
            self.git("rev-parse", "HEAD", cwd=self.destination).stdout.strip(), latest
        )

    def test_rejects_dirty_or_mismatched_existing_repository(self):
        self.assertEqual(self.setup().returncode, 0)
        (self.destination / "solution.txt").write_text("modified\n")
        failed = self.setup()
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("dirty", failed.stderr)
        self.assertEqual((self.destination / "solution.txt").read_text(), "modified\n")
        self.git(
            "remote",
            "set-url",
            "origin",
            "https://github.com/wrong/repo.git",
            cwd=self.destination,
        )
        failed = self.setup()
        self.assertIn("origin mismatch", failed.stderr)

    def test_restores_lock_when_uv_fails(self):
        self.environment["FAKE_UV_EXIT"] = "1"
        self.assertNotEqual(self.setup().returncode, 0)
        self.assertEqual((self.destination / "uv.lock").read_text(), "submitted lock\n")
        self.environment["FAKE_UV_EXIT"] = "0"
        self.assertEqual(self.setup().returncode, 0)

    def test_rejects_missing_commit_and_invalid_sha(self):
        self.assertNotEqual(self.setup("c" * 40).returncode, 0)
        self.assertNotEqual(self.setup("main").returncode, 0)
        self.assertFalse((self.destination / ".venv").exists())

    def test_cleanup_expires_reported_jobs_and_preserves_active_locked_and_recent_jobs(
        self,
    ):
        remote = self.root / "remote"
        self.environment["JUDGE_REMOTE_WORK_ROOT"] = str(remote)
        records = remote / "job-records"
        records.mkdir(parents=True)
        locks = remote / "job-locks"
        locks.mkdir()
        jobs = []
        for number in range(1, 6):
            job_id = str(number) * 32
            job = remote / "jobs" / job_id
            job.mkdir(parents=True)
            (job / "submission").mkdir()
            (records / f"{job_id}.json").write_text("{}")
            if number != 2:  # No marker: running or reporting still pending.
                marker = job / ".finished"
                marker.touch()
                age = (6 if number == 3 else 8) * 86400
                os.utime(marker, (time.time() - age, time.time() - age))
            jobs.append(job)
        # A workspace with no durable record must never be deleted.
        (records / f"{'5' * 32}.json").unlink()
        with (locks / f"{'4' * 32}.lock").open("w") as held:
            lock_file(held, LOCK_EX)
            result = self.setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(jobs[0].exists())
        self.assertTrue(all(job.exists() for job in jobs[1:]))
        self.assertTrue((records / f"{'1' * 32}.json").exists())

    def test_cleanup_does_not_follow_symlinked_workspaces(self):
        remote = self.root / "remote"
        (remote / "jobs").mkdir(parents=True)
        target = self.root / "keep"
        target.mkdir()
        marker = target / ".finished"
        marker.touch()
        old = time.time() - 8 * 86400
        os.utime(marker, (old, old))
        (remote / "jobs" / ("1" * 32)).symlink_to(target, target_is_directory=True)
        self.environment["JUDGE_REMOTE_WORK_ROOT"] = str(remote)
        result = self.setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(target.exists())
