import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from importlib.resources import files
from pathlib import Path

from judge.database import (
    MIGRATIONS,
    claim_next_job,
    complete_job,
    create_job,
    create_remote_job,
    fail_job,
    get_job,
    migrate_database,
    pending_ssh_jobs,
    set_wandb_run,
)
from judge.models import JobStatus, JudgeResult, Resources, Submission
from judge.ssh import RemoteConfig


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "judge.db"
        migrate_database(self.database_path)

    def test_applies_migrations_once(self) -> None:
        migrate_database(self.database_path)

        with closing(sqlite3.connect(self.database_path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()

        self.assertEqual(version, 6)
        self.assertIn(("jobs",), tables)
        self.assertNotIn(("sub_judges",), tables)
        self.assertIn(("remote_events",), tables)

    def test_upgrades_a_database_with_existing_jobs(self) -> None:
        legacy_path = Path(self.temporary_directory.name) / "legacy.db"
        initial_sql = files("judge.migrations").joinpath("001_initial.sql").read_text()
        with closing(sqlite3.connect(legacy_path)) as connection:
            connection.executescript(
                f"BEGIN IMMEDIATE;\n{initial_sql}\nPRAGMA user_version = 1;\nCOMMIT;"
            )
        created = create_job(legacy_path, self.submission())

        migrate_database(legacy_path)
        migrated = get_job(legacy_path, created.id)

        self.assertIsNotNone(migrated)
        assert migrated is not None
        self.assertEqual(migrated.submission, created.submission)
        self.assertEqual(migrated.status, JobStatus.QUEUED)
        self.assertNotIn("assigned_judge_id", migrated.model_dump())
        self.assertIsNone(migrated.slurm_job_id)

    def test_rejects_a_database_from_a_newer_judge(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("PRAGMA user_version = 999")

        with self.assertRaisesRegex(RuntimeError, "newer than this judge"):
            migrate_database(self.database_path)

    def test_serializes_concurrent_migration_attempts(self) -> None:
        database_path = Path(self.temporary_directory.name) / "concurrent.db"

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: migrate_database(database_path), range(2)))

        with closing(sqlite3.connect(database_path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]

        self.assertEqual(version, 6)

    def test_round_trips_a_queued_job(self) -> None:
        submission = Submission(
            repo_url="https://github.com/cerulean-works/example.git",
            commit_sha="a" * 40,
            task_id="example",
            github_actor="student",
        )

        created = create_job(self.database_path, submission)
        loaded = get_job(self.database_path, created.id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.submission, submission)
        self.assertEqual(loaded.status, JobStatus.QUEUED)
        self.assertNotIn("assigned_judge_id", loaded.model_dump())

    def test_returns_none_for_an_unknown_job(self) -> None:
        self.assertIsNone(get_job(self.database_path, "missing"))

    def test_only_one_concurrent_worker_claims_a_job(self) -> None:
        create_job(self.database_path, self.submission())

        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = list(
                executor.map(lambda _: claim_next_job(self.database_path), range(2))
            )

        jobs = [job for job in claimed if job is not None]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, JobStatus.RUNNING)
        self.assertIsNotNone(jobs[0].started_at)

    def remote(self, submission=None, key="run-1"):
        return create_remote_job(
            self.database_path,
            submission or self.submission(),
            request_key=key,
            config=RemoteConfig(host="nano4", user="judge", work_root="/work/oj"),
            resources=Resources(gpus=1),
        )

    def test_remote_queue_is_isolated_and_persists_configuration(self):
        remote = self.remote()
        self.assertEqual(remote.status, JobStatus.QUEUED)
        self.assertIsNone(claim_next_job(self.database_path))
        requests = pending_ssh_jobs(self.database_path)
        self.assertEqual(requests[0].job_id, remote.id)
        self.assertNotIn("revision", requests[0].config.model_dump())
        self.assertEqual(requests[0].resources.gpus, 1)
        local = create_job(self.database_path, self.submission())
        claimed = claim_next_job(self.database_path)
        assert claimed is not None
        self.assertEqual(claimed.id, local.id)

    def test_remote_creation_is_idempotent_under_concurrency(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(lambda _: self.remote(), range(2))
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(pending_ssh_jobs(self.database_path)), 1)
        with self.assertRaisesRegex(ValueError, "another submission"):
            self.remote(self.submission().model_copy(update={"commit_sha": "c" * 40}))

    def test_upgrade_preserves_historical_remote_jobs_and_reports(self):
        path = Path(self.temporary_directory.name) / "v4.db"
        with closing(sqlite3.connect(path)) as connection, connection:
            for name in (
                "001_initial.sql",
                "002_sub_judges.sql",
                "003_remote_events.sql",
                "004_sub_judge_heartbeats.sql",
            ):
                connection.executescript(
                    files("judge.migrations").joinpath(name).read_text()
                )
            connection.execute("PRAGMA user_version = 4")
            connection.execute(
                "INSERT INTO jobs (id, repo_url, commit_sha, task_id, github_actor, status, created_at, assigned_judge_id, slurm_job_id) "
                "VALUES ('legacy', 'https://github.com/example/repo.git', ?, 'lab1', 'student', 'completed', '2026-01-01', 'nano4', '123')",
                ("a" * 40,),
            )
            connection.execute(
                "INSERT INTO remote_events VALUES ('legacy', 1, '{}', '2026-01-01')"
            )
        migrate_database(path)
        job = get_job(path, "legacy")
        assert job is not None
        self.assertEqual(job.execution_backend.value, "ssh_slurm")
        self.assertNotIn("assigned_judge_id", job.model_dump())
        self.assertEqual(job.slurm_job_id, "123")
        with closing(sqlite3.connect(path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            self.assertNotIn("assigned_judge_id", columns)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM remote_events").fetchone()[0],
                1,
            )

    def test_reporting_upgrade_preserves_existing_ssh_queue(self):
        path = Path(self.temporary_directory.name) / "v5.db"
        with closing(sqlite3.connect(path)) as connection:
            for name in MIGRATIONS[:5]:
                connection.executescript(
                    files("judge.migrations").joinpath(name).read_text()
                )
            connection.execute("PRAGMA user_version = 5")
        job = create_remote_job(
            path,
            self.submission(),
            resources=Resources(gpus=1),
            request_key="existing",
            config=RemoteConfig(host="nano4", user="judge", work_root="/work/oj"),
        )
        migrate_database(path)
        migrate_database(path)
        self.assertEqual(pending_ssh_jobs(path)[0].job_id, job.id)
        self.assertEqual(get_job(path, job.id), job)
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(
                connection.execute("SELECT report_pending FROM ssh_jobs").fetchone()[0],
                0,
            )

    def test_upgrade_constraint_uses_normal_sqlite_failure_path(self):
        path = Path(self.temporary_directory.name) / "pending-v4.db"
        with closing(sqlite3.connect(path)) as connection, connection:
            for name in (
                "001_initial.sql",
                "002_sub_judges.sql",
                "003_remote_events.sql",
                "004_sub_judge_heartbeats.sql",
            ):
                connection.executescript(
                    files("judge.migrations").joinpath(name).read_text()
                )
            connection.execute("PRAGMA user_version = 4")
            connection.execute(
                "INSERT INTO jobs (id, repo_url, commit_sha, task_id, github_actor, status, created_at, assigned_judge_id) "
                "VALUES ('legacy', 'https://github.com/example/repo.git', ?, 'lab1', 'student', 'running', '2026-01-01', 'nano4')",
                ("a" * 40,),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            migrate_database(path)
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertEqual(
                connection.execute("SELECT status FROM jobs").fetchone()[0], "running"
            )

    def test_completes_a_running_job_with_its_result(self) -> None:
        created = create_job(self.database_path, self.submission())
        claim_next_job(self.database_path)

        completed = complete_job(
            self.database_path,
            created.id,
            JudgeResult(passed=True),
        )

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertTrue(completed.result and completed.result.passed)
        self.assertIsNotNone(completed.finished_at)

    def test_attaches_a_wandb_run_to_a_running_job(self) -> None:
        created = create_job(self.database_path, self.submission())
        claim_next_job(self.database_path)

        updated = set_wandb_run(
            self.database_path,
            created.id,
            run_id="wandb-run",
            url="https://wandb.example/run",
        )

        self.assertEqual(updated.wandb_run_id, "wandb-run")
        self.assertEqual(updated.wandb_url, "https://wandb.example/run")

    def test_rejects_attaching_a_wandb_run_to_a_queued_job(self) -> None:
        created = create_job(self.database_path, self.submission())

        with self.assertRaisesRegex(RuntimeError, "is not running"):
            set_wandb_run(
                self.database_path,
                created.id,
                run_id="wandb-run",
                url=None,
            )

    def test_fails_a_running_job_with_an_error(self) -> None:
        created = create_job(self.database_path, self.submission())
        claim_next_job(self.database_path)

        failed = fail_job(self.database_path, created.id, "checkout failed")

        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(failed.error, "checkout failed")
        self.assertIsNotNone(failed.finished_at)

    @staticmethod
    def submission() -> Submission:
        return Submission(
            repo_url="https://github.com/cerulean-works/example.git",
            commit_sha="a" * 40,
            task_id="example",
            github_actor="student",
        )


if __name__ == "__main__":
    unittest.main()
