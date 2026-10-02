import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path

from judge.database import (
    claim_next_job,
    complete_job,
    create_job,
    create_remote_job,
    fail_job,
    fail_remote_dispatch,
    get_job,
    get_sub_judge,
    heartbeat_sub_judge,
    list_sub_judges,
    mark_remote_job_queued,
    migrate_database,
    prune_stale_sub_judges,
    register_sub_judge,
    set_wandb_run,
)
from judge.models import (
    JobStatus,
    JudgeResult,
    SubJudge,
    SubJudgeBackend,
    Submission,
)


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

        self.assertEqual(version, 4)
        self.assertIn(("jobs",), tables)
        self.assertIn(("sub_judges",), tables)
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
        self.assertIsNone(migrated.assigned_judge_id)
        self.assertIsNone(migrated.slurm_job_id)

    def test_upgrade_preserves_existing_agent_and_backfills_last_seen(self) -> None:
        legacy_path = Path(self.temporary_directory.name) / "agents-v3.db"
        with closing(sqlite3.connect(legacy_path)) as connection:
            scripts = [
                files("judge.migrations").joinpath(name).read_text()
                for name in (
                    "001_initial.sql",
                    "002_sub_judges.sql",
                    "003_remote_events.sql",
                )
            ]
            migration_sql = "\n".join(scripts)
            connection.executescript(
                f"BEGIN IMMEDIATE;\n{migration_sql}\nPRAGMA user_version = 3;\nCOMMIT;"
            )
        judge = self.sub_judge()
        with closing(sqlite3.connect(legacy_path)) as connection, connection:
            connection.execute(
                "INSERT INTO sub_judges "
                "(id, backend, task_ids_json, max_gpus, judge_revision, registered_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    judge.id,
                    judge.backend.value,
                    '["gpu-lab"]',
                    8,
                    judge.judge_revision,
                    judge.registered_at.isoformat(),
                ),
            )

        migrate_database(legacy_path)

        self.assertEqual(get_sub_judge(legacy_path, judge.id), judge)
        with closing(sqlite3.connect(legacy_path)) as connection:
            last_seen = connection.execute(
                "SELECT last_seen_at FROM sub_judges WHERE id = ?", (judge.id,)
            ).fetchone()[0]
        self.assertEqual(last_seen, judge.registered_at.isoformat())

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

        self.assertEqual(version, 4)

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
        self.assertIsNone(loaded.assigned_judge_id)

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

    def test_registers_and_updates_sub_judge_capabilities(self) -> None:
        first = self.sub_judge()
        register_sub_judge(self.database_path, first)

        self.assertEqual(get_sub_judge(self.database_path, first.id), first)
        self.assertEqual(list_sub_judges(self.database_path), [first])

        updated = first.model_copy(update={"max_gpus": 2})
        register_sub_judge(self.database_path, updated)
        self.assertEqual(get_sub_judge(self.database_path, first.id), updated)
        self.assertEqual(len(list_sub_judges(self.database_path)), 1)

    def test_heartbeat_keeps_registration_until_five_minute_expiry(self) -> None:
        registered_at = datetime(2026, 9, 29, tzinfo=UTC)
        judge = self.sub_judge().model_copy(update={"registered_at": registered_at})
        register_sub_judge(self.database_path, judge)
        self.assertTrue(
            heartbeat_sub_judge(
                self.database_path,
                judge.id,
                at=registered_at + timedelta(minutes=4),
            )
        )
        self.assertEqual(
            prune_stale_sub_judges(
                self.database_path,
                now=registered_at + timedelta(minutes=9) - timedelta(seconds=1),
            ),
            [],
        )
        self.assertEqual(
            prune_stale_sub_judges(
                self.database_path, now=registered_at + timedelta(minutes=9)
            ),
            [judge.id],
        )
        self.assertIsNone(get_sub_judge(self.database_path, judge.id))
        self.assertFalse(heartbeat_sub_judge(self.database_path, judge.id))

    def test_expired_heartbeat_requires_reregistration_and_preserves_job(self) -> None:
        registered_at = datetime(2026, 9, 29, tzinfo=UTC)
        judge = self.sub_judge().model_copy(update={"registered_at": registered_at})
        register_sub_judge(self.database_path, judge)
        job = create_remote_job(
            self.database_path,
            self.submission(),
            judge_id=judge.id,
            request_key="run-1",
        )
        self.assertFalse(
            heartbeat_sub_judge(
                self.database_path,
                judge.id,
                at=registered_at + timedelta(minutes=5),
            )
        )
        self.assertEqual(
            prune_stale_sub_judges(
                self.database_path, now=registered_at + timedelta(minutes=5)
            ),
            [judge.id],
        )
        self.assertEqual(get_job(self.database_path, job.id), job)
        renewed = judge.model_copy(
            update={"registered_at": registered_at + timedelta(minutes=5)}
        )
        register_sub_judge(self.database_path, renewed)
        self.assertEqual(get_sub_judge(self.database_path, judge.id), renewed)

    def test_remote_job_is_not_claimed_by_local_worker(self) -> None:
        register_sub_judge(self.database_path, self.sub_judge())
        remote = create_remote_job(
            self.database_path,
            self.submission(),
            judge_id="nano4",
            request_key="run-1",
        )
        self.assertEqual(remote.status, JobStatus.DISPATCHING)
        self.assertIsNone(claim_next_job(self.database_path))

        queued = mark_remote_job_queued(self.database_path, remote.id, "12345")
        self.assertEqual(queued.status, JobStatus.QUEUED)
        self.assertEqual(queued.slurm_job_id, "12345")
        self.assertEqual(queued.assigned_judge_id, "nano4")
        self.assertIsNone(claim_next_job(self.database_path))

        local = create_job(self.database_path, self.submission())
        claimed = claim_next_job(self.database_path)
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.id, local.id)

    def test_remote_job_creation_is_idempotent_under_concurrency(self) -> None:
        register_sub_judge(self.database_path, self.sub_judge())

        def reserve(_: int):
            return create_remote_job(
                self.database_path,
                self.submission(),
                judge_id="nano4",
                request_key="run-1",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(reserve, range(2))

        self.assertEqual(first.id, second.id)
        with closing(sqlite3.connect(self.database_path)) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE request_key = ?", ("run-1",)
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_rejects_request_key_reuse_for_a_different_submission(self) -> None:
        register_sub_judge(self.database_path, self.sub_judge())
        create_remote_job(
            self.database_path,
            self.submission(),
            judge_id="nano4",
            request_key="run-1",
        )
        changed = self.submission().model_copy(update={"commit_sha": "b" * 40})

        with self.assertRaisesRegex(ValueError, "another submission"):
            create_remote_job(
                self.database_path,
                changed,
                judge_id="nano4",
                request_key="run-1",
            )

    def test_requires_registration_before_assigning_remote_job(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown sub-judge"):
            create_remote_job(
                self.database_path,
                self.submission(),
                judge_id="nano4",
                request_key="run-1",
            )

    def test_remote_dispatch_acceptance_is_idempotent(self) -> None:
        register_sub_judge(self.database_path, self.sub_judge())
        job = create_remote_job(
            self.database_path,
            self.submission(),
            judge_id="nano4",
            request_key="run-1",
        )

        first = mark_remote_job_queued(self.database_path, job.id, "12345")
        again = mark_remote_job_queued(self.database_path, job.id, "12345")

        self.assertEqual(first, again)
        with self.assertRaisesRegex(RuntimeError, "cannot be marked queued"):
            mark_remote_job_queued(self.database_path, job.id, "other")

    def test_fails_a_remote_job_before_queueing(self) -> None:
        register_sub_judge(self.database_path, self.sub_judge())
        job = create_remote_job(
            self.database_path,
            self.submission(),
            judge_id="nano4",
            request_key="run-1",
        )

        failed = fail_remote_dispatch(self.database_path, job.id, "sbatch rejected")

        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(failed.error, "sbatch rejected")
        self.assertIsNotNone(failed.finished_at)
        with self.assertRaisesRegex(RuntimeError, "cannot be marked queued"):
            mark_remote_job_queued(self.database_path, job.id, "12345")

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

    @staticmethod
    def sub_judge() -> SubJudge:
        return SubJudge(
            id="nano4",
            backend=SubJudgeBackend.SLURM,
            task_ids=["gpu-lab"],
            max_gpus=8,
            judge_revision="test-revision",
            registered_at=datetime.now(UTC),
        )


if __name__ == "__main__":
    unittest.main()
