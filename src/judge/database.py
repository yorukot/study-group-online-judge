import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from fcntl import LOCK_EX, LOCK_UN, flock
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from judge.models import (
    ExecutionBackend,
    Job,
    JobStatus,
    JudgeResult,
    RemoteEvent,
    RemoteEventKind,
    Resources,
    Submission,
)
from judge.ssh import RemoteConfig, RemoteRequest, RemoteSnapshot

MIGRATIONS = (
    "001_initial.sql",
    "002_sub_judges.sql",
    "003_remote_events.sql",
    "004_sub_judge_heartbeats.sql",
    "005_ssh_slurm.sql",
    "006_remote_reporting.sql",
)


def migrate_database(path: Path) -> None:
    """Apply each pending database migration in order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.migrate.lock")
    with lock_path.open("a") as lock:
        flock(lock, LOCK_EX)
        try:
            _migrate_database(path)
        finally:
            flock(lock, LOCK_UN)


def _migrate_database(path: Path) -> None:
    with closing(_connect(path)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        current_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if current_version > len(MIGRATIONS):
            raise RuntimeError(
                f"Database schema version {current_version} is newer than this judge"
            )

        for version, filename in enumerate(MIGRATIONS, start=1):
            if version <= current_version:
                continue

            migration = (
                files("judge.migrations").joinpath(filename).read_text(encoding="utf-8")
            )
            try:
                connection.executescript(
                    f"BEGIN IMMEDIATE;\n{migration}\nPRAGMA user_version = {version};\nCOMMIT;"
                )
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise


def create_job(path: Path, submission: Submission) -> Job:
    """Persist a queued submission and return its job record."""

    job = Job(
        id=uuid4().hex,
        submission=submission,
        status=JobStatus.QUEUED,
        created_at=datetime.now(UTC),
    )

    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id,
                repo_url,
                commit_sha,
                task_id,
                github_actor,
                status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                submission.repo_url,
                submission.commit_sha,
                submission.task_id,
                submission.github_actor,
                job.status.value,
                job.created_at.isoformat(),
            ),
        )

    return job


def create_remote_job(
    path: Path,
    submission: Submission,
    *,
    config: RemoteConfig,
    resources: Resources,
    request_key: str,
) -> Job:
    """Atomically queue a GPU job and freeze its remote configuration."""
    if not request_key:
        raise ValueError("request_key must not be empty")
    with closing(_connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM jobs WHERE repo_url = ? AND request_key = ?",
            (submission.repo_url, request_key),
        ).fetchone()
        if row is not None:
            existing = _job_from_row(row)
            if existing.submission != submission:
                raise ValueError("request_key was already used for another submission")
            return existing
        job = Job(
            id=uuid4().hex,
            submission=submission,
            status=JobStatus.QUEUED,
            created_at=datetime.now(UTC),
            execution_backend=ExecutionBackend.SSH_SLURM,
        )
        connection.execute(
            "INSERT INTO jobs (id, repo_url, commit_sha, task_id, github_actor, "
            "status, created_at, execution_backend, request_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job.id,
                submission.repo_url,
                submission.commit_sha,
                submission.task_id,
                submission.github_actor,
                job.status.value,
                job.created_at.isoformat(),
                job.execution_backend.value,
                request_key,
            ),
        )
        connection.execute(
            "INSERT INTO ssh_jobs (job_id, config_json, resources_json) VALUES (?, ?, ?)",
            (job.id, config.model_dump_json(), resources.model_dump_json()),
        )
    return job


def pending_ssh_jobs(path: Path) -> list[RemoteRequest]:
    """Recover all unfinished SSH jobs, including preparation after a restart."""
    with closing(_connect(path)) as connection:
        rows = connection.execute(
            "SELECT j.*, s.config_json, s.resources_json, s.setup_offset, s.log_offset "
            "FROM jobs j JOIN ssh_jobs s ON s.job_id = j.id "
            "WHERE j.status IN ('queued', 'dispatching', 'running') OR s.report_pending = 1 "
            "ORDER BY j.created_at, j.id"
        ).fetchall()
    return [
        RemoteRequest(
            job_id=row["id"],
            submission=_job_from_row(row).submission,
            config=RemoteConfig.model_validate_json(row["config_json"]),
            resources=Resources.model_validate_json(row["resources_json"]),
            setup_offset=row["setup_offset"],
            log_offset=row["log_offset"],
        )
        for row in rows
    ]


def begin_remote_job(path: Path, job_id: str) -> None:
    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            "UPDATE jobs SET status = 'dispatching' WHERE id = ? "
            "AND execution_backend = 'ssh_slurm' AND status = 'queued' AND slurm_job_id IS NULL",
            (job_id,),
        )
        if cursor.rowcount:
            _append_remote_event(
                connection,
                job_id,
                RemoteEvent(
                    sequence=_next_sequence(connection, job_id),
                    kind=RemoteEventKind.LOG,
                    line="[judge] preparing remote repository\n",
                ),
            )


def get_job(path: Path, job_id: str) -> Job | None:
    """Load a job by ID, or return ``None`` when it does not exist."""

    with closing(_connect(path)) as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

    return None if row is None else _job_from_row(row)


def get_job_by_request_key(path: Path, repo_url: str, request_key: str) -> Job | None:
    """Find a prior remote submission attempt for an idempotency key."""

    with closing(_connect(path)) as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE repo_url = ? AND request_key = ?",
            (repo_url, request_key),
        ).fetchone()
    return None if row is None else _job_from_row(row)


def claim_next_job(path: Path) -> Job | None:
    """Atomically move the oldest queued job into the running state."""

    with closing(_connect(path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = ? AND execution_backend = 'docker'
                ORDER BY created_at, id
                LIMIT 1
                """,
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None

            started_at = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?
                WHERE id = ? AND status = ? AND execution_backend = 'docker'
                """,
                (
                    JobStatus.RUNNING.value,
                    started_at,
                    row["id"],
                    JobStatus.QUEUED.value,
                ),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (row["id"],),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    if claimed is None:
        raise RuntimeError("Claimed job disappeared from the database")
    return _job_from_row(claimed)


def set_wandb_run(
    path: Path,
    job_id: str,
    *,
    run_id: str,
    url: str | None,
) -> Job:
    """Attach a W&B run to a local or remotely completed job."""

    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET wandb_run_id = ?, wandb_url = ?
            WHERE id = ? AND (
                status = ? OR execution_backend != 'docker'
            )
            """,
            (run_id, url, job_id, JobStatus.RUNNING.value),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Job {job_id!r} is not running")

    job = get_job(path, job_id)
    if job is None:
        raise RuntimeError(f"Job {job_id!r} disappeared from the database")
    return job


def complete_job(path: Path, job_id: str, result: JudgeResult) -> Job:
    """Store a valid result and mark a running job completed."""

    return _finish_job(
        path,
        job_id,
        status=JobStatus.COMPLETED,
        result_json=result.model_dump_json(),
        error=None,
    )


def fail_job(path: Path, job_id: str, error: str) -> Job:
    """Store an infrastructure error and mark a running job failed."""

    return _finish_job(
        path,
        job_id,
        status=JobStatus.ERROR,
        result_json=None,
        error=error,
    )


def _next_sequence(connection: sqlite3.Connection, job_id: str) -> int:
    return connection.execute(
        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM remote_events WHERE job_id = ?",
        (job_id,),
    ).fetchone()[0]


def _append_remote_event(
    connection: sqlite3.Connection, job_id: str, event: RemoteEvent
) -> None:
    row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None or row["execution_backend"] != "ssh_slurm":
        raise ValueError("Job is not an SSH Slurm job")
    encoded = event.model_dump_json()
    existing = connection.execute(
        "SELECT event_json FROM remote_events WHERE job_id = ? AND sequence = ?",
        (job_id, event.sequence),
    ).fetchone()
    if existing:
        if existing[0] != encoded:
            raise ValueError("Event sequence already contains different data")
        return
    expected = _next_sequence(connection, job_id)
    if event.sequence != expected:
        raise ValueError(f"Expected event sequence {expected}")
    if row["status"] in ("completed", "error"):
        raise ValueError("Job is not running")
    if event.slurm_job_id and row["slurm_job_id"] not in (None, event.slurm_job_id):
        raise ValueError("Slurm job ID does not match the scheduling receipt")
    if event.kind == RemoteEventKind.STARTED:
        if row["status"] == "running":
            raise ValueError("Job is already running")
        connection.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, slurm_job_id = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), event.slurm_job_id, job_id),
        )
    elif event.kind in (RemoteEventKind.COMPLETED, RemoteEventKind.FAILED):
        if event.kind == RemoteEventKind.COMPLETED and row["status"] != "running":
            raise ValueError("Job is not running")
        connection.execute(
            "UPDATE jobs SET status = ?, finished_at = ?, result_json = ?, error = ? WHERE id = ?",
            (
                "completed" if event.result else "error",
                datetime.now(UTC).isoformat(),
                event.result.model_dump_json() if event.result else None,
                event.error,
                job_id,
            ),
        )
    connection.execute(
        "INSERT INTO remote_events (job_id, sequence, event_json, reported_at) VALUES (?, ?, ?, ?)",
        (job_id, event.sequence, encoded, datetime.now(UTC).isoformat()),
    )


def append_remote_event(path: Path, job_id: str, event: RemoteEvent) -> Job:
    """Persist an ordered event once, including errors before Slurm acceptance."""
    with closing(_connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        _append_remote_event(connection, job_id, event)
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _job_from_row(row)


def record_ssh_snapshot(
    path: Path, request: RemoteRequest, snapshot: RemoteSnapshot
) -> Job:
    """Commit logs, their byte offsets, and state transitions in one transaction."""
    with closing(_connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (request.job_id,)
        ).fetchone()
        cursor = connection.execute(
            "SELECT * FROM ssh_jobs WHERE job_id = ?", (request.job_id,)
        ).fetchone()
        if row is None or cursor is None:
            raise ValueError("Unknown SSH job")
        connection.execute(
            "UPDATE ssh_jobs SET report_pending = ? WHERE job_id = ?",
            (snapshot.report_pending, request.job_id),
        )
        if snapshot.wandb_url:
            connection.execute(
                "UPDATE jobs SET wandb_run_id = ?, wandb_url = ? WHERE id = ?",
                (request.job_id, snapshot.wandb_url, request.job_id),
            )
        if (
            cursor["setup_offset"] != request.setup_offset
            or cursor["log_offset"] != request.log_offset
        ):
            raise ValueError("Stale remote log offsets")
        if (
            snapshot.setup_offset < request.setup_offset
            or snapshot.log_offset < request.log_offset
        ):
            raise ValueError("Remote logs moved backwards")
        if row["status"] in ("completed", "error"):
            updated = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (request.job_id,)
            ).fetchone()
            return _job_from_row(updated)
        if snapshot.slurm_job_id:
            if row["slurm_job_id"] not in (None, snapshot.slurm_job_id):
                raise ValueError("Slurm job ID changed")
            connection.execute(
                "UPDATE jobs SET slurm_job_id = ?, status = CASE WHEN status = 'running' "
                "THEN status ELSE 'queued' END WHERE id = ?",
                (snapshot.slurm_job_id, request.job_id),
            )

        def event(kind: RemoteEventKind, **payload) -> None:
            _append_remote_event(
                connection,
                request.job_id,
                RemoteEvent(
                    sequence=_next_sequence(connection, request.job_id),
                    kind=kind,
                    slurm_job_id=snapshot.slurm_job_id,
                    **payload,
                ),
            )

        if (
            row["status"] != "running"
            and snapshot.slurm_job_id
            and (
                snapshot.slurm_state in ("RUNNING", "COMPLETING")
                or snapshot.result is not None
            )
        ):
            event(RemoteEventKind.STARTED)
        for text in (snapshot.setup_log, snapshot.slurm_log):
            if text:
                event(RemoteEventKind.LOG, line=text)
        connection.execute(
            "UPDATE ssh_jobs SET setup_offset = ?, log_offset = ? WHERE job_id = ?",
            (snapshot.setup_offset, snapshot.log_offset, request.job_id),
        )
        if not snapshot.logs_remaining:
            if snapshot.error:
                event(RemoteEventKind.FAILED, error=snapshot.error)
            elif snapshot.result is not None:
                event(RemoteEventKind.COMPLETED, result=snapshot.result)
        updated = connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (request.job_id,)
        ).fetchone()
    return _job_from_row(updated)


def next_unreported_event(path: Path) -> tuple[Job, RemoteEvent] | None:
    """Return the next durable event in per-job order for W&B publication."""

    with closing(_connect(path)) as connection:
        row = connection.execute(
            """
            SELECT e.job_id, e.event_json FROM remote_events e
            LEFT JOIN remote_report_cursor c ON c.job_id = e.job_id
            WHERE e.sequence = COALESCE(c.last_sequence, 0) + 1
            ORDER BY e.reported_at, e.job_id LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    job = get_job(path, row["job_id"])
    assert job is not None
    return job, RemoteEvent.model_validate_json(row["event_json"])


def mark_remote_event_reported(path: Path, job_id: str, sequence: int) -> None:
    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO remote_report_cursor (job_id, last_sequence) VALUES (?, ?)
            ON CONFLICT(job_id) DO UPDATE SET last_sequence = excluded.last_sequence
            WHERE remote_report_cursor.last_sequence = excluded.last_sequence - 1
            """,
            (job_id, sequence),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Remote report cursor is out of order")


def _finish_job(
    path: Path,
    job_id: str,
    *,
    status: JobStatus,
    result_json: str | None,
    error: str | None,
) -> Job:
    finished_at = datetime.now(UTC).isoformat()
    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = ?, finished_at = ?, result_json = ?, error = ?
            WHERE id = ? AND status = ?
            """,
            (
                status.value,
                finished_at,
                result_json,
                error,
                job_id,
                JobStatus.RUNNING.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Job {job_id!r} is not running")

    job = get_job(path, job_id)
    if job is None:
        raise RuntimeError(f"Job {job_id!r} disappeared from the database")
    return job


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _job_from_row(row: sqlite3.Row) -> Job:
    result = None
    if row["result_json"] is not None:
        result = JudgeResult.model_validate(json.loads(row["result_json"]))

    return Job(
        id=row["id"],
        submission=Submission(
            repo_url=row["repo_url"],
            commit_sha=row["commit_sha"],
            task_id=row["task_id"],
            github_actor=row["github_actor"],
        ),
        status=JobStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        result=result,
        error=row["error"],
        wandb_run_id=row["wandb_run_id"],
        wandb_url=row["wandb_url"],
        execution_backend=row["execution_backend"],
        slurm_job_id=row["slurm_job_id"],
    )
