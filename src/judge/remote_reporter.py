"""Publish durable remote judge events to W&B from the master host."""

import os
import tempfile
import time
from pathlib import Path

import wandb

from judge.database import (
    mark_remote_event_reported,
    migrate_database,
    next_unreported_event,
    set_wandb_run,
)
from judge.models import Job, JobStatus, RemoteEvent, RemoteEventKind
from judge.tasks import TASKS
from judge.worker import _publish_result


def publish_event(
    job: Job,
    event: RemoteEvent,
    *,
    database_path: Path,
    wandb_project: str,
    wandb_entity: str | None,
    active_runs: dict[str, wandb.Run],
) -> None:
    run = active_runs.get(job.id)
    if run is None:
        task = TASKS[job.submission.task_id]
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=job.id,
            resume="allow",
            reinit="create_new",
            name=f"{task.id}-{job.submission.github_actor}-{job.id[:8]}",
            job_type="submission",
            config={
                "job_id": job.id,
                "github_actor": job.submission.github_actor,
                "task_id": job.submission.task_id,
                "repo_url": job.submission.repo_url,
                "commit_sha": job.submission.commit_sha,
                "resources": task.resources.model_dump(),
                "sub_judge": job.assigned_judge_id,
                "slurm_job_id": job.slurm_job_id,
            },
            save_code=False,
        )
        active_runs[job.id] = run
        run.summary["judge_status"] = JobStatus.RUNNING.value
    if job.wandb_run_id is None:
        set_wandb_run(database_path, job.id, run_id=run.id, url=run.url)
    if event.kind == RemoteEventKind.STARTED:
        print(f"[judge] started Slurm job {event.slurm_job_id}", flush=True)
    elif event.kind == RemoteEventKind.LOG:
        assert event.line is not None
        print(event.line, end="" if event.line.endswith("\n") else "\n", flush=True)
    elif event.kind == RemoteEventKind.COMPLETED:
        assert event.result is not None
        with tempfile.TemporaryDirectory(prefix="judge-result-") as directory:
            result_path = Path(directory) / "result.json"
            result_path.write_text(event.result.model_dump_json(indent=2))
            _publish_result(run, job, event.result, result_path)
        if event.result.passed is not None:
            print(
                f"[judge] verdict: {'PASS' if event.result.passed else 'FAIL'}",
                flush=True,
            )
        for test in event.result.tests:
            if not test.passed:
                print(
                    f"[judge] failed {test.name}: "
                    f"{test.message or 'no reason provided'}",
                    flush=True,
                )
    else:
        assert event.error is not None
        run.summary["judge_status"] = JobStatus.ERROR.value
        run.summary["error"] = event.error
        print(f"[judge] failed job {job.id}: {event.error}", flush=True)
    if event.kind in {RemoteEventKind.COMPLETED, RemoteEventKind.FAILED}:
        run.finish(exit_code=1 if event.kind == RemoteEventKind.FAILED else 0)
        del active_runs[job.id]


def run_reporter(
    *,
    database_path: Path,
    wandb_project: str,
    wandb_entity: str | None = None,
    once: bool = False,
    poll_interval_seconds: float = 2,
) -> None:
    migrate_database(database_path)
    active_runs: dict[str, wandb.Run] = {}
    while True:
        pending = next_unreported_event(database_path)
        if pending is None:
            if once:
                return
            time.sleep(poll_interval_seconds)
            continue
        job, event = pending
        try:
            publish_event(
                job,
                event,
                database_path=database_path,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                active_runs=active_runs,
            )
            mark_remote_event_reported(database_path, job.id, event.sequence)
        except Exception as error:  # Retry durable event later.
            print(
                f"[judge] W&B report retry for {job.id}/{event.sequence}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            if once:
                raise
            time.sleep(poll_interval_seconds)


def main() -> None:
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        raise RuntimeError("WANDB_PROJECT must be configured")
    run_reporter(
        database_path=Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db")),
        wandb_project=project,
        wandb_entity=os.environ.get("WANDB_ENTITY") or None,
    )


if __name__ == "__main__":
    main()
