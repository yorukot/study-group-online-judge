import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

import wandb

from judge.database import (
    claim_next_job,
    complete_job,
    fail_job,
    migrate_database,
    set_wandb_run,
)
from judge.executor import DockerExecutor
from judge.models import Job, JobStatus, JudgeResult
from judge.repository import checkout_repository, create_job_workspace
from judge.tasks import TASKS


def run_job(
    job: Job,
    *,
    database_path: Path,
    work_root: Path,
    executor: DockerExecutor,
    on_output: Callable[[str], None],
    wandb_project: str,
    wandb_entity: str | None = None,
) -> Job:
    """Check out, execute, and persist one previously claimed job."""

    run: wandb.Run | None = None
    try:
        task = TASKS[job.submission.task_id]
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=job.id,
            resume="allow",
            name=f"{task.id}-{job.submission.github_actor}-{job.id[:8]}",
            job_type="submission",
            config={
                "job_id": job.id,
                "github_actor": job.submission.github_actor,
                "task_id": job.submission.task_id,
                "repo_url": job.submission.repo_url,
                "commit_sha": job.submission.commit_sha,
                "resources": task.resources.model_dump(),
            },
            save_code=False,
        )
        run.summary["judge_status"] = JobStatus.RUNNING.value
        job = set_wandb_run(
            database_path,
            job.id,
            run_id=run.id,
            url=run.url,
        )
        on_output(f"[judge] starting job {job.id}\n")

        workspace = create_job_workspace(work_root, job.id)
        submission = checkout_repository(
            job.submission.repo_url,
            job.submission.commit_sha,
            workspace / "submission",
        )
        output_directory = workspace / "output"
        execution = executor.run(
            task_id=task.id,
            resources=task.resources,
            submission=submission,
            output_directory=output_directory,
            on_output=on_output,
        )
        if execution.returncode != 0:
            raise RuntimeError(f"Task runner exited with status {execution.returncode}")

        result_path = output_directory / "result.json"
        result = JudgeResult.model_validate_json(result_path.read_text())
        if result.passed is not None:
            on_output(f"[judge] verdict: {'PASS' if result.passed else 'FAIL'}\n")
        for test in result.tests:
            if not test.passed:
                on_output(
                    f"[judge] failed {test.name}: "
                    f"{test.message or 'no reason provided'}\n"
                )
        _publish_result(run, job, result, result_path)
        completed = complete_job(database_path, job.id, result)
        on_output(f"[judge] completed job {job.id}\n")
        return completed
    except Exception as error:  # noqa: BLE001 - a worker must persist job failures
        message = f"{type(error).__name__}: {error}"
        if run is not None:
            try:
                run.summary["judge_status"] = JobStatus.ERROR.value
                run.summary["error"] = message
            except Exception as reporting_error:  # noqa: BLE001
                on_output(
                    f"[judge] failed to update W&B run for {job.id}: "
                    f"{type(reporting_error).__name__}: {reporting_error}\n"
                )
        failed = fail_job(database_path, job.id, message)
        on_output(f"[judge] failed job {job.id}: {message}\n")
        return failed
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as error:  # noqa: BLE001 - reporting must not alter state
                on_output(
                    f"[judge] failed to finish W&B run for {job.id}: "
                    f"{type(error).__name__}: {error}\n"
                )


def _publish_result(
    run: wandb.Run,
    job: Job,
    result: JudgeResult,
    result_path: Path,
) -> None:
    metrics = dict(result.metrics)
    if result.score is not None:
        metrics["score"] = result.score
    if result.passed is not None:
        metrics["passed"] = result.passed
    if metrics:
        run.log(metrics)

    if result.tests:
        table = wandb.Table(
            columns=["test", "passed", "message"],
            data=[[test.name, test.passed, test.message] for test in result.tests],
        )
        run.log({"tests": table})

    artifact = wandb.Artifact(name=f"judge-result-{job.id}", type="judge-result")
    artifact.add_file(str(result_path), name="result.json")
    run.log_artifact(artifact)

    for name, value in result.metrics.items():
        run.summary[name] = value
    if result.passed is not None:
        run.summary["passed"] = result.passed
    if result.score is not None:
        run.summary["score"] = result.score
    run.summary["judge_status"] = JobStatus.COMPLETED.value


def run_worker(
    *,
    database_path: Path,
    work_root: Path,
    runner_image: str,
    wandb_project: str,
    wandb_entity: str | None = None,
    hf_cache_volume: str | None = None,
    uv_cache_volume: str | None = None,
    poll_interval_seconds: float = 1,
    once: bool = False,
) -> Job | None:
    """Poll SQLite and process jobs one at a time."""

    migrate_database(database_path)
    executor = DockerExecutor(
        runner_image,
        hf_cache_volume=hf_cache_volume,
        uv_cache_volume=uv_cache_volume,
    )

    while True:
        job = claim_next_job(database_path)
        if job is None:
            if once:
                return None
            time.sleep(poll_interval_seconds)
            continue

        completed = run_job(
            job,
            database_path=database_path,
            work_root=work_root,
            executor=executor,
            on_output=lambda line: print(line, end="", flush=True),
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
        if once:
            return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()

    runner_image = os.environ.get("JUDGE_RUNNER_IMAGE")
    if not runner_image:
        raise RuntimeError("JUDGE_RUNNER_IMAGE must be configured")
    wandb_project = os.environ.get("WANDB_PROJECT")
    if not wandb_project:
        raise RuntimeError("WANDB_PROJECT must be configured")

    run_worker(
        database_path=Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db")),
        work_root=Path(os.environ.get("JUDGE_WORK_ROOT", "work")),
        runner_image=runner_image,
        wandb_project=wandb_project,
        wandb_entity=os.environ.get("WANDB_ENTITY") or None,
        hf_cache_volume=os.environ.get("JUDGE_HF_CACHE_VOLUME") or None,
        uv_cache_volume=os.environ.get("JUDGE_UV_CACHE_VOLUME") or None,
        once=arguments.once,
    )


if __name__ == "__main__":
    main()
