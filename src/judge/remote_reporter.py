"""Publish GPU logs and results on the remote login node using its .netrc."""

import netrc
from datetime import UTC, datetime
from pathlib import Path

import wandb

from judge.models import Job, JobStatus
from judge.ssh import RemoteRequest, RemoteSnapshot
from judge.worker import _publish_result


def publish_report(
    request: RemoteRequest,
    snapshot: RemoteSnapshot,
    workspace: Path,
    record: dict,
) -> None:
    """Flush progress without finishing the server run until judging completes.

    The caller holds the remote job lock and durably saves the updated record.
    A failed upload can replay its last batch, but never schedules another job.
    """
    progress = dict(record.get("report", {}))
    snapshot.wandb_url = progress.get("url")
    terminal = snapshot.error is not None or snapshot.result is not None
    if progress.get("complete"):
        return
    state = snapshot.slurm_state or record["state"]
    chunks = []
    for name in ("setup", "slurm"):
        path = workspace / "output" / f"{name}.log"
        offset = progress.get(f"{name}_offset", 0)
        if path.is_file():
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read(65536)
                remaining = bool(stream.read(1))
            chunks.append((name, chunk.decode(errors="replace"), offset + len(chunk)))
            snapshot.report_pending |= remaining
    # Use only the work-directory credential, without prompting or writing keys.
    credentials = netrc.netrc(str(Path(request.config.work_root) / ".netrc"))
    authentication = credentials.authenticators("api.wandb.ai")
    if authentication is None or not authentication[2]:
        raise ValueError("Remote .netrc must contain credentials for api.wandb.ai")
    directory = workspace / "output" / "wandb"
    directory.mkdir(parents=True, exist_ok=True)

    def initialize(*, finalize: bool = False):
        # Each SSH helper exits after its poll. Keep the server run open while
        # flushing this process's SDK session, including its implicit atexit.
        # Resume even idle polls so queued/running jobs continue heartbeating.
        return wandb.init(
            project=request.config.wandb_project,
            entity=request.config.wandb_entity,
            id=request.job_id,
            resume="allow",
            reinit="create_new",
            name=f"{request.submission.task_id}-{request.submission.github_actor}-{request.job_id[:8]}",
            job_type="submission",
            dir=str(directory),
            config={
                "job_id": request.job_id,
                **request.submission.model_dump(),
                "resources": request.resources.model_dump(),
                "execution_backend": "ssh_slurm",
            },
            settings=wandb.Settings(
                api_key=authentication[2],
                init_timeout=30,
                finish_timeout=30,
                finish_timeout_raises=True,
                mode="online",
                console="wrap",
                x_update_finish_state=finalize,
            ),
            save_code=False,
        )

    run = initialize()
    try:
        run.summary["judge_status"] = JobStatus.RUNNING.value
        snapshot.wandb_url = run.url
        progress["url"] = run.url
        if snapshot.slurm_job_id:
            run.summary["slurm_job_id"] = snapshot.slurm_job_id
        if progress.get("state") != state:
            print(f"[judge] state: {state}", flush=True)
        for name, chunk, offset in chunks:
            if chunk:
                print(chunk, end="" if chunk.endswith("\n") else "\n", flush=True)
            progress[f"{name}_offset"] = offset
        if terminal and not snapshot.report_pending:
            if snapshot.result is not None:
                job = Job(
                    id=request.job_id,
                    submission=request.submission,
                    status=JobStatus.COMPLETED,
                    created_at=datetime.now(UTC),
                )
                _publish_result(
                    run, job, snapshot.result, workspace / "output" / "result.json"
                )
                if snapshot.result.passed is not None:
                    print(
                        f"[judge] verdict: {'PASS' if snapshot.result.passed else 'FAIL'}",
                        flush=True,
                    )
                for test in snapshot.result.tests:
                    if not test.passed:
                        print(f"[judge] failed {test.name}: {test.message}", flush=True)
            else:
                run.summary["judge_status"] = JobStatus.ERROR.value
                run.summary["error"] = snapshot.error
                print(
                    f"[judge] failed job {request.job_id}: {snapshot.error}", flush=True
                )
            progress["complete"] = True
        progress["state"] = state
    finally:
        # This acknowledges uploaded logs but cannot mark the server run done,
        # even if publishing the result raised and needs another attempt.
        run.finish()
    if progress.get("complete"):
        # All logs and the terminal report are acknowledged before authorizing
        # a final-state update. A failure here leaves durable cursors unchanged.
        finalizer = initialize(finalize=True)
        finalizer.finish(exit_code=1 if snapshot.error is not None else 0)
    record["report"] = progress
