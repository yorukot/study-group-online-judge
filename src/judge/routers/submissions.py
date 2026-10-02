import re
from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from judge import database
from judge.agent_channel import AgentUnavailable, OfferAlreadyPending
from judge.models import (
    Job,
    JobOffer,
    JobStatus,
    SubJudge,
    SubJudgeBackend,
    Submission,
)
from judge.tasks import TASKS

bearer = HTTPBearer(auto_error=False)
REQUEST_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def require_api_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    """Require the shared submission token configured on the judge."""

    expected_token = request.app.state.api_token
    if credentials is None or not compare_digest(
        credentials.credentials, expected_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter(
    tags=["Submissions"],
    dependencies=[Depends(require_api_token)],
)


@router.post("/submissions", response_model=Job, status_code=status.HTTP_201_CREATED)
async def submit(
    submission: Submission,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> Job:
    """Queue locally or wait for a sub-judge to submit a GPU job to Slurm."""

    task = TASKS.get(submission.task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown task: {submission.task_id}",
        )

    database_path = request.app.state.database_path
    if task.resources.gpus == 0:
        return database.create_job(database_path, submission)

    if (
        idempotency_key is None
        or REQUEST_KEY_PATTERN.fullmatch(idempotency_key) is None
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="GPU submissions require a valid Idempotency-Key header",
        )

    previous = database.get_job_by_request_key(
        database_path, submission.repo_url, idempotency_key
    )
    if previous is not None:
        if previous.submission != submission:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Idempotency-Key was already used for another submission",
            )
        job = previous
    else:
        database.prune_stale_sub_judges(database_path)
        available = await request.app.state.agent_channel.available_ids()
        judge = next(
            (
                candidate
                for candidate in database.list_sub_judges(database_path)
                if _can_run(
                    candidate,
                    task.id,
                    task.resources.gpus,
                    available,
                    request.app.state.judge_revision,
                )
            ),
            None,
        )
        if judge is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No available sub-judge for this GPU task",
            )
        try:
            job = database.create_remote_job(
                database_path,
                submission,
                judge_id=judge.id,
                request_key=idempotency_key,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error

    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED) or (
        job.status == JobStatus.ERROR and job.slurm_job_id is not None
    ):
        return job
    if job.status == JobStatus.ERROR:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Scheduling previously failed: {job.error}",
        )

    assert job.assigned_judge_id is not None
    offer = JobOffer(
        job_id=job.id,
        submission=job.submission,
        resources=task.resources,
    )
    try:
        receipt = await request.app.state.agent_channel.offer(
            job.assigned_judge_id, offer
        )
    except (AgentUnavailable, OfferAlreadyPending) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Sub-judge did not accept job {job.id}: {error}",
        ) from error

    if not receipt.accepted or receipt.slurm_job_id is None:
        reason = receipt.error or "Sub-judge did not return a Slurm job ID"
        database.fail_remote_dispatch(database_path, job.id, reason)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Sub-judge rejected job {job.id}: {reason}",
        )

    return database.mark_remote_job_queued(database_path, job.id, receipt.slurm_job_id)


def _can_run(
    judge: SubJudge,
    task_id: str,
    gpus: int,
    available: set[str],
    expected_revision: str | None,
) -> bool:
    return (
        judge.id in available
        and judge.backend == SubJudgeBackend.SLURM
        and task_id in judge.task_ids
        and judge.max_gpus >= gpus
        and (expected_revision is None or judge.judge_revision == expected_revision)
    )


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str, request: Request) -> Job:
    """Return the current state of a submission job."""

    job = database.get_job(request.app.state.database_path, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return job
