import re
from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from judge import database
from judge.execution_logging import log_event
from judge.models import Job, Submission
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
    """Persist a CPU or GPU job without waiting for remote preparation."""

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

    config = request.app.state.remote_config
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Remote GPU judge is not configured",
        )
    try:
        job = database.create_remote_job(
            database_path,
            submission,
            config=config,
            resources=task.resources,
            request_key=idempotency_key,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error
    log_event(
        "submission.received",
        job_id=job.id,
        task_id=submission.task_id,
        repo_url=submission.repo_url,
        commit_sha=submission.commit_sha,
        host=config.host,
        status=job.status.value,
    )
    return job


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
