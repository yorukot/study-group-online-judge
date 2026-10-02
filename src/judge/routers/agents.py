from datetime import UTC, datetime
from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from judge.agent_channel import AgentPollConflict, UnknownOffer
from judge.database import (
    append_remote_event,
    get_sub_judge,
    heartbeat_sub_judge,
    register_sub_judge,
)
from judge.models import (
    Job,
    JobOffer,
    JobReceipt,
    RemoteEvent,
    SubJudge,
    SubJudgeRegistration,
)
from judge.tasks import TASKS

bearer = HTTPBearer(auto_error=False)


def require_agent_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    expected = request.app.state.agent_token
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sub-judge registration is not configured",
        )
    if credentials is None or not compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid sub-judge token",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter(
    prefix="/agents",
    tags=["Sub-judges"],
    dependencies=[Depends(require_agent_token)],
)


@router.post("/register", response_model=SubJudge)
def register(registration: SubJudgeRegistration, request: Request) -> SubJudge:
    unknown = sorted(set(registration.task_ids) - TASKS.keys())
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown tasks: {', '.join(unknown)}",
        )

    sub_judge = SubJudge(
        **registration.model_dump(),
        registered_at=datetime.now(UTC),
    )
    return register_sub_judge(request.app.state.database_path, sub_judge)


@router.get("/{judge_id}/next", response_model=JobOffer)
async def next_job(judge_id: str, request: Request) -> JobOffer | Response:
    if get_sub_judge(request.app.state.database_path, judge_id) is None:
        raise HTTPException(status_code=404, detail="Sub-judge is not registered")

    try:
        offer = await request.app.state.agent_channel.wait_for_offer(judge_id)
    except AgentPollConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if offer is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return offer


@router.post("/{judge_id}/heartbeat", status_code=204)
def heartbeat(judge_id: str, request: Request) -> Response:
    if not heartbeat_sub_judge(request.app.state.database_path, judge_id):
        raise HTTPException(status_code=404, detail="Sub-judge is not registered")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{judge_id}/jobs/{job_id}/receipt", status_code=204)
async def acknowledge(
    judge_id: str, job_id: str, receipt: JobReceipt, request: Request
) -> Response:
    try:
        await request.app.state.agent_channel.acknowledge(judge_id, job_id, receipt)
    except UnknownOffer as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{judge_id}/jobs/{job_id}/events", response_model=Job)
def report_event(
    judge_id: str, job_id: str, event: RemoteEvent, request: Request
) -> Job:
    try:
        return append_remote_event(
            request.app.state.database_path, judge_id, job_id, event
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
