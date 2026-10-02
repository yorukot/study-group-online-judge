import re
from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobStatus(StrEnum):
    """Lifecycle state for a queued submission."""

    DISPATCHING = "dispatching"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class MetricDirection(StrEnum):
    """How a task's primary metric should be ranked."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class GradingType(StrEnum):
    PASS_FAIL = "pass_fail"
    SCORE = "score"


class TaskMetadata(BaseModel):
    """Public ranking rules for a registered lab."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    grading_type: GradingType
    primary_metric: str | None = Field(default=None, min_length=1)
    metric_direction: MetricDirection | None = None

    @model_validator(mode="after")
    def validate_ranking(self) -> Self:
        if self.grading_type == GradingType.SCORE:
            if self.primary_metric is None or self.metric_direction is None:
                raise ValueError(
                    "score tasks require a primary_metric and metric_direction"
                )
        elif self.primary_metric is not None or self.metric_direction is not None:
            raise ValueError(
                "pass/fail tasks rank by passing time without a score metric"
            )
        return self


class SubJudgeBackend(StrEnum):
    DOCKER = "docker"
    SLURM = "slurm"


class RemoteEventKind(StrEnum):
    STARTED = "started"
    LOG = "log"
    COMPLETED = "completed"
    FAILED = "failed"


class Submission(BaseModel):
    """An immutable repository revision submitted for one task."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str = Field(min_length=1)
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    task_id: str = Field(min_length=1)
    github_actor: str = Field(min_length=1)

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, value: str) -> str:
        pattern = r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
        if re.fullmatch(pattern, value) is None:
            raise ValueError("repo_url must be an HTTPS GitHub repository URL")
        return value


class TestResult(BaseModel):
    """The outcome of one named correctness check."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    passed: bool
    message: str | None = None


class JudgeResult(BaseModel):
    """A common result shape for pass/fail and scored tasks."""

    model_config = ConfigDict(extra="forbid")

    passed: bool | None = None
    score: float | None = Field(default=None, allow_inf_nan=False)
    metrics: dict[str, float] = Field(default_factory=dict)
    tests: list[TestResult] = Field(default_factory=list)


class Resources(BaseModel):
    """Resource limits requested by a task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus: int = Field(default=2, ge=1)
    memory_gb: int = Field(default=8, ge=1)
    gpus: int = Field(default=0, ge=0)
    timeout_seconds: int = Field(default=300, ge=1)


class SubJudgeRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,62}$")
    backend: SubJudgeBackend
    task_ids: list[str] = Field(min_length=1)
    max_gpus: int = Field(ge=0)
    judge_revision: str = Field(min_length=1)

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, task_ids: list[str]) -> list[str]:
        if any(not task_id for task_id in task_ids):
            raise ValueError("task IDs must not be empty")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task IDs must be unique")
        return task_ids


class SubJudge(SubJudgeRegistration):
    registered_at: datetime


class JobOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    submission: Submission
    resources: Resources


class JobReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    slurm_job_id: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.accepted:
            if self.error is not None or self.slurm_job_id == "":
                raise ValueError("accepted receipts must not contain an error")
        elif not self.error or self.slurm_job_id is not None:
            raise ValueError("rejected receipts require an error and no Slurm job ID")
        return self


class RemoteEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    kind: RemoteEventKind
    slurm_job_id: str = Field(min_length=1)
    line: str | None = Field(default=None, max_length=8192)
    result: JudgeResult | None = None
    error: str | None = Field(default=None, max_length=8192)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        required = {
            RemoteEventKind.STARTED: (False, False, False),
            RemoteEventKind.LOG: (True, False, False),
            RemoteEventKind.COMPLETED: (False, True, False),
            RemoteEventKind.FAILED: (False, False, True),
        }[self.kind]
        present = (
            self.line is not None,
            self.result is not None,
            self.error is not None,
        )
        if present != required:
            raise ValueError(f"Invalid payload for {self.kind.value} event")
        return self


class Job(BaseModel):
    """A submission and its persistent judge state."""

    model_config = ConfigDict(extra="forbid")

    id: str
    submission: Submission
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: JudgeResult | None = None
    error: str | None = None
    wandb_run_id: str | None = None
    wandb_url: str | None = None
    assigned_judge_id: str | None = None
    slurm_job_id: str | None = None
