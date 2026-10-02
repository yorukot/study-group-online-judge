"""Public leaderboard snapshots; credentials and source access stay on the server."""

import json
import logging
import math
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import NotRequired, TypedDict
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from judge.tasks import TASKS

# Inherit Uvicorn's configured handlers and level so INFO reaches container logs.
logger = logging.getLogger("uvicorn.error.judge.leaderboard")


def safe_traceback(error: Exception) -> str:
    detail = "".join(traceback.format_exception(error))
    for name in (
        "WANDB_API_KEY",
        "JUDGE_API_TOKEN",
        "JUDGE_AGENT_TOKEN",
        "JUDGE_SLACK_WEBHOOK_URL",
    ):
        secret = os.getenv(name)
        if secret:
            detail = detail.replace(secret, "[REDACTED]")
    return detail


class SubmissionEntry(TypedDict):
    github_actor: str
    score: float | None
    submitted_at: str
    run_url: str | None
    submission_id: str


class RankedEntry(SubmissionEntry):
    rank: int
    attempts: int


class Leaderboard(TypedDict):
    id: str
    grading_type: str
    primary_metric: str | None
    metric_direction: str | None
    entries: list[RankedEntry]
    submissions: int
    participants: int


class LeaderboardSnapshot(TypedDict):
    source: str
    updated_at: str | None
    error: str | None
    labs: list[Leaderboard]
    notification_error: NotRequired[str]


def timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def rank_submissions(submissions) -> list[Leaderboard]:
    boards: list[Leaderboard] = []
    for task in TASKS.values():
        metadata = task.metadata()
        scored = metadata.grading_type == "score"
        best: dict[str, tuple[tuple[float, datetime, str], SubmissionEntry]] = {}
        attempts: dict[str, int] = {}
        for item in submissions:
            if item["task_id"] != task.id:
                continue
            actor = item["github_actor"]
            key = actor.casefold()
            attempts[key] = attempts.get(key, 0) + 1
            if item.get("passed") is False or (
                not scored and item.get("passed") is not True
            ):
                continue
            value: float | None = None
            priority: float = 0
            if scored:
                metric = item["metrics"].get(task.primary_metric)
                if (
                    isinstance(metric, bool)
                    or not isinstance(metric, (int, float))
                    or not math.isfinite(metric)
                ):
                    continue
                value = metric
                priority = -value if metadata.metric_direction == "maximize" else value
            order = (priority, timestamp(item["submitted_at"]), item["id"])
            if key not in best or order < best[key][0]:
                best[key] = (
                    order,
                    {
                        "github_actor": actor,
                        "score": value,
                        "submitted_at": timestamp(item["submitted_at"]).isoformat(),
                        "run_url": item.get("run_url"),
                        "submission_id": item["id"],
                    },
                )
        entries: list[RankedEntry] = []
        for rank, (_, entry) in enumerate(
            sorted(best.values(), key=lambda pair: pair[0]), 1
        ):
            entries.append(
                {
                    **entry,
                    "rank": rank,
                    "attempts": attempts[entry["github_actor"].casefold()],
                }
            )
        boards.append(
            {
                "id": metadata.id,
                "grading_type": metadata.grading_type.value,
                "primary_metric": metadata.primary_metric,
                "metric_direction": (
                    metadata.metric_direction.value
                    if metadata.metric_direction
                    else None
                ),
                "entries": entries,
                "submissions": sum(attempts.values()),
                "participants": len(attempts),
            }
        )
    return boards


def wandb_submissions():
    import wandb

    entity = os.getenv("WANDB_ENTITY") or "cerulean-labs"
    project = os.getenv("WANDB_PROJECT") or "study-group-labs"
    api = wandb.Api(timeout=20)
    runs = api.runs(
        f"{entity}/{project}", filters={"summary_metrics.judge_status": "completed"}
    )
    result = []
    scanned = skipped_metadata = skipped_timestamp = 0
    for run in runs:
        scanned += 1
        config, summary = run.config, dict(run.summary)
        if (
            config.get("task_id") not in TASKS
            or not isinstance(config.get("github_actor"), str)
            or not config["github_actor"].strip()
        ):
            skipped_metadata += 1
            continue
        submitted = config.get("submitted_at") or run.created_at
        try:
            timestamp(submitted)
        except ValueError, TypeError:
            skipped_timestamp += 1
            continue
        result.append(
            {
                "id": run.id,
                "task_id": config["task_id"],
                "github_actor": config["github_actor"],
                "submitted_at": submitted,
                "passed": summary.get("passed"),
                "metrics": summary,
                "run_url": run.url,
            }
        )
    logger.info(
        "W&B runs fetched project=%s/%s scanned=%d accepted=%d skipped_metadata=%d skipped_timestamp=%d",
        entity,
        project,
        scanned,
        len(result),
        skipped_metadata,
        skipped_timestamp,
    )
    return result


def new_record(board, previous):
    """Compare with the best acknowledged result, not the previous poll."""
    if not board["entries"]:
        return False
    if not previous:
        return True
    leader, old = board["entries"][0], previous[0]
    if board["grading_type"] == "pass_fail":
        return timestamp(leader["submitted_at"]) < timestamp(old["submitted_at"])
    if board["metric_direction"] == "minimize":
        return leader["score"] < old["score"]
    return leader["score"] > old["score"]


def post_slack(webhook, boards, changed):
    names_path = Path(
        os.getenv(
            "JUDGE_LAB_NAMES_PATH",
            str(Path(__file__).resolve().parents[2] / "ui/src/lab-names.json"),
        )
    )
    try:
        names = json.loads(names_path.read_text())
    except OSError, ValueError:
        names = {}

    def escape(value):
        return (
            str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    sections = []
    for board in boards:
        if board["id"] not in changed or not board["entries"]:
            continue
        label = board["id"].replace("lab", "Lab ", 1)
        name = names.get(board["id"])
        title = f"{label} ({name})" if name else label
        leader = board["entries"][0]
        if board["grading_type"] == "score":
            direction = (
                "↓ Lower is better"
                if board["metric_direction"] == "minimize"
                else "↑ Higher is better"
            )
            rule = f"{escape(board['primary_metric'])} · {direction}"
            result = f"{leader['score']:.6g}"
        else:
            rule = "Pass / fail · Earliest passing submission"
            result = "Passed"
        sections.append(
            f"🧪 *{escape(title)}*\n\n{rule}\n🏅 {escape(leader['github_actor'])} — {result}"
        )
    text = "*OJ Leaderboard update*\n\n" + "\n\n".join(sections)
    payload = {
        "text": text,
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}],
    }
    workflow = urlsplit(webhook).path.startswith(("/triggers/", "/workflows/"))
    if workflow:
        payload = {"text": text}
    request = Request(
        webhook,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "study-group-online-judge/0.1",
        },
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        body = response.read().strip()
        acknowledged = body == b"ok"
        if workflow and not acknowledged:
            try:
                acknowledged = json.loads(body).get("ok") is True
            except ValueError, AttributeError:
                acknowledged = False
        if not 200 <= response.status < 300 or not acknowledged:
            raise RuntimeError("Slack did not acknowledge notification")


class LeaderboardService:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.source = "wandb"
        self.state_path = database_path.with_suffix(".leaderboard.json")
        self.snapshot: LeaderboardSnapshot = {
            "source": self.source,
            "updated_at": None,
            "error": None,
            "labs": rank_submissions([]),
        }
        self.notified = None
        if self.state_path.exists():
            try:
                saved = json.loads(self.state_path.read_text())
                if saved["snapshot"]["source"] == self.source:
                    self.snapshot = saved["snapshot"]
                    self.notified = saved["notified"]
            except ValueError, KeyError:
                pass

    def refresh(self):
        # A single API worker owns polling. HTTP reads only access the cached snapshot.
        started = monotonic()
        project = f"{os.getenv('WANDB_ENTITY') or 'cerulean-labs'}/{os.getenv('WANDB_PROJECT') or 'study-group-labs'}"
        stage = "fetch"
        logger.info("W&B refresh started project=%s", project)
        try:
            submissions = wandb_submissions()
            stage = "rank"
            boards = rank_submissions(submissions)
        except Exception as error:  # noqa: BLE001 - retain last good standings
            logger.error(
                "W&B refresh failed project=%s stage=%s elapsed=%.2fs last_success=%s\n%s",
                project,
                stage,
                monotonic() - started,
                self.snapshot["updated_at"],
                safe_traceback(error),
            )
            self.snapshot = {
                **self.snapshot,
                "error": "Source refresh failed. Showing the last successful snapshot.",
            }
            return
        logger.info(
            "W&B refresh succeeded project=%s elapsed=%.2fs submissions=%d ranked=%s",
            project,
            monotonic() - started,
            len(submissions),
            {board["id"]: len(board["entries"]) for board in boards},
        )
        self.snapshot = {
            "source": self.source,
            "updated_at": datetime.now(UTC).isoformat(),
            "error": None,
            "labs": boards,
        }
        current = {
            board["id"]: [
                {
                    key: value
                    for key, value in entry.items()
                    if key not in {"attempts", "run_url"}
                }
                for entry in board["entries"]
            ]
            for board in boards
        }
        webhook = os.getenv("JUDGE_SLACK_WEBHOOK_URL")
        changed = [
            board["id"]
            for board in boards
            if self.notified is not None
            and new_record(board, self.notified.get(board["id"]))
        ]
        if self.notified is None:
            self.notified = current  # Initial sync seeds the all-time best baseline.
        elif changed:
            try:
                if webhook:
                    post_slack(webhook, boards, changed)
            except Exception:  # noqa: BLE001 - retain the record for delivery retry
                self.snapshot["notification_error"] = (
                    "Slack delivery failed; retrying on the next refresh."
                )
            else:
                for task_id in changed:
                    self.notified[task_id] = current[task_id]
        # Ties, lower-ranked changes, and deleted runs never lower the record.
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"snapshot": self.snapshot, "notified": self.notified})
        )
        temporary.replace(self.state_path)
