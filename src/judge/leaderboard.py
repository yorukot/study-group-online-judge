"""Public leaderboard snapshots; credentials and source access stay on the server."""

import json
import logging
import math
import os
import traceback
from datetime import UTC, datetime, time
from pathlib import Path
from time import monotonic, sleep
from typing import NotRequired, TypedDict
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from judge.leaderboard_images import render_standings
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
        "JUDGE_SLACK_BOT_TOKEN",
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


def lab_names():
    names_path = Path(
        os.getenv(
            "JUDGE_LAB_NAMES_PATH",
            str(Path(__file__).resolve().parents[2] / "ui/src/lab-names.json"),
        )
    )
    try:
        return json.loads(names_path.read_text())
    except OSError, ValueError:
        return {}


def escape(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


MEDALS = ("first_place_medal", "second_place_medal", "third_place_medal")
_next_slack_attempt: dict[str, float] = {}


def open_slack(request, workflow, *, timeout=10):
    """Pace app webhooks at 1/s, workflow triggers at 10/min; honor 429 backoff."""
    webhook = request.full_url
    delay = _next_slack_attempt.get(webhook, 0) - monotonic()
    if delay > 30:
        raise RuntimeError("Slack rate limit backoff; retrying on a later refresh")
    if delay > 0:
        sleep(delay)
    _next_slack_attempt[webhook] = monotonic() + (6 if workflow else 1)
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as error:
        if error.code == 429:
            try:
                delay = float(error.headers.get("Retry-After", "60"))
            except ValueError:
                delay = 60
            _next_slack_attempt[webhook] = monotonic() + max(1, delay)
        raise


def slack_api(method, token, payload):
    request = Request(
        f"https://slack.com/api/{method}",
        data=urlencode(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with open_slack(request, False, timeout=30) as response:
        result = json.load(response)
        if not 200 <= response.status < 300 or result.get("ok") is not True:
            raise RuntimeError(
                f"Slack {method} failed: {result.get('error', 'unacknowledged')}"
            )
        return result


def upload_slack_card(token, channel, path, title, blocks):
    """Upload PNG bytes, then share the file and rankings in one Slack message."""
    content = path.read_bytes()
    ticket = slack_api(
        "files.getUploadURLExternal",
        token,
        {
            "filename": path.name,
            "length": len(content),
            "alt_txt": title,
        },
    )
    request = Request(
        ticket["upload_url"],
        data=content,
        headers={"Content-Type": "image/png"},
        method="POST",
    )
    # The upload URL is temporary; the bot token is only sent to Slack API methods.
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError("Slack PNG upload failed")
            response.read()
    except HTTPError as error:
        # Do not log the temporary upload URL in exception tracebacks.
        raise RuntimeError(f"Slack PNG upload failed (HTTP {error.code})") from None
    slack_api(
        "files.completeUploadExternal",
        token,
        {
            "files": json.dumps([{"id": ticket["file_id"], "title": title}]),
            "channel_id": channel,
            "blocks": json.dumps(blocks),
        },
    )


def post_slack(
    webhook,
    boards,
    changed,
    *,
    daily=False,
    image_directory=None,
    first_page=0,
    on_page_sent=None,
):
    """One lab per message, with complete daily standings paged for Slack limits."""
    names = lab_names()
    workflow = urlsplit(webhook or "").path.startswith(("/triggers/", "/workflows/"))
    bot_token = os.getenv("JUDGE_SLACK_BOT_TOKEN")
    channel = os.getenv("JUDGE_SLACK_CHANNEL_ID")
    if bot_token or channel:
        if not bot_token or not channel:
            raise RuntimeError(
                "Inline Slack images require JUDGE_SLACK_BOT_TOKEN and JUDGE_SLACK_CHANNEL_ID"
            )
        if image_directory is None:
            raise RuntimeError("Inline Slack images require an image directory")
    elif not webhook:
        raise RuntimeError("Slack notification credentials are not configured")
    for board in boards:
        if board["id"] not in changed or (not daily and not board["entries"]):
            continue
        label = board["id"].replace("lab", "Lab ", 1)
        name = names.get(board["id"])
        title = f"{label} ({name})" if name else label
        heading = (
            "Daily Updates"
            if daily
            else f"New record by {escape(board['entries'][0]['github_actor'])}!"
        )
        entries = board["entries"] if daily else board["entries"][:3]
        pages = [
            entries[offset : offset + 20] for offset in range(0, len(entries), 20)
        ] or [[]]
        for page_number, page in enumerate(pages, 1):
            if page_number <= first_page:
                continue
            page_label = f" · {page_number}/{len(pages)}" if len(pages) > 1 else ""
            intro = f":test_tube: {heading}"
            blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": intro}}]
            lines = []
            for entry in page:
                rank = entry["rank"]
                prefix = f":{MEDALS[rank - 1]}:" if rank <= 3 else f"{rank}."
                result = (
                    f"{entry['score']:.7g}" if entry["score"] is not None else "Passed"
                )
                line = f"{prefix} {escape(entry['github_actor'])}: {result}"
                lines.append(line)
            if not page:
                lines.append("No ranked participants yet.")
            standings = f"*{escape(title)}*{page_label}\n" + "\n".join(lines)
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": standings},
                }
            )
            text = intro + "\n\n" + standings
            if bot_token and image_directory is not None:
                filename = render_standings(board, page, image_directory)
                upload_slack_card(
                    bot_token,
                    channel,
                    image_directory / filename,
                    title + page_label,
                    blocks,
                )
                if on_page_sent is not None:
                    on_page_sent(page_number)
                continue
            payload = {"text": text} if workflow else {"text": text, "blocks": blocks}
            request = Request(
                webhook,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "study-group-online-judge/0.1",
                },
                method="POST",
            )
            with open_slack(request, workflow) as response:
                body = response.read().strip()
                acknowledged = body == b"ok"
                if workflow and not acknowledged:
                    try:
                        acknowledged = json.loads(body).get("ok") is True
                    except ValueError, AttributeError:
                        acknowledged = False
                if not 200 <= response.status < 300 or not acknowledged:
                    raise RuntimeError("Slack did not acknowledge notification")
            if on_page_sent is not None:
                on_page_sent(page_number)


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
        self.image_directory = database_path.parent / "leaderboard-images"
        self.daily_time = time.fromisoformat(
            os.getenv("JUDGE_SLACK_DAILY_TIME", "19:00")
        )
        self.timezone = ZoneInfo(os.getenv("JUDGE_SLACK_TIMEZONE", "Asia/Taipei"))
        self.daily_notified = {}
        self.daily_pending = {}
        self.notified = None
        if self.state_path.exists():
            try:
                saved = json.loads(self.state_path.read_text())
                if saved["snapshot"]["source"] == self.source:
                    self.snapshot = saved["snapshot"]
                    self.notified = saved["notified"]
                    self.daily_notified = saved.get("daily_notified", {})
                    self.daily_pending = saved.get("daily_pending", {})
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
        notifications_enabled = webhook or os.getenv("JUDGE_SLACK_BOT_TOKEN")
        changed = [
            board["id"]
            for board in boards
            if self.notified is not None
            and new_record(board, self.notified.get(board["id"]))
        ]
        if self.notified is None:
            self.notified = current  # Initial sync seeds the all-time best baseline.
        elif changed:
            for task_id in changed:
                try:
                    if notifications_enabled:
                        post_slack(
                            webhook,
                            boards,
                            [task_id],
                            image_directory=self.image_directory,
                        )
                except Exception as error:  # noqa: BLE001 - retry delivery
                    self.notification_failed(error)
                else:
                    self.notified[task_id] = current[task_id]
                    self.persist()
        now = datetime.now(self.timezone)
        today = now.date().isoformat()
        if notifications_enabled and now.time() >= self.daily_time:
            for board in boards:
                if self.daily_notified.get(board["id"]) == today:
                    continue
                task_id = board["id"]
                if self.daily_pending.get(task_id, {}).get("date") != today:
                    self.daily_pending[task_id] = {
                        "date": today,
                        "board": board,
                        "next_page": 0,
                    }
                    self.persist()
                pending = self.daily_pending[task_id]

                def page_sent(page_number, pending=pending):
                    pending["next_page"] = page_number
                    self.persist()

                try:
                    post_slack(
                        webhook,
                        [pending["board"]],
                        [task_id],
                        daily=True,
                        image_directory=self.image_directory,
                        first_page=pending["next_page"],
                        on_page_sent=page_sent,
                    )
                except Exception as error:  # noqa: BLE001 - retry delivery
                    self.notification_failed(error)
                else:
                    self.daily_notified[task_id] = today
                    del self.daily_pending[task_id]
                    self.persist()
        # Ties, lower-ranked changes, and deleted runs never lower the record.
        self.persist()

    def notification_failed(self, error):
        self.snapshot["notification_error"] = (
            "Slack delivery failed; retrying on the next refresh."
        )
        logger.error("Slack notification failed\n%s", safe_traceback(error))

    def persist(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "snapshot": self.snapshot,
                    "notified": self.notified,
                    "daily_notified": self.daily_notified,
                    "daily_pending": self.daily_pending,
                }
            )
        )
        temporary.replace(self.state_path)
