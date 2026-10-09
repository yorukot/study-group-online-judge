import json
import struct
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from judge.leaderboard import LeaderboardService, rank_submissions, wandb_submissions
from judge.models import GradingType, MetricDirection
from judge.tasks.base import Task


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    instant = [datetime(2026, 10, 5, 10, 59, tzinfo=UTC)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant[0].astimezone(tz)

    monkeypatch.setattr("judge.leaderboard.datetime", Clock)
    monkeypatch.setenv("JUDGE_SLACK_DAILY_TIME", "19:00")
    monkeypatch.setenv("JUDGE_SLACK_TIMEZONE", "Asia/Taipei")
    monkeypatch.delenv("JUDGE_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("JUDGE_SLACK_CHANNEL_ID", raising=False)
    monkeypatch.setattr("judge.leaderboard._next_slack_attempt", {})
    monkeypatch.setattr("judge.leaderboard.sleep", lambda _: None)
    return instant


@pytest.fixture(autouse=True)
def notification_image(monkeypatch):
    # Upload protocol tests isolate the renderer; its browser integration is
    # exercised separately below against the real built website.
    def render(board, entries, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        return "test.png"

    monkeypatch.setattr("judge.leaderboard.render_standings", render)


def submission(actor="alice", score=4, passed=True, lab="lab4", day=1, id="one"):
    return {
        "id": id,
        "github_actor": actor,
        "task_id": lab,
        "metrics": {"score": score},
        "passed": passed,
        "submitted_at": f"2026-09-{day:02}T00:00:00Z",
        "run_url": None,
    }


def board(items, lab="lab4"):
    return next(b for b in rank_submissions(items) if b["id"] == lab)


def test_minimum_per_user_and_stable_ties():
    entries = board(
        [
            submission(score=5),
            submission(actor="ALICE", score=2, id="two", day=3),
            submission(actor="bob", score=2, day=2),
            submission(actor="eve", score=float("nan")),
            submission(actor="fail", score=1, passed=False),
        ]
    )["entries"]
    assert [e["github_actor"] for e in entries] == ["bob", "ALICE"]
    assert entries[1]["attempts"] == 2


def test_earliest_passing_submission_not_earliest_attempt():
    entries = board(
        [
            submission(lab="lab1", passed=False),
            submission(lab="lab1", day=3, id="later"),
            submission(actor="bob", lab="lab1", day=2),
        ],
        "lab1",
    )["entries"]
    assert [e["github_actor"] for e in entries] == ["bob", "alice"]
    assert entries[1]["submitted_at"].startswith("2026-09-03")


def test_maximize_custom_metric():
    class MaxTask(Task):
        id = "max"
        grading_type = GradingType.SCORE
        primary_metric = "accuracy"
        metric_direction = MetricDirection.MAXIMIZE

        def evaluate(self, submission):
            raise NotImplementedError

    items = [submission(lab="max"), submission(actor="bob", lab="max")]
    items[0]["metrics"] = {"accuracy": 0.8}
    items[1]["metrics"] = {"accuracy": 0.9}
    with patch("judge.leaderboard.TASKS", {"max": MaxTask()}):
        assert board(items, "max")["entries"][0]["github_actor"] == "bob"


def test_wandb_adapter(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "group")
    monkeypatch.setenv("WANDB_PROJECT", "judge")
    run = SimpleNamespace(
        id="run",
        config={
            "task_id": "lab4",
            "github_actor": "alice",
            "submitted_at": "2026-09-01T00:00:00Z",
        },
        summary={"score": 2},
        created_at="2026-09-02T00:00:00Z",
        url="https://wandb.ai/run",
    )
    with patch("wandb.Api") as api:
        api.return_value.runs.return_value = [run]
        items = wandb_submissions()
        assert items[0]["submitted_at"].startswith("2026-09-01")
        api.return_value.runs.assert_called_once_with(
            "group/judge", filters={"summary_metrics.judge_status": "completed"}
        )


def test_notifications_retry_and_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/example")
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch(
            "judge.leaderboard.wandb_submissions", return_value=[submission()]
        ) as source,
        patch("judge.leaderboard.post_slack") as slack,
    ):
        service.refresh()
        slack.assert_not_called()  # Initial snapshot establishes the baseline.
        source.return_value = [submission(score=2)]
        slack.side_effect = RuntimeError("unavailable")
        service.refresh()
        lab4 = next(lab for lab in service.snapshot["labs"] if lab["id"] == "lab4")
        assert lab4["entries"][0]["score"] == 2
        assert "notification_error" in service.snapshot
        service = LeaderboardService(tmp_path / "judge.db")
        slack.side_effect = None
        service.refresh()
        assert slack.call_count == 2
        service.refresh()
        assert slack.call_count == 2
        source.side_effect = RuntimeError("secret credentials")
        service.refresh()
        assert service.snapshot["error"]
        assert "secret" not in json.dumps(service.snapshot)
        lab4 = next(lab for lab in service.snapshot["labs"] if lab["id"] == "lab4")
        assert lab4["entries"][0]["score"] == 2


def test_public_endpoint_keeps_submission_auth(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from judge.main import app

    monkeypatch.setenv("JUDGE_API_TOKEN", "private-token")
    monkeypatch.setenv("JUDGE_DATABASE_PATH", str(tmp_path / "judge.db"))
    with (
        patch("judge.leaderboard.wandb_submissions", return_value=[]),
        TestClient(app) as client,
    ):
        response = client.get("/api/leaderboard")
        assert response.status_code == 200
        assert response.json()["source"] == "wandb"
        assert "private-token" not in response.text
        assert client.get("/jobs/missing").status_code == 401
        assert client.get("/healthz").status_code == 200


def test_slack_payload_uses_markdown():
    from judge.leaderboard import post_slack

    with patch("judge.leaderboard.urlopen") as send:
        send.return_value.__enter__.return_value.status = 200
        send.return_value.__enter__.return_value.read.return_value = b"ok"
        post_slack(
            "https://hooks.slack.com/example",
            rank_submissions([submission()]),
            ["lab4"],
        )
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        assert payload["blocks"][0]["text"]["type"] == "mrkdwn"
        assert ":first_place_medal: alice: 4" in payload["text"]
        assert (
            "*Lab 4 (Training GPT-2 from Scratch)*\n:first_place_medal:"
            in payload["text"]
        )
        assert "New record by alice!" in payload["text"]
        assert all("accessory" not in block for block in payload["blocks"])
        assert len(payload["blocks"]) == 2


def test_slack_workflow_payload_and_acknowledgment():
    from judge.leaderboard import post_slack

    with patch("judge.leaderboard.urlopen") as send:
        response = send.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = b'{"ok":true}'
        post_slack(
            "https://hooks.slack.com/triggers/example",
            rank_submissions([submission()]),
            ["lab4"],
        )
        payload = json.loads(send.call_args.args[0].data)
        assert set(payload) == {"text"}
        assert ":first_place_medal: alice:" in payload["text"]
        assert ":test_tube: New record by alice!\n\n*Lab 4" in payload["text"]
        response.read.return_value = b'{"ok":false}'
        import pytest

        with pytest.raises(RuntimeError):
            post_slack(
                "https://hooks.slack.com/triggers/example",
                rank_submissions([submission()]),
                ["lab4"],
            )


def test_only_all_time_records_notify(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "JUDGE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/triggers/test"
    )
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch(
            "judge.leaderboard.wandb_submissions", return_value=[submission(score=4)]
        ) as source,
        patch("judge.leaderboard.post_slack") as send,
    ):
        service.refresh()
        for items in (
            [submission(score=4), submission(actor="bob", score=5)],
            [submission(actor="bob", score=4)],
            [],
            [submission(score=6)],
        ):
            source.return_value = items
            service.refresh()
        send.assert_not_called()
        service = LeaderboardService(tmp_path / "judge.db")
        source.return_value = [submission(score=3)]
        service.refresh()
        send.assert_called_once()
        assert send.call_args.args[2] == ["lab4"]


def test_record_direction_and_pass_fail():
    from judge.leaderboard import new_record

    scored = {
        "grading_type": "score",
        "metric_direction": "maximize",
        "entries": [{"score": 9}],
    }
    assert new_record(scored, [{"score": 8}])
    assert not new_record(scored, [{"score": 9}])
    passed = {
        "grading_type": "pass_fail",
        "entries": [{"submitted_at": "2026-09-02T00:00:00Z"}],
    }
    assert new_record(passed, [])
    assert not new_record(passed, [{"submitted_at": "2026-09-01T00:00:00Z"}])


def test_refresh_logs_failure_details_without_credentials(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("WANDB_API_KEY", "test-secret-key")
    service = LeaderboardService(tmp_path / "judge.db")
    with patch(
        "judge.leaderboard.wandb_submissions",
        side_effect=RuntimeError("permission denied for test-secret-key"),
    ):
        service.refresh()
    assert "stage=fetch" in caplog.text
    assert "RuntimeError: permission denied for [REDACTED]" in caplog.text
    assert "test-secret-key" not in caplog.text
    assert "Traceback" in caplog.text
    assert "permission denied" not in json.dumps(service.snapshot)


def test_refresh_logs_success_and_ranking_failure(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="uvicorn.error.judge.leaderboard")
    service = LeaderboardService(tmp_path / "judge.db")
    with patch("judge.leaderboard.wandb_submissions", return_value=[submission()]):
        service.refresh()
        assert "W&B refresh started" in caplog.text
        assert "W&B refresh succeeded" in caplog.text
        assert "submissions=1" in caplog.text
        with patch(
            "judge.leaderboard.rank_submissions",
            side_effect=ValueError("bad ranking input"),
        ):
            service.refresh()
    assert "stage=rank" in caplog.text
    assert "ValueError: bad ranking input" in caplog.text
    lab4 = next(lab for lab in service.snapshot["labs"] if lab["id"] == "lab4")
    assert lab4["entries"][0]["score"] == 4


def test_podium_and_daily_complete_rankings(tmp_path, monkeypatch):
    from judge.leaderboard import post_slack

    monkeypatch.setenv("JUDGE_SLACK_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("JUDGE_SLACK_CHANNEL_ID", "C123")
    items = [
        submission(actor=f"user-{i:02}", score=i + 1, id=str(i)) for i in range(45)
    ]
    boards = rank_submissions(items)
    with patch("judge.leaderboard.upload_slack_card") as upload:
        post_slack(None, boards, ["lab4"], image_directory=tmp_path)
        blocks = upload.call_args.args[4]
        text = "\n".join(block["text"]["text"] for block in blocks)
        assert "New record by user-00!" in text
        assert ":second_place_medal: user-01: 2" in text
        assert ":third_place_medal: user-02: 3" in text
        assert "user-03" not in text
        assert upload.call_args.args[2].is_file()
        assert all("accessory" not in block for block in blocks)
        upload.reset_mock()
        post_slack(None, boards, ["lab4"], daily=True, image_directory=tmp_path)
        assert upload.call_count == 3
        messages = [call.args[4] for call in upload.call_args_list]
        assert all(
            ":test_tube: Daily Updates" in blocks[0]["text"]["text"]
            for blocks in messages
        )
        text = "\n".join(
            block["text"]["text"] for blocks in messages for block in blocks
        )
        for i in range(45):
            assert text.count(f" user-{i:02}: ") == 1
        assert "45. user-44: 45" in text
        assert "image_url" not in json.dumps(messages)


def test_daily_schedule_retry_and_restart(tmp_path, monkeypatch, clock):
    lab_ids = [lab["id"] for lab in rank_submissions([])]
    lab_count = len(lab_ids)
    monkeypatch.delenv("JUDGE_SLACK_DAILY_TIME")  # Verify the default 19:00 schedule.
    monkeypatch.setenv("JUDGE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/example")
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch("judge.leaderboard.wandb_submissions", return_value=[submission()]),
        patch("judge.leaderboard.post_slack") as send,
    ):
        service.refresh()  # 18:59 Taipei: baseline, no messages.
        send.assert_not_called()
        clock[0] = datetime(2026, 10, 5, 11, tzinfo=UTC)
        send.side_effect = [None, RuntimeError("unavailable")] + [None] * (
            lab_count - 2
        )
        service.refresh()
        assert send.call_count == lab_count
        assert service.daily_notified == {
            lab_id: "2026-10-05" for lab_id in lab_ids if lab_id != "lab2"
        }
        assert "notification_error" in service.snapshot
        service = LeaderboardService(tmp_path / "judge.db")
        send.side_effect = None
        service.refresh()
        assert send.call_count == lab_count + 1
        assert send.call_args.args[2] == ["lab2"]
        assert send.call_args.kwargs["daily"] is True
        service = LeaderboardService(tmp_path / "judge.db")
        service.refresh()
        assert send.call_count == lab_count + 1
        clock[0] = datetime(2026, 10, 6, 10, 59, tzinfo=UTC)
        service.refresh()
        assert send.call_count == lab_count + 1
        clock[0] = datetime(2026, 10, 6, 11, tzinfo=UTC)
        service.refresh()
        assert send.call_count == 2 * lab_count + 1


def test_daily_custom_timezone_and_time(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("JUDGE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/example")
    monkeypatch.setenv("JUDGE_SLACK_DAILY_TIME", "09:00")
    monkeypatch.setenv("JUDGE_SLACK_TIMEZONE", "America/New_York")
    clock[0] = datetime(2026, 10, 5, 12, 59, tzinfo=UTC)
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch("judge.leaderboard.wandb_submissions", return_value=[]),
        patch("judge.leaderboard.post_slack") as send,
    ):
        service.refresh()  # 08:59 New York.
        send.assert_not_called()
        clock[0] = datetime(2026, 10, 5, 13, tzinfo=UTC)
        service.refresh()
        assert send.call_count == len(rank_submissions([]))


def test_record_delivery_independent_of_daily_failure(tmp_path, monkeypatch, clock):
    lab_count = len(rank_submissions([]))
    monkeypatch.setenv("JUDGE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/example")
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch(
            "judge.leaderboard.wandb_submissions", return_value=[submission()]
        ) as source,
        patch("judge.leaderboard.post_slack") as send,
    ):
        service.refresh()
        source.return_value = [submission(score=2)]
        clock[0] = datetime(2026, 10, 5, 11, tzinfo=UTC)
        send.side_effect = [None, RuntimeError("unavailable")] + [None] * (
            lab_count - 1
        )
        service.refresh()
        assert send.call_count == lab_count + 1
        service = LeaderboardService(tmp_path / "judge.db")
        send.side_effect = None
        service.refresh()
        assert send.call_count == lab_count + 2
        assert send.call_args.kwargs["daily"] is True
        assert send.call_args.args[2] == ["lab1"]


def test_no_daily_summary_from_failed_source(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("JUDGE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/example")
    clock[0] = datetime(2026, 10, 5, 11, tzinfo=UTC)
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch(
            "judge.leaderboard.wandb_submissions", side_effect=RuntimeError("offline")
        ),
        patch("judge.leaderboard.post_slack") as send,
    ):
        service.refresh()
        send.assert_not_called()
        assert service.daily_notified == {}


def test_workflow_text_only_and_escaping(tmp_path):
    from judge.leaderboard import post_slack

    with patch("judge.leaderboard.urlopen") as send:
        response = send.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = b'{"ok":true}'
        post_slack(
            "https://hooks.slack.com/triggers/example",
            rank_submissions([submission(actor="a<&")]),
            ["lab4"],
            image_directory=tmp_path,
        )
        payload = json.loads(send.call_args.args[0].data)
        assert set(payload) == {"text"}
        assert "a&lt;&amp;" in payload["text"]
        assert "https://" not in payload["text"]
        assert not list(tmp_path.glob("*.png"))


def test_standings_png_remains_immutable(tmp_path, monkeypatch):
    from playwright.sync_api import Browser, Locator

    from judge.leaderboard_images import render_standings

    new_page = Browser.new_page

    def page_with_avatar(self, **kwargs):
        page = new_page(self, **kwargs)
        page.route(
            "https://github.com/**",
            lambda route: route.fulfill(
                content_type="image/svg+xml",
                body='<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36"><rect width="36" height="36" fill="blue"/></svg>',
            ),
        )
        return page

    monkeypatch.setattr(Browser, "new_page", page_with_avatar)
    screenshot = Locator.screenshot

    def capture_website(self, **kwargs):
        page = self.page
        assert page.locator(".board h2").inner_text() == "Training GPT-2 from Scratch"
        assert "Georgia" in page.locator(".board h2").evaluate(
            "element => getComputedStyle(element).fontFamily"
        )
        assert page.locator(".search input").is_visible()
        assert page.locator(".avatar img").evaluate("image => image.naturalWidth") == 36
        assert (
            "Results from Weights & Biases"
            in page.locator(".board-footer").inner_text()
        )
        return screenshot(self, **kwargs)

    monkeypatch.setattr(Locator, "screenshot", capture_website)
    standings = board([submission()])
    filename = render_standings(standings, standings["entries"], tmp_path)
    first = (tmp_path / filename).read_bytes()
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">I", first[16:20])[0] == 2400
    standings["entries"][0]["score"] = 2
    second = render_standings(standings, standings["entries"], tmp_path)
    assert second != filename
    assert (tmp_path / filename).read_bytes() == first


def test_renderer_requires_built_ui(tmp_path, monkeypatch):
    from judge.leaderboard_images import render_standings

    monkeypatch.setenv("JUDGE_UI_DIST", str(tmp_path / "missing-ui"))
    standings = board([submission()])
    with pytest.raises(RuntimeError, match="npm run build"):
        render_standings(standings, standings["entries"], tmp_path)


def test_partial_daily_resumes_saved_page_after_restart(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("JUDGE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/example")
    items = [
        submission(actor=f"user-{i:02}", score=i + 1, id=str(i)) for i in range(25)
    ]
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch("judge.leaderboard.wandb_submissions", return_value=items) as source,
        patch("judge.leaderboard.urlopen") as send,
    ):
        service.refresh()
        clock[0] = datetime(2026, 10, 5, 11, tzinfo=UTC)
        response = send.return_value
        response.__enter__.return_value.status = 200
        response.__enter__.return_value.read.return_value = b"ok"
        # Two pass/fail labs, then the first score page succeeds; second fails.
        send.side_effect = [response, response, response, RuntimeError("offline")] + [
            response
        ] * (len(rank_submissions([])) - 3)
        service.refresh()
        assert service.daily_pending["lab4"]["next_page"] == 1
        service = LeaderboardService(tmp_path / "judge.db")
        # Rankings change during the retry; finish today's original daily snapshot.
        source.return_value = items[:-1]
        send.side_effect = None
        send.reset_mock()
        service.refresh()
        send.assert_called_once()
        payload = json.loads(send.call_args.args[0].data)
        assert "user-20" in payload["text"]
        assert "user-24" in payload["text"]
        assert "user-00" not in payload["text"]
        assert not service.daily_pending
        assert service.daily_notified["lab4"] == "2026-10-05"


def test_slack_rate_limit_backoff(monkeypatch):
    from email.message import Message
    from urllib.error import HTTPError
    from urllib.request import Request

    from judge.leaderboard import open_slack

    headers = Message()
    headers["Retry-After"] = "60"
    request = Request("https://hooks.slack.com/example")
    with patch(
        "judge.leaderboard.urlopen",
        side_effect=HTTPError(request.full_url, 429, "limited", headers, None),
    ) as send:
        with pytest.raises(HTTPError):
            open_slack(request, False)
        with pytest.raises(RuntimeError, match="backoff"):
            open_slack(request, False)
        send.assert_called_once()


def test_slack_webhook_pacing():
    from urllib.request import Request

    from judge.leaderboard import open_slack

    with (
        patch("judge.leaderboard.monotonic", return_value=100),
        patch("judge.leaderboard.sleep") as wait,
        patch("judge.leaderboard.urlopen"),
    ):
        request = Request("https://hooks.slack.com/example")
        open_slack(request, False)
        open_slack(request, False)
        wait.assert_called_once_with(1)
        wait.reset_mock()
        request = Request("https://hooks.slack.com/triggers/example")
        open_slack(request, True)
        open_slack(request, True)
        wait.assert_called_once_with(6)


def slack_response(payload, status=200):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = status
    response.read.return_value = (
        json.dumps(payload).encode() if isinstance(payload, dict) else payload
    )
    return response


def test_inline_png_upload_and_share_without_webhook(tmp_path, monkeypatch):
    from urllib.parse import parse_qs

    from judge.leaderboard import post_slack

    monkeypatch.setenv("JUDGE_SLACK_BOT_TOKEN", "test-bot-secret")
    monkeypatch.setenv("JUDGE_SLACK_CHANNEL_ID", "C123")
    responses = [
        slack_response(
            {
                "ok": True,
                "file_id": "F123",
                "upload_url": "https://files.slack.com/upload/v1/test",
            }
        ),
        slack_response(b"ok"),
        slack_response({"ok": True, "files": [{"id": "F123"}]}),
    ]
    acknowledged = []
    with patch("judge.leaderboard.urlopen", side_effect=responses) as send:
        post_slack(
            None,
            rank_submissions([submission()]),
            ["lab4"],
            image_directory=tmp_path,
            on_page_sent=acknowledged.append,
        )
    assert acknowledged == [1]
    assert send.call_count == 3
    ticket, upload, complete = [call.args[0] for call in send.call_args_list]
    assert ticket.full_url == "https://slack.com/api/files.getUploadURLExternal"
    assert upload.full_url == "https://files.slack.com/upload/v1/test"
    assert complete.full_url == "https://slack.com/api/files.completeUploadExternal"
    fields = parse_qs(ticket.data.decode())
    png = next(tmp_path.glob("*.png"))
    assert int(fields["length"][0]) == len(png.read_bytes())
    assert upload.data == png.read_bytes()
    assert upload.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert ticket.get_header("Authorization") == "Bearer test-bot-secret"
    assert complete.get_header("Authorization") == "Bearer test-bot-secret"
    assert upload.get_header("Authorization") is None
    fields = parse_qs(complete.data.decode())
    assert fields["channel_id"] == ["C123"]
    assert json.loads(fields["files"][0])[0]["id"] == "F123"
    blocks = json.loads(fields["blocks"][0])
    assert "New record by alice!" in blocks[0]["text"]["text"]
    assert ":first_place_medal: alice: 4" in blocks[1]["text"]["text"]
    assert all("accessory" not in block for block in blocks)
    assert "image_url" not in fields["blocks"][0]
    assert "initial_comment" not in fields  # Would override Slack's ranking blocks.
    assert "test-bot-secret" not in complete.data.decode()


@pytest.mark.parametrize("stage", ["ticket", "upload", "share"])
def test_inline_upload_failures_are_not_acknowledged(tmp_path, monkeypatch, stage):
    from judge.leaderboard import post_slack

    monkeypatch.setenv("JUDGE_SLACK_BOT_TOKEN", "test-bot-secret")
    monkeypatch.setenv("JUDGE_SLACK_CHANNEL_ID", "C123")
    ticket = slack_response(
        {
            "ok": True,
            "file_id": "F123",
            "upload_url": "https://files.slack.com/upload/v1/test",
        }
    )
    responses = {
        "ticket": [slack_response({"ok": False, "error": "missing_scope"})],
        "upload": [ticket, slack_response(b"failed", status=503)],
        "share": [
            ticket,
            slack_response(b"ok"),
            slack_response({"ok": False, "error": "not_in_channel"}),
        ],
    }[stage]
    acknowledged = []
    with (
        patch("judge.leaderboard.urlopen", side_effect=responses) as send,
        pytest.raises(RuntimeError),
    ):
        post_slack(
            None,
            rank_submissions([submission()]),
            ["lab4"],
            image_directory=tmp_path,
            on_page_sent=acknowledged.append,
        )
    assert acknowledged == []
    assert send.call_count == len(responses)


def test_daily_bot_upload_does_not_need_a_webhook(tmp_path, monkeypatch, clock):
    lab_count = len(rank_submissions([]))
    monkeypatch.delenv("JUDGE_SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("JUDGE_SLACK_BOT_TOKEN", "test-bot-secret")
    monkeypatch.setenv("JUDGE_SLACK_CHANNEL_ID", "C123")
    service = LeaderboardService(tmp_path / "judge.db")
    with (
        patch("judge.leaderboard.wandb_submissions", return_value=[submission()]),
        patch("judge.leaderboard.upload_slack_card") as share,
    ):
        service.refresh()
        share.assert_not_called()
        clock[0] = datetime(2026, 10, 5, 11, tzinfo=UTC)
        service.refresh()
        assert share.call_count == lab_count
        assert len(service.daily_notified) == lab_count
        service = LeaderboardService(tmp_path / "judge.db")
        service.refresh()
        assert share.call_count == lab_count


def test_partial_bot_config_is_an_error(tmp_path, monkeypatch):
    from judge.leaderboard import post_slack

    monkeypatch.setenv("JUDGE_SLACK_BOT_TOKEN", "test-bot-secret")
    with patch("judge.leaderboard.urlopen") as send:
        with pytest.raises(RuntimeError, match="JUDGE_SLACK_CHANNEL_ID"):
            post_slack(
                "https://hooks.slack.com/example",
                rank_submissions([submission()]),
                ["lab4"],
                image_directory=tmp_path,
            )
        send.assert_not_called()


def test_bot_token_is_redacted_from_notification_logs(
    tmp_path, monkeypatch, clock, caplog
):
    monkeypatch.setenv("JUDGE_SLACK_BOT_TOKEN", "test-bot-secret")
    monkeypatch.setenv("JUDGE_SLACK_CHANNEL_ID", "C123")
    service = LeaderboardService(tmp_path / "judge.db")
    clock[0] = datetime(2026, 10, 5, 11, tzinfo=UTC)
    with (
        patch("judge.leaderboard.wandb_submissions", return_value=[submission()]),
        patch(
            "judge.leaderboard.upload_slack_card",
            side_effect=RuntimeError("denied for test-bot-secret"),
        ),
    ):
        service.refresh()
    assert "denied for [REDACTED]" in caplog.text
    assert "test-bot-secret" not in caplog.text
    assert "test-bot-secret" not in json.dumps(service.snapshot)
    assert not service.daily_notified
