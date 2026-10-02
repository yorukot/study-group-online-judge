import json
from types import SimpleNamespace
from unittest.mock import patch

from judge.leaderboard import LeaderboardService, rank_submissions, wandb_submissions
from judge.models import GradingType, MetricDirection
from judge.tasks.base import Task


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
        assert service.snapshot["labs"][-1]["entries"][0]["score"] == 2
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
        assert service.snapshot["labs"][-1]["entries"][0]["score"] == 2


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
        assert "🏅 alice — 4" in payload["text"]
        assert "↓ Lower is better" in payload["text"]
        assert "*OJ Leaderboard update*" in payload["text"]


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
        assert "🏅 alice" in payload["text"]
        assert "🧪 *Lab 4" in payload["text"]
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
    assert service.snapshot["labs"][-1]["entries"][0]["score"] == 4
