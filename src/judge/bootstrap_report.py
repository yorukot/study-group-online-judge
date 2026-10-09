"""Standalone fallback for W&B failures before the trusted repo is usable.

The transport streams this packaged source with remote_store.py to uv. It only
needs W&B and the standard library, not the failed repository's environment.
"""

import json
import netrc
import os
import sys
import time
from pathlib import Path

import wandb

from judge.remote_store import job_lock, reporting_output, save_record


def report_failure(request: dict, error: str) -> dict:
    identity = {
        key: value
        for key, value in request.items()
        if key not in ("setup_offset", "log_offset")
    }
    root = Path(request["config"]["work_root"])
    job_id = request["job_id"]
    workspace = root / "jobs" / job_id
    path = root / "job-records" / f"{job_id}.json"
    snapshot = {
        "error": error,
        "report_pending": True,
        "setup_offset": request.get("setup_offset", 0),
        "log_offset": request.get("log_offset", 0),
    }
    with job_lock(workspace):
        existing = path if path.exists() else workspace / "submission.json"
        record = json.loads(existing.read_text()) if existing.exists() else None
        if record is not None:
            if record["request"]["submission"] != request["submission"]:
                raise ValueError("Remote job identity mismatch")
            if record["state"] not in ("preparing", "failed"):
                # A bootstrap problem during recovery cannot invalidate an
                # already accepted Slurm job. Retry monitoring after repair.
                return {
                    "report_pending": True,
                    "setup_offset": snapshot["setup_offset"],
                    "log_offset": snapshot["log_offset"],
                }
            if record.get("report", {}).get("complete"):
                return record["terminal"]
            if record["state"] == "failed":
                error = record["error"]
                snapshot["error"] = error
        finished_at = record.get("finished_at", time.time()) if record else time.time()
        record = {
            "request": identity,
            "state": "failed",
            "error": error,
            "finished_at": finished_at,
            "terminal": snapshot,
        }
        output = workspace / "output"
        output.mkdir(parents=True, exist_ok=True)
        save_record(path, record)
        with reporting_output(output / "reporting.log"):
            try:
                auth = netrc.netrc(str(root / ".netrc")).authenticators("api.wandb.ai")
                if auth is None or not auth[2]:
                    raise ValueError("Missing remote W&B credential")
                run = wandb.init(
                    project=request["config"]["wandb_project"],
                    entity=request["config"]["wandb_entity"],
                    id=job_id,
                    resume="allow",
                    job_type="submission",
                    dir=str(output),
                    config={
                        "job_id": job_id,
                        **request["submission"],
                        "resources": request["resources"],
                        "execution_backend": "ssh_slurm",
                    },
                    settings=wandb.Settings(
                        api_key=auth[2],
                        init_timeout=30,
                        finish_timeout=30,
                        finish_timeout_raises=True,
                        mode="online",
                        console="wrap",
                    ),
                    save_code=False,
                )
                try:
                    run.summary["judge_status"] = "error"
                    run.summary["error"] = error
                    print(f"[judge] {error}", flush=True)
                    snapshot["wandb_url"] = run.url
                finally:
                    run.finish(exit_code=1)
                snapshot["report_pending"] = False
                record["report"] = {"complete": True, "url": snapshot["wandb_url"]}
            except Exception as failure:  # noqa: BLE001 - retry W&B independently
                record["report_error"] = (
                    f"{type(failure).__name__}: Remote W&B publication failed"
                )
                print(record["report_error"], flush=True)
        record["terminal"] = snapshot
        save_record(path, record)
        save_record(workspace / "submission.json", record)
        if not snapshot["report_pending"]:
            marker = workspace / ".finished"
            marker.touch()
            os.utime(marker, (record["finished_at"], record["finished_at"]))
        return snapshot


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    print(json.dumps(report_failure(payload["request"], payload["error"])))
