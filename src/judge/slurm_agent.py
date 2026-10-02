"""Outbound Nano4 sub-judge daemon; participant jobs run directly under Slurm."""

import argparse
import json
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from judge.models import (
    JobOffer,
    JobReceipt,
    JudgeResult,
    RemoteEvent,
    RemoteEventKind,
    SubJudgeBackend,
    SubJudgeRegistration,
)
from judge.repository import checkout_repository, create_job_workspace
from judge.slurm_executor import SlurmExecutor


class MasterError(RuntimeError):
    pass


class MasterClient:
    def __init__(self, base_url: str, token: str, judge_id: str) -> None:
        if not base_url.startswith("https://") and not base_url.startswith(
            "http://127.0.0.1:"
        ):
            raise ValueError("Master URL must use HTTPS outside local tests")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.judge_id = judge_id

    def request(self, method: str, path: str, body: Any = None) -> Any:
        payload = None if body is None else json.dumps(body).encode()
        request = Request(
            f"{self.base_url}{path}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "study-group-online-judge/0.1",
            },
        )
        try:
            with urlopen(request, timeout=35) as response:
                if response.status == 204:
                    return None
                return json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode(errors="replace")
            raise MasterError(f"Master returned HTTP {error.code}: {detail}") from error

    def register(self, task_ids: list[str], max_gpus: int, revision: str) -> None:
        registration = SubJudgeRegistration(
            id=self.judge_id,
            backend=SubJudgeBackend.SLURM,
            task_ids=task_ids,
            max_gpus=max_gpus,
            judge_revision=revision,
        )
        self.request("POST", "/agents/register", registration.model_dump(mode="json"))

    def next_offer(self) -> JobOffer | None:
        response = self.request("GET", f"/agents/{self.judge_id}/next")
        return None if response is None else JobOffer.model_validate(response)

    def heartbeat(self) -> None:
        self.request("POST", f"/agents/{self.judge_id}/heartbeat")

    def receipt(self, job_id: str, receipt: JobReceipt) -> None:
        self.request(
            "POST",
            f"/agents/{self.judge_id}/jobs/{job_id}/receipt",
            receipt.model_dump(mode="json"),
        )

    def event(self, job_id: str, event: RemoteEvent) -> None:
        self.request(
            "POST",
            f"/agents/{self.judge_id}/jobs/{job_id}/events",
            event.model_dump(mode="json"),
        )


class AgentLedger:
    """Keep Slurm IDs and unacknowledged events across daemon restarts."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    offer_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    slurm_job_id TEXT,
                    log_offset INTEGER NOT NULL DEFAULT 0,
                    next_sequence INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    delivered INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (job_id, sequence)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def claim(self, offer: JobOffer) -> sqlite3.Row:
        encoded = offer.model_dump_json()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO jobs (job_id, offer_json, state) "
                "VALUES (?, ?, 'preparing')",
                (offer.job_id, encoded),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (offer.job_id,)
            ).fetchone()
        assert row is not None
        if row["offer_json"] != encoded:
            raise ValueError("Job ID was reused for a different offer")
        return row

    def state(self, job_id: str) -> sqlite3.Row:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def set_submitting(self, job_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE jobs SET state = 'submitting' "
                "WHERE job_id = ? AND state = 'preparing'",
                (job_id,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Job is not preparing")

    def set_submitted(self, job_id: str, slurm_job_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE jobs SET state = 'submitted', slurm_job_id = ? "
                "WHERE job_id = ? AND state = 'submitting'",
                (slurm_job_id, job_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Job is not submitting")

    def append_event(
        self,
        job_id: str,
        kind: RemoteEventKind,
        *,
        log_offset: int | None = None,
        line: str | None = None,
        result: JudgeResult | None = None,
        error: str | None = None,
    ) -> RemoteEvent:
        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None or row["slurm_job_id"] is None:
                    raise ValueError("Job has not been submitted")
                event = RemoteEvent(
                    sequence=row["next_sequence"],
                    kind=kind,
                    slurm_job_id=row["slurm_job_id"],
                    line=line,
                    result=result,
                    error=error,
                )
                connection.execute(
                    "INSERT INTO events (job_id, sequence, event_json) VALUES (?, ?, ?)",
                    (job_id, event.sequence, event.model_dump_json()),
                )
                terminal = kind in (RemoteEventKind.COMPLETED, RemoteEventKind.FAILED)
                connection.execute(
                    "UPDATE jobs SET next_sequence = next_sequence + 1, "
                    "log_offset = COALESCE(?, log_offset), state = ? WHERE job_id = ?",
                    (log_offset, "finished" if terminal else row["state"], job_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return event

    def pending_events(self, job_id: str) -> list[RemoteEvent]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT event_json FROM events WHERE job_id = ? AND delivered = 0 "
                "ORDER BY sequence",
                (job_id,),
            ).fetchall()
        return [RemoteEvent.model_validate_json(row[0]) for row in rows]

    def delivered(self, job_id: str, sequence: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE events SET delivered = 1 WHERE job_id = ? AND sequence = ?",
                (job_id, sequence),
            )

    def submitted_job_ids(self) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT job_id FROM jobs WHERE slurm_job_id IS NOT NULL"
            ).fetchall()
        return [row[0] for row in rows]


class SlurmAgent:
    def __init__(
        self,
        *,
        client: MasterClient,
        ledger: AgentLedger,
        executor: SlurmExecutor,
        work_root: Path,
        task_ids: list[str],
        max_gpus: int,
        revision: str,
        poll_seconds: float = 30,
    ) -> None:
        self.client = client
        self.ledger = ledger
        self.executor = executor
        self.work_root = work_root
        self.task_ids = task_ids
        self.max_gpus = max_gpus
        self.revision = revision
        self.poll_seconds = poll_seconds
        self._monitors: set[str] = set()
        self._monitor_lock = threading.Lock()
        self._registered = threading.Event()

    def _heartbeat_loop(self) -> None:
        while True:
            time.sleep(60)
            if not self._registered.is_set():
                continue
            try:
                self.client.heartbeat()
            except Exception as error:  # noqa: BLE001 - reconnect after transient failures
                print(
                    f"[agent] heartbeat failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                self._registered.clear()

    def accept(self, offer: JobOffer) -> JobReceipt:
        if offer.submission.task_id not in self.task_ids:
            return JobReceipt(accepted=False, error="Task is not enabled on this agent")
        if offer.resources.gpus < 1 or offer.resources.gpus > self.max_gpus:
            return JobReceipt(
                accepted=False, error="GPU request exceeds agent capacity"
            )
        row = self.ledger.claim(offer)
        if row["slurm_job_id"] is not None:
            self._start_monitor(offer.job_id)
            return JobReceipt(accepted=True, slurm_job_id=row["slurm_job_id"])
        if row["state"] == "submitting":
            return JobReceipt(
                accepted=False,
                error="Slurm submission outcome is ambiguous; manual reconciliation required",
            )
        try:
            workspace = self.work_root / offer.job_id
            if not workspace.exists():
                create_job_workspace(self.work_root, offer.job_id)
            submission = workspace / "submission"
            if not submission.exists():
                checkout_repository(
                    offer.submission.repo_url,
                    offer.submission.commit_sha,
                    submission,
                )
            output = workspace / "output"
            self.ledger.set_submitting(offer.job_id)
            slurm_job_id = self.executor.submit(
                task_id=offer.submission.task_id,
                resources=offer.resources,
                submission=submission,
                output_directory=output,
            )
            self.ledger.set_submitted(offer.job_id, slurm_job_id)
            self._start_monitor(offer.job_id)
            return JobReceipt(accepted=True, slurm_job_id=slurm_job_id)
        except Exception as error:  # noqa: BLE001 - report scheduling failures to master
            return JobReceipt(
                accepted=False,
                error=f"{type(error).__name__}: {error}",
            )

    def _start_monitor(self, job_id: str) -> None:
        with self._monitor_lock:
            if job_id in self._monitors:
                return
            self._monitors.add(job_id)
        thread = threading.Thread(target=self._monitor, args=(job_id,), daemon=True)
        thread.start()

    def _monitor(self, job_id: str) -> None:
        try:
            self.monitor(job_id)
        finally:
            with self._monitor_lock:
                self._monitors.discard(job_id)

    def _deliver_pending(self, job_id: str) -> None:
        for event in self.ledger.pending_events(job_id):
            self.client.event(job_id, event)
            self.ledger.delivered(job_id, event.sequence)

    def _drain_log(self, job_id: str, *, force: bool = False) -> bool:
        row = self.ledger.state(job_id)
        path = self.work_root / job_id / "output" / "slurm.log"
        if not path.is_file():
            return False
        offset = row["log_offset"]
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(8192)
        if not chunk:
            return False
        boundary = max(chunk.rfind(b"\n"), chunk.rfind(b"\r")) + 1
        if boundary == 0:
            if len(chunk) < 8192 and not force:
                return False
            boundary = len(chunk)
        self.ledger.append_event(
            job_id,
            RemoteEventKind.LOG,
            line=chunk[:boundary].decode(errors="replace"),
            log_offset=offset + boundary,
        )
        return True

    def monitor(self, job_id: str) -> None:
        while True:
            try:
                row = self.ledger.state(job_id)
                if row["next_sequence"] == 1:
                    self.ledger.append_event(job_id, RemoteEventKind.STARTED)
                self._deliver_pending(job_id)
                if row["state"] == "finished":
                    return
                self._drain_log(job_id)
                self._deliver_pending(job_id)
                slurm_job_id = row["slurm_job_id"]
                assert isinstance(slurm_job_id, str)
                state = self.executor.status(slurm_job_id)
                if state is not None and state.terminal:
                    while self._drain_log(job_id, force=True):
                        self._deliver_pending(job_id)
                    if state.succeeded:
                        result_path = self.work_root / job_id / "output" / "result.json"
                        try:
                            result = JudgeResult.model_validate_json(
                                result_path.read_text()
                            )
                        except (OSError, ValueError) as error:
                            self.ledger.append_event(
                                job_id,
                                RemoteEventKind.FAILED,
                                error=f"Missing or invalid judge result: {error}",
                            )
                        else:
                            self.ledger.append_event(
                                job_id, RemoteEventKind.COMPLETED, result=result
                            )
                    else:
                        self.ledger.append_event(
                            job_id,
                            RemoteEventKind.FAILED,
                            error=f"Slurm job {slurm_job_id} ended in {state.state} "
                            f"(exit {state.exit_code or 'unknown'})",
                        )
                    self._deliver_pending(job_id)
                    return
            except Exception as error:  # noqa: BLE001 - retry after transient failures
                print(
                    f"[agent] monitor retry for {job_id}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            time.sleep(self.poll_seconds)

    def run(self) -> None:
        for job_id in self.ledger.submitted_job_ids():
            self._start_monitor(job_id)
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        while True:
            try:
                if not self._registered.is_set():
                    self.client.register(self.task_ids, self.max_gpus, self.revision)
                    self._registered.set()
                offer = self.client.next_offer()
                if offer is None:
                    continue
                receipt = self.accept(offer)
                self.client.receipt(offer.job_id, receipt)
            except Exception as error:  # noqa: BLE001 - reconnect after transient failures
                print(
                    f"[agent] reconnecting: {type(error).__name__}: {error}", flush=True
                )
                self._registered.clear()
                time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Register and poll once")
    arguments = parser.parse_args()
    token = os.environ.get("JUDGE_AGENT_TOKEN")
    master_url = os.environ.get("JUDGE_MASTER_URL")
    if not token or not master_url:
        raise RuntimeError("JUDGE_AGENT_TOKEN and JUDGE_MASTER_URL are required")
    task_ids = [
        part.strip()
        for part in os.environ.get("JUDGE_TASK_IDS", "").split(",")
        if part.strip()
    ]
    if not task_ids:
        raise RuntimeError("JUDGE_TASK_IDS must list enabled GPU tasks")
    revision = os.environ.get("JUDGE_REVISION")
    if not revision:
        raise RuntimeError("JUDGE_REVISION must identify the trusted judge checkout")
    work_root = Path(os.environ.get("JUDGE_WORK_ROOT", "/work/study-group-oj"))
    trusted_root = Path(os.environ.get("JUDGE_TRUSTED_ROOT", str(Path.cwd())))
    agent = SlurmAgent(
        client=MasterClient(
            master_url, token, os.environ.get("JUDGE_AGENT_ID", "nano4")
        ),
        ledger=AgentLedger(work_root / "agent.db"),
        executor=SlurmExecutor(
            account=os.environ.get("JUDGE_SLURM_ACCOUNT", "ACD115198"),
            gpu_resource=os.environ.get("JUDGE_SLURM_GPU_RESOURCE", "gpu"),
            trusted_root=trusted_root,
        ),
        work_root=work_root,
        task_ids=task_ids,
        max_gpus=int(os.environ.get("JUDGE_MAX_GPUS", "8")),
        revision=revision,
    )
    if arguments.once:
        agent.client.register(agent.task_ids, agent.max_gpus, agent.revision)
        offer = agent.client.next_offer()
        if offer is not None:
            agent.client.receipt(offer.job_id, agent.accept(offer))
    else:
        agent.run()


if __name__ == "__main__":
    main()
