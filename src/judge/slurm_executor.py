"""Submit a participant's sbatch file to Nano4 without Docker."""

import logging
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from judge.execution_logging import log_event
from judge.models import Resources
from judge.repository import SECRET_ENVIRONMENT_VARIABLES

SLURM_JOB_ID = re.compile(r"[0-9]+")
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
EXCLUDE_NODES = ["25a-hgpn003", "25a-hgpn062", "25a-hgpn145"]


class _SlurmLogContext(TypedDict):
    job_id: str | None
    command: str


@dataclass(frozen=True)
class SlurmState:
    state: str
    exit_code: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.state == "COMPLETED" and self.exit_code == "0:0"


class SlurmExecutor:
    def __init__(
        self,
        *,
        account: str = "ACD115198",
        gpu_resource: str = "gpu",
        trusted_root: Path,
        sbatch_binary: str = "sbatch",
        sacct_binary: str = "sacct",
        job_id: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", account):
            raise ValueError("Invalid Slurm account")
        if not re.fullmatch(r"[A-Za-z0-9_:-]+", gpu_resource):
            raise ValueError("Invalid Slurm GPU resource")
        self.account = account
        self.gpu_resource = gpu_resource
        self.trusted_root = trusted_root.resolve()
        self.sbatch_binary = sbatch_binary
        self.sacct_binary = sacct_binary
        self.job_id = job_id

    def _run_command(self, command: list[str], **kwargs):
        started = time.monotonic()
        context: _SlurmLogContext = {
            "job_id": self.job_id,
            "command": shlex.join(command),
        }
        log_event("slurm.command.started", **context)
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                **kwargs,
            )
        except (subprocess.SubprocessError, OSError) as error:
            log_event(
                "slurm.command.failed",
                level=logging.WARNING,
                **context,
                elapsed_seconds=round(time.monotonic() - started, 3),
                error_type=type(error).__name__,
                error=str(error),
                returncode=getattr(error, "returncode", None),
                stdout=getattr(error, "stdout", None),
                stderr=getattr(error, "stderr", None),
            )
            raise
        log_event(
            "slurm.command.response",
            **context,
            elapsed_seconds=round(time.monotonic() - started, 3),
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )
        return process

    def build_submit_command(
        self,
        *,
        task_id: str,
        resources: Resources,
        submission: Path,
        output_directory: Path,
        job_name: str | None = None,
    ) -> list[str]:
        if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", task_id) is None:
            raise ValueError("Invalid task ID for Slurm")
        if resources.gpus < 1 or resources.gpus > 8:
            raise ValueError("Nano4 H200 jobs require 1 to 8 GPUs")
        if resources.cpus > 12 * resources.gpus:
            raise ValueError("Nano4 H200 allows at most 12 CPUs per GPU")
        if resources.memory_gb > 200 * resources.gpus:
            raise ValueError("Nano4 H200 allows at most 200 GiB per GPU")
        if resources.timeout_seconds > 48 * 3600:
            raise ValueError("Nano4 H200 jobs cannot exceed 48 hours")

        if job_name is not None and re.fullmatch(r"[A-Za-z0-9_-]+", job_name) is None:
            raise ValueError("Invalid Slurm job name")
        script = submission / "src" / "labs" / f"{task_id}.sbatch"
        if not script.is_file():
            raise FileNotFoundError(f"Expected src/labs/{task_id}.sbatch")
        partition = "dev" if resources.timeout_seconds <= 4 * 3600 else "8gpus"
        return [
            self.sbatch_binary,
            "--parsable",
            f"--account={self.account}",
            f"--partition={partition}",
            f"--gres={self.gpu_resource}:{resources.gpus}",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory_gb}G",
            f"--time={math.ceil(resources.timeout_seconds / 60)}",
            f"--job-name={job_name or f'judge-{task_id}'}",
            f"--chdir={submission.resolve()}",
            f"--output={(output_directory / 'slurm.log').resolve()}",
            f"--error={(output_directory / 'slurm.log').resolve()}",
            f"--exclude={','.join(EXCLUDE_NODES)}" if EXCLUDE_NODES else "",
            str(script.resolve()),
        ]

    def submit(
        self,
        *,
        task_id: str,
        resources: Resources,
        submission: Path,
        output_directory: Path,
        job_name: str | None = None,
        hf_home: Path | None = None,
        uv_cache: Path | None = None,
    ) -> str:
        output_directory.mkdir(parents=True, exist_ok=True)
        command = self.build_submit_command(
            task_id=task_id,
            resources=resources,
            submission=submission,
            output_directory=output_directory,
            job_name=job_name,
        )
        environment = os.environ.copy()
        for variable in SECRET_ENVIRONMENT_VARIABLES:
            environment.pop(variable, None)
        environment.update(
            {
                "JUDGE_TASK_ID": task_id,
                "JUDGE_SUBMISSION_DIR": str(submission.resolve()),
                "JUDGE_OUTPUT_DIR": str(output_directory.resolve()),
                "JUDGE_TRUSTED_ROOT": str(self.trusted_root),
                "HF_HOME": str(
                    hf_home or Path(os.environ.get("JUDGE_HF_HOME", "/work/hf-cache"))
                ),
                "UV_CACHE_DIR": str(
                    uv_cache
                    or Path(os.environ.get("JUDGE_UV_CACHE_DIR", "/work/uv-cache"))
                ),
            }
        )
        process = self._run_command(command, env=environment)
        job_id = process.stdout.strip().split(";", maxsplit=1)[0]
        if SLURM_JOB_ID.fullmatch(job_id) is None:
            log_event(
                "slurm.submission.invalid_response",
                level=logging.ERROR,
                job_id=self.job_id,
                stdout=process.stdout,
            )
            raise RuntimeError(
                f"Unrecognized sbatch response: {process.stdout.strip()}"
            )
        log_event("slurm.submission.accepted", job_id=self.job_id, slurm_job_id=job_id)
        return job_id

    def status(self, slurm_job_id: str) -> SlurmState | None:
        if SLURM_JOB_ID.fullmatch(slurm_job_id) is None:
            raise ValueError("Invalid Slurm job ID")
        process = self._run_command(
            [
                self.sacct_binary,
                "--jobs",
                slurm_job_id,
                "--format=JobIDRaw,State,ExitCode",
                "--parsable2",
                "--noheader",
            ],
        )
        for line in process.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 3 and fields[0] == slurm_job_id:
                state = SlurmState(fields[1].split(" ", maxsplit=1)[0], fields[2])
                log_event(
                    "slurm.status",
                    job_id=self.job_id,
                    slurm_job_id=slurm_job_id,
                    state=state.state,
                    exit_code=state.exit_code,
                    terminal=state.terminal,
                )
                return state
        log_event(
            "slurm.status.unavailable", job_id=self.job_id, slurm_job_id=slurm_job_id
        )
        return None
