import os
import re
import shutil
import subprocess
from pathlib import Path

JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SECRET_ENVIRONMENT_VARIABLES = {
    "GITHUB_TOKEN",
    "JUDGE_API_TOKEN",
    "WANDB_API_KEY",
    "TS_AUTHKEY",
    "TS_CLIENT_SECRET",
}


class CheckoutError(RuntimeError):
    pass


def create_job_workspace(work_root: Path, job_id: str) -> Path:
    """Create a new workspace for an internally generated job ID."""

    if JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError("job_id must be a 32-character lowercase hexadecimal value")

    work_root.mkdir(parents=True, exist_ok=True)
    workspace = work_root / job_id
    workspace.mkdir()
    return workspace


def checkout_repository(
    repo_url: str,
    commit_sha: str,
    destination: Path,
    *,
    timeout_seconds: int = 120,
) -> Path:
    """Clone a repository and check out exactly one submitted commit."""

    if COMMIT_SHA_PATTERN.fullmatch(commit_sha) is None:
        raise ValueError(
            "commit_sha must be a 40-character lowercase hexadecimal value"
        )
    if destination.exists():
        raise FileExistsError(f"Checkout destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run_git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--no-tags",
                "--config",
                f"core.hooksPath={os.devnull}",
                "--",
                repo_url,
                str(destination),
            ],
            timeout_seconds=timeout_seconds,
        )
        _run_git(
            ["checkout", "--detach", commit_sha],
            workdir=destination,
            timeout_seconds=timeout_seconds,
        )
        resolved_commit = _run_git(
            ["rev-parse", "--verify", "HEAD^{commit}"],
            workdir=destination,
            timeout_seconds=timeout_seconds,
        ).stdout.strip()
        if resolved_commit != commit_sha:
            raise CheckoutError(
                f"Checked out {resolved_commit}, expected submitted commit {commit_sha}"
            )
    except subprocess.CalledProcessError as error:
        _remove_failed_checkout(destination)
        detail = error.stderr.strip() or "git command failed without an error message"
        raise CheckoutError(detail) from error
    except subprocess.TimeoutExpired as error:
        _remove_failed_checkout(destination)
        raise CheckoutError("Git checkout timed out") from error
    except Exception:
        _remove_failed_checkout(destination)
        raise

    return destination


def _run_git(
    arguments: list[str],
    *,
    workdir: Path | None = None,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for variable in SECRET_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )

    return subprocess.run(
        ["git", *arguments],
        cwd=workdir,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _remove_failed_checkout(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
