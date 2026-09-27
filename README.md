# Study Group Online Judge

A small online judge for machine-learning study groups. Participants submit a
commit from their fork, and the judge evaluates it with its own task tests.

## Participants

### Submission flow

1. Fork `cerulean-works/study-group-online-judge` into your GitHub account.
2. Implement the task in your fork and push your changes to `main`.
3. In your fork, open **Actions** → **Submit to study group judge** → **Run
   workflow**, select the task, and run it on `main`.

Your fork needs an Actions secret named `JUDGE_API_TOKEN` and a repository
variable named `WANDB_PROJECT_URL`. Ask the study-group organizer for the token
and project URL if they are not already configured. The workflow submits the
exact commit you selected and succeeds once the judge queues it; this does not
mean the submission passed.

### Where to implement

Put each implementation in the `src/labs/` file specified by its task. The
judge checks out the submitted commit, then its task loads that file directly
as a Python module and calls the required function. For example, `lab1` loads
`src/labs/lab1.py` and calls `gpt2_complete`; it checks the completions and
logits against GPT-2 Small using 20 Tiny Shakespeare prompts.

For `lab2`, implement `mmlu_eval()` in `src/labs/lab2.py` and include
`src/labs/lab2.sbatch`. The function returns an A/B/C/D prediction for every
test row in the `all` configuration of
[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), pinned to revision
`c30699e8356da336a370243923dbaf21066bb9fe`. Use the first four `dev`
rows of the same subject as exemplars and truncate prompts to the final 1024
GPT-2 tokens. Score the next-token logits of the plain `A`, `B`, `C`, and `D`
tokens after `Answer: `. The result key is the SHA-256 of this exact compact
UTF-8 JSON encoding, where `index` is the zero-based row number in the pinned
`all` test split:

```python
payload = {
    "index": index,
    "subject": row["subject"],
    "question": row["question"],
    "choices": row["choices"],
}
key = hashlib.sha256(
    json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()
```

The row index is necessary because some MMLU questions, including some with
identical subjects and choices, repeat. The H200 judge evaluates every test
question against GPT-2 Small. It reports overall agreement and per-subject
mismatch row numbers and hash prefixes to W&B, not one W&B row per question.
Lab 2 passes when at least 97% of the predictions match and has no scoreboard.
Its W&B `score` is the fraction matching the GPT-2 reference, and
`samples_passed` is the corresponding question count. `mmlu_accuracy` is the
fraction of participant predictions matching MMLU's answer labels, not the
fraction matching the reference.

For a GPU task, also put an editable `src/labs/labX.sbatch` in your fork,
replacing `X` with the task number. The Nano4 sub-judge submits that file to
Slurm. Use [the template](src/labs/labX.sbatch) as a starting point. The judge
supplies `JUDGE_TASK_ID`, `JUDGE_SUBMISSION_DIR`, `JUDGE_OUTPUT_DIR`, and
`JUDGE_TRUSTED_ROOT`, and overrides resource limits when calling `sbatch`.
The script must leave a valid `result.json` in `JUDGE_OUTPUT_DIR` by running
the trusted task runner. CPU tasks, including `lab1` today, do not use sbatch.

Keep your implementation in your fork. Changing the judge's task code in your
fork does not change the evaluation used by the deployed judge.

### View results

The submission workflow prints the queued job ID and a link to its W&B run in
the run summary. The link becomes active when the judge starts reporting. Open
it for progress, logs, metrics, and the final pass/fail result.

## Contributors

### Defining tasks

Each task is a `Task` subclass in `src/judge/tasks/`. It declares resource
limits and evaluates the checked-out participant submission:

```python
from pathlib import Path

from judge.models import JudgeResult, Resources
from judge.tasks.base import Task


class Assignment01(Task):
    id = "assignment-01"
    resources = Resources(cpus=2, memory_gb=4, timeout_seconds=60)

    def evaluate(self, submission: Path) -> JudgeResult:
        # Import the participant implementation and evaluate it here.
        ...
```

Register the task instance in the `TASKS` dictionary in
`src/judge/tasks/__init__.py`. Add its ID to the `task_id` choices in
`.github/workflows/submit.yaml` so participants can select it. Return
`JudgeResult(passed=...)` for correctness tasks, or include `score` and
`metrics` for benchmarks. Set `gpus` in `Resources` when a task needs a GPU;
this routes submissions to a registered Slurm sub-judge. Each such submission
must include `src/labs/<task_id>.sbatch` at its submitted commit. The master
trusts its own task definitions, while participants can edit their batch file.

### Development

```console
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python -m pytest -q tests
```

The API exposes `GET /healthz`, `POST /submissions`, and `GET /jobs/{job_id}`.
Submission and job routes require `Authorization: Bearer <JUDGE_API_TOKEN>`.
GPU submissions additionally require an `Idempotency-Key`. The `/agents`
registration, long-poll, receipt, and event routes use a separate
`JUDGE_AGENT_TOKEN`. The master does not poll agents for health: it schedules
only to an agent with an active long-poll and returns an error if the offer is
not acknowledged. Keep the master API to one process because its pending
long-polls are held in memory; the job and report records are in SQLite. The
published master image embeds its source commit; a GPU agent must register
with the same trusted judge checkout revision to receive work.

### Deployment

The master requires Docker with Compose, access to the GHCR judge image, a
W&B project and API key, and a host work directory writable by the judge
container's UID/GID `10001`. CPU tasks run in the local Docker worker. GPU
tasks run on a registered bare-metal Slurm sub-judge, not in Docker.

Copy `.env.example` to `.env` and set its required values. `JUDGE_WORK_ROOT`
must be an absolute host path, mounted at the same path in the worker and its
evaluator containers. For example:

```console
sudo mkdir -p /var/lib/study-group-online-judge/work
sudo chown 10001:10001 /var/lib/study-group-online-judge/work
stat -c '%g' /var/run/docker.sock
```

Set `DOCKER_GID` to the reported Docker socket group. The `999` in
`.env.example` is an example socket group, not the judge user's UID or primary
GID. Docker Desktop and OrbStack commonly report group `0`. Generate separate
submission and agent tokens, put both in `.env`, then start the services:

```console
openssl rand -hex 32 # JUDGE_API_TOKEN
openssl rand -hex 32 # JUDGE_AGENT_TOKEN
docker compose up -d
docker compose ps
```

Compose does not publish an API port on the host. The `api` container listens
on port `8000` internally; configure Dokploy to route to that service and
port. SQLite and local W&B files live in `judge-data`. The `remote-reporter`
service publishes remote progress, results, and failure reasons to W&B using
the master's API key; Nano4 does not need a W&B key. Checked-out CPU
repositories and result files live under `JUDGE_WORK_ROOT`. The HF and uv
cache volumes persist local-worker downloads across submissions and restarts.

```console
docker compose logs --follow api worker remote-reporter
docker compose restart worker
docker compose down
```

### Nano4 Slurm sub-judge

The sub-judge runs on the Nano4 login node and makes outbound HTTPS requests
to the master. It does not run participant code on the login node: after
checking out the exact submitted commit, it submits the participant's
`src/labs/<task_id>.sbatch` to Slurm. Put its checkout, agent ledger, job
workspaces, HF cache, and uv cache on shared `/work` storage visible to the
H200 compute nodes. The agent needs Python 3.14, `pydantic`, Git, and `uv`;
the compute job installs its Python environment inside the Slurm allocation.
For example, from a trusted judge checkout on Nano4:

```console
uv venv --python 3.14 /work/$USER/study-group-oj-agent-venv
uv pip install --python /work/$USER/study-group-oj-agent-venv/bin/python 'pydantic>=2.13.5'
export PYTHONPATH=/work/$USER/study-group-online-judge/src
export JUDGE_TRUSTED_ROOT=/work/$USER/study-group-online-judge
export JUDGE_WORK_ROOT=/work/$USER/study-group-oj
export JUDGE_HF_HOME=/work/$USER/study-group-oj/hf-cache
export JUDGE_UV_CACHE_DIR=/work/$USER/study-group-oj/uv-cache
export JUDGE_MASTER_URL=https://oj.cerulean.works
export JUDGE_AGENT_ID=nano4
export JUDGE_AGENT_TOKEN='paste-the-master-agent-token-here'
export JUDGE_TASK_IDS='lab2,lab3'
export JUDGE_MAX_GPUS=8
export JUDGE_REVISION="$(git -C "$JUDGE_TRUSTED_ROOT" rev-parse HEAD)"
/work/$USER/study-group-oj-agent-venv/bin/python -m judge.slurm_agent
```

Run the agent under a persistent user service or session. Do not put the
agent token in a participant repository or sbatch file. `JUDGE_SLURM_ACCOUNT`
defaults to `ACD115198`; `JUDGE_SLURM_GPU_RESOURCE` defaults to `gpu` and can
be changed after checking Nano4's actual H200 GRES name. The executor requests
`dev` for jobs up to 4 hours, otherwise `8gpus` up to 48 hours, and enforces
12 CPU cores and 200 GiB per requested GPU. The sample batch file loads
`cuda/13.0`, then resolves dependencies on the compute node with
`uv sync --no-sources --no-dev` to ignore the repository's Linux CPU-only
PyTorch source while resolving dependencies. A Nano4 development run verified
the generic `--gres=gpu:1` request, CUDA 13.0, and CUDA-enabled PyTorch on H200.

The agent records accepted Slurm IDs and pending reports in
`JUDGE_WORK_ROOT/agent.db`. It sends ordered, retryable events to the master;
the master stores them before W&B publication. If the agent crashes between
invoking `sbatch` and recording its response, the outcome is ambiguous and a
retry is rejected for manual reconciliation rather than risking two GPU jobs.
There are no periodic agent health checks; a scheduling request fails if no
agent is actively polling or if its offer is not acknowledged. A temporary
Nano4 development master and sub-judge completed a Lab 2 GPU submission; the
production master-to-Nano4 deployment remains unvalidated.
