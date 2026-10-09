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

For a GPU task, also put an editable `src/labs/labX.sbatch` in your fork,
replacing `X` with the task number. The SSH worker submits that file to Slurm
after preparing its repository. Use an existing lab batch file as a starting
point. The judge supplies `JUDGE_TASK_ID`, `JUDGE_SUBMISSION_DIR`,
`JUDGE_OUTPUT_DIR`, and `JUDGE_TRUSTED_ROOT`, and overrides resource limits
when calling `sbatch`. The script must leave a valid `result.json` in
`JUDGE_OUTPUT_DIR` by running the trusted task runner. CPU tasks, including
`lab1` today, do not use sbatch.

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
this routes submissions through the SSH worker to Slurm. Each such submission
must include `src/labs/<task_id>.sbatch` at its submitted commit. The master
trusts its own task definitions, while participants can edit their batch file.

### Reusable evaluators

Model scoring lives in `src/judge/evaluators/`, independently of a lab's dataset
and participant module. An `Evaluator` validates its runtime and implements
`evaluate(model_id, dataset) -> JudgeResult`. `ModelEvaluationTask` handles the
participant's `eval_model_id`; its lab subclass supplies `load_dataset()` and
an evaluator. Labs 4 and 5 share `PerplexityEvaluator`, which can also be called
directly with a Hugging Face model ID and a `datasets.Dataset` of `text` rows.

The perplexity evaluator supports causal language models loadable by
`AutoModelForCausalLM` and defaults to the submitted model's tokenizer. Set
`tokenizer_id` to use a fixed tokenizer for a lab. Lab 4 uses
`openai-community/gpt2`; Lab 5 evaluates Llama 3.2 models with the tokenizer
from [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)
and batches of one document with an 8,192-token limit. The generic evaluator
defaults to 1,024 tokens, configurable through `max_length` and capped by the
model's declared context length. Existing tokenizer padding tokens are preserved;
otherwise EOS is used for padding. Padding stays on the right and is excluded
from scoring.

### Lab 5 submission

Shuffle the entire `train` split of `allenai/dolma3_mix-150B-1025` with seed 42
and reserve the last **50,000 documents** before tokenization or packing. Exclude
all of those documents from training. Participants may use at most **10,000**
of them for periodic validation. The OJ evaluates all 50,000, truncating each
document to its first 8,192 tokenizer tokens; it does not pack documents or use
sliding windows. The corpus score weights each non-padding next-token target
equally, with document p90/p99 reported separately.

Train the Llama 3.2 1B architecture from randomly initialized weights with an
8,192-token training context. Upload both weights and tokenizer to Hugging Face.
Fill `eval_model_id`, `training_run_url`, and `training_config` in
`src/labs/lab5.py`. Include optimizer parameter groups, hyperparameters, and
schedules, plus actual non-padding tokens seen (counting repeats), run duration,
GPU allocation, and the number of holdout documents used for validation.

The OJ checks the submitted architecture and RoPE settings against the published
Llama 3.2 1B configuration, requires support for at least 8,192 context tokens,
and checks the uploaded tokenizer's vocabulary and BOS/EOS IDs. It validates the
reported training configuration and requires a run URL in `lab5-training-llama`.
It includes that evidence in judge logs/results for organizer review.

Training provenance is reviewed using the linked run and training code: random
initialization, holdout exclusion, correct token accounting, the assigned token,
H200, and duration limits, and training logs. Those limits are the organizer's
training allocation; the OJ's one-GPU/four-hour resources apply to evaluation.
The specification's three allocation placeholders must be filled by the organizer;
the OJ does not invent or enforce unspecified training limits.

Log `train/loss` as the average over `logging_steps`. Log `train/grad_norm`,
`train/learning_rate`, `train/tokens_per_second`, and `train/total_tokens_seen`
at the reported training step, rather than averaging them over the logging
interval. Log `eval/perplexity` at least once every 10% of planned training steps.
The organizer reviews these metrics and cadence; the OJ does not fetch W&B
training history or treat self-reported configuration as proof of compliance.
Submit a training run before October 9, 2026; the full lab is due October 20, 2026.

Perplexity evaluation requires a CUDA GPU and moves all model parameters,
buffers, and tokenized inputs to `cuda:0`. It verifies logits remain there,
disables the KV cache, and requires
[`torch.compile`](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
with the Inductor backend, `fullgraph=True`, and `dynamic=True`. CUDA absence,
compilation errors, graph breaks, and recompilation-limit exhaustion fail the
job as infrastructure errors instead of silently switching to CPU or eager
execution. The runtime check runs before dataset preparation. Compile-capable
CUDA dependencies and a host C/C++ compiler must be installed on compute nodes.

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
GPU submissions additionally require an `Idempotency-Key`. Successful submission
means the job is persisted locally, not that Slurm has accepted it. The job ID
and W&B run ID remain stable when the same request is retried. Reusing a key
for different submission contents returns HTTP 409. `/agents` routes and
`JUDGE_AGENT_TOKEN` are no longer used.

### Deployment

Compose builds the judge application from this repository; no image publishing
workflow or GHCR judge pull is required. CPU tasks continue using the local
Docker worker and the same locally built image. GPU jobs use Tailscale SSH to
one configured Slurm login node. A separate SSH worker prepares submissions
and monitors them; the remote helper publishes logs and results directly to W&B
using a credential stored on that node.

Copy `.env.example` to `.env` and set the API token, W&B credentials, and remote
configuration. Each new GPU job fetches the latest default-branch version of
`JUDGE_REPO_URL` for its evaluator. No `JUDGE_REVISION` setting is required.
Push evaluator updates to that branch before submitting jobs that should use them.

Required remote settings:

- `JUDGE_SSH_HOST`: the login node's Tailscale hostname or IP.
- `JUDGE_SSH_USER`: its Unix user with Slurm submission permissions.
- `JUDGE_REMOTE_WORK_ROOT`: an absolute shared-storage directory visible to
  the login node and compute nodes, writable by that user.
- `JUDGE_REPO_URL`: trusted HTTPS GitHub repository; defaults to this repository.
- `JUDGE_SLURM_ACCOUNT`: defaults to `ACD115198`.
- `JUDGE_SLURM_GPU_RESOURCE`: defaults to `gpu`.
- `TS_AUTHKEY`: auth key for the Compose Tailscale node, generated with
  **Generate auth key** in the Tailscale admin console. Use the complete
  `tskey-auth-...` value, rather than an API access token.
- `TS_HOSTNAME`: defaults to `study-group-online-judge`.

The official Tailscale image and the CLI copied into the judge image both use
`v1.102.4`. Keep these two versions aligned when upgrading. Tailscale runs in
userspace mode without `/dev/net/tun` or network
capabilities. Its identity persists in `tailscale-state`. Only the SSH worker
mounts its socket; the sidecar's operator user matches the worker UID `10001`.
The worker uses `tailscale ssh`, which authenticates through tailnet identity
and verifies the destination's advertised SSH host key. No private SSH key is
mounted. See [Tailscale SSH](https://tailscale.com/docs/features/tailscale-ssh)
and [container configuration](https://tailscale.com/docs/features/containers/docker/docker-params).

Enable Tailscale SSH on the remote login node. Configure both a network grant
to TCP port 22 and an SSH `accept` rule for the Compose node's identity and the
specific remote Unix user. Automated connections cannot use interactive
`check` rules. If the auth key assigns a tag, the destination must also be
appropriately tagged. Tailnet configuration and remote installation are
administrator prerequisites; Compose does not change them.

The remote node needs Bash, Git, `flock`, uv (with Python 3.14 available or
uv-managed Python downloads enabled), `sbatch`, and `sacct` on the noninteractive
SSH PATH. Compute nodes need the `cuda/13.0` module used by the batch templates.
Setup runs on the login node before submission; evaluation runs in Slurm.
Dependencies and GitHub must be reachable from the login node.

The local CPU `JUDGE_WORK_ROOT` must be an absolute host directory mounted at
the same path in the worker and evaluator containers. Prepare it and find the
Docker socket group:

```console
sudo mkdir -p /var/lib/study-group-online-judge/work
sudo chown 10001:10001 /var/lib/study-group-online-judge/work
stat -c '%g' /var/run/docker.sock
```

Set `DOCKER_GID` to that group (`999` in the example is only a placeholder).
Then, from the committed source checkout:

```console
openssl rand -hex 32 # use as JUDGE_API_TOKEN in .env
docker compose up -d --build
docker compose ps
docker compose logs --follow api worker ssh-worker tailscale
```

GPU execution diagnostics are enabled at INFO level. Follow job receipt in
`api` and execution in `ssh-worker` with:

```console
docker compose logs --follow api ssh-worker
```

Each event includes the judge `job_id`; scheduler events also include the
`slurm_job_id` once assigned. Logs cover worker dispatch, SSH destination and
operation, elapsed time, retries, remote setup, `sbatch` commands/responses,
`sacct` commands/responses (the current status poller), accounting delays,
terminal results, and pending W&B reporting. Remote diagnostics use stderr
and are forwarded into the SSH worker logs; stdout remains the JSON snapshot.
Captured diagnostics are limited to 8 KiB per field and configured credentials
are redacted. SSH request bodies and environment dumps are not logged. Detailed
dependency setup output remains in the remote job's `output/setup.log`.

Compose does not publish the API on a host port. Configure Dokploy or your
reverse proxy to reach the `api` service on port `8000`. SQLite and W&B files
persist in `judge-data`; local CPU caches use the HF and uv cache volumes.
The SSH worker has neither W&B credentials nor the Docker socket. Remote
commands receive job configuration, never the submission token, Tailscale
auth key, or the local W&B key. The local `WANDB_API_KEY` remains necessary for
CPU reporting and API leaderboard reads.

Create `<JUDGE_REMOTE_WORK_ROOT>/.netrc` manually on the remote login node:

```text
machine api.wandb.ai
  login judge
  password YOUR_REMOTE_WANDB_API_KEY
```

Make it readable by `JUDGE_SSH_USER` and set its permissions to `600`. The
remote helper reads that exact file without prompting or copying its key into
job records or submission environments. GPU runs use `WANDB_ENTITY` and
`WANDB_PROJECT` captured when the API queues each job. The remote login node
must be able to reach W&B. Reporting failures retry independently, including
after execution has completed; they do not submit another Slurm job. Trusted
bootstrap failures use a packaged fallback through `uv run --no-project --with
wandb` so they can be reported even when the evaluator environment is absent.

### Remote repository setup and recovery

The trusted `src/judge/setup-repo.sh` accepts repository URL, an exact commit SHA
or `latest`, and absolute destination. It locks the destination, clones when absent,
fetches when present, rejects mismatched origins or dirty checkouts, checks
out the requested commit (the remote default-branch tip for `latest`), then runs
`uv sync --no-sources --no-dev`. Participant checkouts always use the exact
submitted SHA. Ignoring the
repository's Linux CPU-only PyTorch source resolves CUDA dependencies. The
script restores `uv.lock` afterward so retries do not mistake setup changes
for participant changes. Runtime caches live under the remote work root.

Trusted evaluators live in `<work-root>/jobs/<job-id>/judge`. Each new job fetches
the latest OJ code and prepares its environment there. A durable `judge.ready`
marker lets reconnects reuse that job's prepared evaluator without changing
running jobs when upstream advances. Seven-day workspace cleanup also removes
these evaluator checkouts. Each job has
`<work-root>/jobs/<job-id>/submission`, `output/setup.log`, `output/slurm.log`,
and `output/result.json`. W&B SDK and reporting diagnostics are retained under
`output/wandb` and `output/reporting.log`. GPU batch scripts use the prepared participant
virtualenv with `PYTHONPATH` pointing at the trusted evaluator source.
The worker supplies the `JUDGE_*` runner variables and overrides Slurm resource
limits. Existing `dev`/`8gpus` partition selection and Nano4 GPU limits remain.

The remote one-shot helper locks each job and persists its record under
`<work-root>/job-records/<job-id>.json`, with a copy in the workspace's
`submission.json`, before and after invoking `sbatch`. Repeated polls reuse its Slurm ID. A restart or
SSH disconnect resumes preparation or monitoring. Two setup slots and four
monitoring slots prevent dependency installation from blocking running jobs;
monitoring repeats every 30 seconds. Logs are read and uploaded on the remote
node; they are not repeatedly transferred to the local host. Remote report
offsets are durable, while SQLite stores execution status, results, W&B URLs,
and pending-report recovery state. An interrupted upload can replay its last
batch into the same W&B run.

W&B runs remain running during setup, queueing, and execution, including idle
monitoring polls. Each short-lived SSH helper resumes the same run ID and flushes
its SDK session with `x_update_finish_state=False`, preserving console uploads
and durable reporting cursors without marking the server run finished. Only
after the terminal report and all logs upload successfully does a final session
mark the run finished (or failed for infrastructure errors). Reporting failures
leave the run available for retry. CPU workers keep their run open until execution
completes.

Each invocation of `setup-repo.sh` with `JUDGE_REMOTE_WORK_ROOT` set removes
finished, successfully reported job workspaces older than seven days, measured
from their completion time. It skips active or locked jobs, unresolved Slurm
submissions, and jobs whose W&B uploads are pending. Cleanup runs when setup
runs, rather than on an idle timer. Small durable job records and locks remain
outside the deleted workspaces, preventing delayed reconnects from resubmitting
retired jobs. Shared caches and `.netrc` are kept. Older evaluator checkouts
under the former `trusted/<judge-revision>` layout are left untouched.

A crash between `sbatch` and recording its response is deliberately treated as
ambiguous. The job reports an infrastructure error and will not resubmit. Check
Slurm using the unique job name `judge-<job-id>` and its remote `submission.json`
record before making a new submission with a new idempotency key. Never retry
an ambiguous job without confirming whether Slurm accepted it.

Migration 005 removes agent registrations and assignment metadata while
retaining completed jobs, Slurm IDs, and reporting history. Back up `judge.db`
with SQLite's backup API before deploying; rollback requires that backup.
Migration 006 adds recovery state for reports published on the remote node.
When upgrading the Compose stack, remove the old local `remote-reporter`
container (`docker compose up -d --build --remove-orphans`) to avoid duplicate
local and remote W&B publishers.

### Slack leaderboard notifications

Notifications use the lab names in `ui/src/lab-names.json` (override the server
file location with `JUDGE_LAB_NAMES_PATH`). The PNG card is uploaded directly to
Slack and shared inline alongside compact ranking text. Podium entries use one
medal emoji each; there are no oversized duplicate medal image accessories.

To enable inline image notifications:

1. Open the Slack app at <https://api.slack.com/apps>.
2. Under **OAuth & Permissions → Bot Token Scopes**, add **`files:write`**.
3. Install or reinstall the app to your workspace, then copy its **Bot User OAuth
   Token** (`xoxb-...`). Invite the app to the notification channel.
4. In the channel's details, copy the **Channel ID**. Set
   `JUDGE_SLACK_BOT_TOKEN` and `JUDGE_SLACK_CHANNEL_ID` in the deployment's `.env`.
5. Set `JUDGE_SLACK_DAILY_TIME=19:00` and `JUDGE_SLACK_TIMEZONE=Asia/Taipei`.
6. After deploying this code, rebuild and recreate the API:
   `docker compose up -d --build --force-recreate api`.

Only `files:write` is required: the API gets a Slack upload URL, sends the PNG
bytes, then calls `files.completeUploadExternal` with the destination channel
and ranking blocks. No public image server or `JUDGE_PUBLIC_URL` is needed.
The bot upload configuration takes precedence over `JUDGE_SLACK_WEBHOOK_URL`.
The running API detects records and schedules the daily summary itself.

An existing incoming app webhook or webhook-triggered Workflow Builder workflow
can still be used for **text-only** notifications when bot upload is unset.
Workflow Builder receives its existing single `text` variable for the Send a
message step. A partially configured bot upload is an error and retries rather
than silently dropping its image. Upload or sharing failures retain the pending
notification for retry; it is acknowledged only after Slack confirms the share.

After the initial baseline, SOTA alerts are sent only for a strictly better
all-time score (respecting minimize/maximize), or a new earliest passing submission.
Ties and changes below first place stay quiet. Persisted records survive restarts
and are not lowered when a run disappears. Messages start with
`🧪 New record by {username}!` and list the current top three as `🥇 username: score`.
Failed Slack deliveries retry on the next successful source refresh.

Daily messages start with `🧪 Daily Updates` and include **every ranked participant
in every lab**, with separate messages per lab and pages of 20 participants.
The default schedule is **19:00 Asia/Taipei**; configure `JUDGE_SLACK_DAILY_TIME`
(`HH:MM`) and `JUDGE_SLACK_TIMEZONE` (IANA timezone) to change it. Delivery happens
on the first successful leaderboard refresh at or after that time. Daily delivery
is acknowledged per lab and date, saved across restarts, and retried on failure.
Partially delivered labs resume from their next page using the saved standings,
so earlier pages are not repeated or reordered when new submissions arrive.
Messages are paced to Slack's webhook limits and honor `Retry-After` backoff.
If the service was offline on previous days, it sends only today's summary.

PNG cards are Chromium screenshots of the actual React leaderboard, using its
CSS, fonts, GitHub avatars, scores, attempt counts, and Taipei submission times.
The UI receives the same frozen snapshot as the message. Images are saved in
`leaderboard-images/` next to the database before their bytes are uploaded to
Slack. The Docker image includes the built UI and Chromium. For a Python-only
installation, build `ui/` with `npm ci --prefix ui` and `npm run build --prefix ui`,
then run `playwright install --with-deps --only-shell chromium`. Set `JUDGE_UI_DIST`
to the built UI directory if needed, and `JUDGE_CHROMIUM_EXECUTABLE` to use an
existing Chromium installation.

The OJ frontend opens the lab with the highest lab number by default and keeps
the user's selected lab during refreshes.

The API container also needs `WANDB_API_KEY`: workers publish submissions, while
`api` independently reads W&B for the leaderboard. Compose's `.env` supplies
interpolation values; it does not automatically inject every variable into every
container. After updating `compose.yaml` or `.env`, recreate the API with
`docker compose up -d --force-recreate api` so it receives the new environment.
