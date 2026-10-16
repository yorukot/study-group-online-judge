#!/usr/bin/env bash
# Trusted setup used for both evaluator and participant checkouts.
set -euo pipefail

repo_url="${1:?repository URL is required}"
revision="${2:?exact commit or latest is required}"
destination="${3:?destination is required}"
[[ "$revision" == latest || "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit SHA' >&2; exit 2; }
[[ "$destination" == /* && "$destination" != / ]] || { echo 'Destination must be absolute' >&2; exit 2; }
export GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
unset JUDGE_API_TOKEN WANDB_API_KEY GITHUB_TOKEN TS_AUTHKEY TS_CLIENT_SECRET

# Cleanup is triggered by setup, not a background timer. Only remotely reported
# terminal jobs receive this marker; durable records and locks live outside jobs.
if [[ -n "${JUDGE_REMOTE_WORK_ROOT:-}" ]]; then
  work_root="$JUDGE_REMOTE_WORK_ROOT"
  [[ "$work_root" == /* && "$work_root" != / && ! -L "$work_root/jobs" ]] || { echo 'Invalid remote cleanup root' >&2; exit 2; }
  if [[ -d "$work_root/jobs" ]]; then
    mkdir -p "$work_root/job-locks"
    while IFS= read -r -d '' marker; do
      job_directory="$(dirname "$marker")"
      job_id="$(basename "$job_directory")"
      [[ "$job_id" =~ ^[0-9a-f]{32}$ && ! -L "$job_directory" && ! -L "$marker" ]] || continue
      [[ -f "$work_root/job-records/$job_id.json" && ! -L "$work_root/job-records/$job_id.json" ]] || continue
      [[ "$destination" != "$job_directory" && "$destination" != "$job_directory/"* ]] || continue
      exec 8>"$work_root/job-locks/$job_id.lock"
      if flock -n 8; then
        if [[ -f "$marker" ]] && [[ -n "$(find "$marker" -mmin +10080 -print)" ]]; then
          rm -rf -- "$job_directory"
          echo "[judge] removed expired job $job_id"
        fi
      fi
      exec 8>&-
    done < <(find "$work_root/jobs" -mindepth 2 -maxdepth 2 -type f -name .finished -mmin +10080 -print0)
  fi
fi

mkdir -p "$(dirname "$destination")"
exec 9>"${destination}.setup.lock"
flock 9
echo '[judge] preparing repository'

checkout_revision() {
  local checkout="$1" ref="$revision" commit
  [[ "$ref" != latest ]] || ref=HEAD
  git -C "$checkout" -c core.hooksPath=/dev/null fetch --no-tags origin "$ref"
  commit="$(git -C "$checkout" rev-parse 'FETCH_HEAD^{commit}')"
  [[ "$revision" == latest || "$commit" == "$revision" ]] || { echo 'Commit mismatch' >&2; exit 2; }
  git -C "$checkout" -c core.hooksPath=/dev/null checkout --detach "$commit"
}

if [[ ! -e "$destination" ]]; then
  temporary="$(mktemp -d "${destination}.clone.XXXXXX")"
  trap 'rm -rf -- "$temporary"' EXIT
  git clone --filter=blob:none --no-checkout --no-tags --config core.hooksPath=/dev/null -- "$repo_url" "$temporary"
  checkout_revision "$temporary"
  mv -- "$temporary" "$destination"
  trap - EXIT
fi
[[ -d "$destination/.git" && ! -L "$destination" ]] || { echo 'Expected a repository directory' >&2; exit 2; }
cd "$destination"
[[ "$(git remote get-url origin)" == "$repo_url" ]] || { echo 'Repository origin mismatch' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo 'Existing checkout is dirty' >&2; exit 2; }
# Each job gets the latest evaluator at preparation time. Reconnects use that
# same job's prepared environment, even if the upstream default branch advances.
if [[ "$revision" == latest && "${JUDGE_SETUP_ONCE:-}" == 1 && -f "${destination}.ready" && -x .venv/bin/python ]]; then
  [[ "$(cat "${destination}.ready")" == "$(git rev-parse 'HEAD^{commit}')" ]] || { echo 'Prepared evaluator commit mismatch' >&2; exit 2; }
  echo '[judge] repository ready'
  exit 0
fi
checkout_revision "$destination"

# Ignoring the Linux CPU-only source can update uv.lock. Keep the submitted
# checkout intact while retaining its resolved CUDA environment in .venv.
lock_backup="$(mktemp)"
runtime_constraints=""
restore_lock=0
if [[ -f uv.lock ]]; then
  cp uv.lock "$lock_backup"
  restore_lock=1
fi
cleanup_setup() {
  local setup_exit_code=$?
  if [[ "$restore_lock" == 1 ]]; then
    cp "$lock_backup" uv.lock
  else
    rm -f uv.lock
  fi
  rm -f "$lock_backup"
  [[ -z "$runtime_constraints" ]] || rm -f "$runtime_constraints"
  exit "$setup_exit_code"
}
trap cleanup_setup EXIT
uv sync --no-sources --no-dev
# Slurm runs the trusted evaluator with the participant's Python environment.
# Resolve the whole evaluator stack together; installing only kernels can
# upgrade huggingface-hub while retaining an incompatible old Transformers.
if [[ "${JUDGE_TASK_ID:-}" == lab5 ]]; then
  if [[ -z "${JUDGE_TRUSTED_ROOT:-}" ]]; then
    echo 'trusted evaluator directory is required' >&2
    exit 2
  fi
  trusted_root="$JUDGE_TRUSTED_ROOT"
  runtime_constraints="$(mktemp)"
  uv pip freeze --python "$trusted_root/.venv/bin/python" --exclude-editable \
    --exclude study-group-online-judge >"$runtime_constraints"
  echo '[judge] installing current evaluator runtime into submission environment'
  uv pip install --python .venv/bin/python --requirements "$trusted_root/pyproject.toml" \
    --constraints "$runtime_constraints"
  uv pip check --python .venv/bin/python
  PYTHONPATH="$trusted_root/src" .venv/bin/python "$trusted_root/src/judge/check_runtime.py"
fi
if [[ "$revision" == latest && "${JUDGE_SETUP_ONCE:-}" == 1 ]]; then
  git rev-parse 'HEAD^{commit}' >"${destination}.ready.tmp"
  mv -- "${destination}.ready.tmp" "${destination}.ready"
fi
echo '[judge] repository ready'
