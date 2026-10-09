ALTER TABLE jobs ADD COLUMN execution_backend TEXT NOT NULL DEFAULT 'docker';
UPDATE jobs SET execution_backend = 'ssh_slurm' WHERE assigned_judge_id IS NOT NULL;

CREATE TABLE ssh_jobs (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id),
    config_json TEXT NOT NULL,
    resources_json TEXT NOT NULL,
    setup_offset INTEGER NOT NULL DEFAULT 0,
    log_offset INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX jobs_backend_status_created_at
ON jobs (execution_backend, status, created_at);

CREATE TEMP TABLE require_drained_agents (pending INTEGER CHECK (pending = 0));
INSERT INTO require_drained_agents
SELECT COUNT(*) FROM jobs WHERE assigned_judge_id IS NOT NULL
AND status IN ('queued', 'dispatching', 'running');
DROP TABLE require_drained_agents;
DROP INDEX jobs_assignment_status_created_at;
ALTER TABLE jobs DROP COLUMN assigned_judge_id;
DROP TABLE sub_judges;
