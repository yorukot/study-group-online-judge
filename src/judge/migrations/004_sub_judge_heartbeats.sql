ALTER TABLE sub_judges ADD COLUMN last_seen_at TEXT;
UPDATE sub_judges SET last_seen_at = registered_at;

CREATE INDEX sub_judges_last_seen_at ON sub_judges (last_seen_at);
