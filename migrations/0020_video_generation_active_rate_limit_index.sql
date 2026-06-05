DROP INDEX IF EXISTS video_jobs_successful_completed_idx;

CREATE INDEX video_jobs_active_rate_limit_idx
    ON video_jobs (
        (
            CASE
                WHEN status = 'pending' THEN submitted_at
                ELSE completed_at
            END
        ) DESC,
        turn_id
    )
    WHERE status = 'pending'
       OR (status = 'done' AND output_uri IS NOT NULL AND completed_at IS NOT NULL);
