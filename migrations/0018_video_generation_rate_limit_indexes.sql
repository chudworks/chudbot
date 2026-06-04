CREATE INDEX video_jobs_successful_completed_idx
    ON video_jobs (completed_at DESC, turn_id)
    WHERE status = 'done' AND output_uri IS NOT NULL;

CREATE INDEX turns_video_rate_limit_user_idx
    ON turns (user_message_provider, user_key, id);
