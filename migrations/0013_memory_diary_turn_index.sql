-- Speed memory diary scheduling over recent completed turns.

CREATE INDEX turns_memory_diary_candidates_idx
    ON turns (
        user_message_provider,
        (
            CASE
                WHEN user_message_channel LIKE 'guild:%:channel:%'
                THEN 'guild:' || split_part(user_message_channel, ':', 2)
                ELSE 'global'
            END
        ),
        user_key,
        completed_at
    )
    WHERE status = 'completed' AND completed_at IS NOT NULL;
