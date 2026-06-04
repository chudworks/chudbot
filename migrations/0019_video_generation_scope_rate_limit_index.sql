CREATE INDEX turns_video_rate_limit_channel_idx
    ON turns (user_message_provider, user_message_channel, id);
