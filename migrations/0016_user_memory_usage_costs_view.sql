-- Include both diary generation and compact profile generation costs.

CREATE OR REPLACE VIEW user_memory_usage_costs AS
SELECT source.message_provider,
       source.scope_key,
       source.subject_user_key,
       source.agent_name,
       source.llm_provider,
       source.llm_model,
       usage_records.usage_record ->> 'provider'::text AS usage_provider,
       usage_records.usage_record ->> 'model'::text AS usage_model,
       (usage_records.usage_record -> 'subject'::text) ->> 'kind'::text AS usage_subject,
       (usage_records.usage_record -> 'cost'::text) ->> 'unit'::text AS cost_unit,
       (usage_records.usage_record -> 'cost'::text) ->> 'estimated'::text AS cost_estimated,
       CASE (usage_records.usage_record -> 'cost'::text) ->> 'unit'::text
           WHEN 'usd_ticks'::text THEN
               (((usage_records.usage_record -> 'cost'::text) ->> 'amount'::text)::numeric)
               / '10000000000'::bigint::numeric
           WHEN 'usd'::text THEN ((usage_records.usage_record -> 'cost'::text) ->> 'amount'::text)::numeric
           ELSE NULL::numeric
       END AS cost_usd,
       source.created_at
  FROM (
        SELECT de.message_provider,
               de.scope_key,
               de.subject_user_key,
               de.agent_name,
               de.llm_provider,
               de.llm_model,
               de.usage,
               de.created_at
          FROM user_memory_diary_entries de
        UNION ALL
        SELECT dv.message_provider,
               dv.scope_key,
               dv.subject_user_key,
               dv.agent_name,
               dv.llm_provider,
               dv.llm_model,
               dv.usage,
               dv.created_at
          FROM user_memory_document_versions dv
       ) source
       CROSS JOIN LATERAL jsonb_array_elements(source.usage) usage_records(usage_record)
 WHERE usage_records.usage_record ? 'cost'::text
   AND (usage_records.usage_record -> 'cost'::text) IS NOT NULL;
