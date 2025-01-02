SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['status']) }} AS status_id,
    status AS status_name,
    CURRENT_TIMESTAMP() AS created_at
FROM {{ source('ecom_staging', 'stg_orders') }}
WHERE status IS NOT NULL