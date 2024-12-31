SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['marital_status']) }} AS marital_status_id,
    marital_status AS status_type,
    CURRENT_TIMESTAMP() AS created_at
FROM {{ source('ecom_staging', 'stg_customers') }}
WHERE marital_status IS NOT NULL