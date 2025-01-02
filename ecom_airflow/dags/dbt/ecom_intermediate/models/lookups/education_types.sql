SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['education']) }} AS education_id,
    education AS education_type,
    current_timestamp() AS created_at
FROM {{ source('ecom_staging', 'stg_customers') }}
WHERE education IS NOT NULL