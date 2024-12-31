SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['payment_method']) }} AS payment_method_id,
    payment_method AS method_name,
    CURRENT_TIMESTAMP() AS created_at
FROM {{ source('ecom_staging', 'stg_orders') }}
WHERE payment_method IS NOT NULL