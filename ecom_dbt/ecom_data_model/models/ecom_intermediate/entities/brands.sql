WITH cleaned_brands AS (
    SELECT DISTINCT
        TRIM(brand) as brand_name
    FROM {{ source('ecom_staging', 'stg_products') }}
    WHERE brand IS NOT NULL
    AND TRIM(brand) != ''
)

SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['brand_name']) }} AS brand_id,
    brand_name,
    current_timestamp() AS created_at
FROM cleaned_brands