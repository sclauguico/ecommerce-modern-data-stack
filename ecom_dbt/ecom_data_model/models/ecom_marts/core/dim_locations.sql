{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

SELECT
    location_id,
    city,
    state,
    country,
FROM {{ source('ecom_intermediate', 'locations') }} l
LEFT JOIN {{ source('ecom_intermediate', 'customers_enriched') }} c USING (location_id)
LEFT JOIN {{ source('ecom_intermediate', 'orders') }} o USING (customer_id)
GROUP BY 1, 2, 3, 4