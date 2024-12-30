WITH shipping_addresses AS (
    SELECT DISTINCT
        TRIM(SPLIT_PART(shipping_address, ',', -3)) AS city,
        TRIM(SPLIT_PART(shipping_address, ',', -2)) AS state,
        TRIM(SPLIT_PART(shipping_address, ',', -1)) AS country
    FROM {{ source('staging', 'stg_orders') }}
    WHERE shipping_address IS NOT NULL
),

billing_addresses AS (
    SELECT DISTINCT
        TRIM(SPLIT_PART(billing_address, ',', -3)) AS city,
        TRIM(SPLIT_PART(billing_address, ',', -2)) AS state,
        TRIM(SPLIT_PART(billing_address, ',', -1)) AS country
    FROM {{ source('staging', 'stg_orders') }}
    WHERE billing_address IS NOT NULL
),

customer_addresses AS (
    SELECT DISTINCT
        TRIM(city) as city,
        TRIM(state) as state,
        TRIM(country) as country
    FROM {{ source('staging', 'stg_customers') }}
    WHERE city IS NOT NULL 
    AND state IS NOT NULL 
    AND country IS NOT NULL
),

all_locations AS (
    SELECT * FROM shipping_addresses
    UNION DISTINCT
    SELECT * FROM billing_addresses
    UNION DISTINCT
    SELECT * FROM customer_addresses
),

cleaned_locations AS (
    SELECT DISTINCT
        city,
        state,
        country
    FROM all_locations
    WHERE city != ''
    AND state != ''
    AND country != ''
    AND city IS NOT NULL
    AND state IS NOT NULL
    AND country IS NOT NULL
)

SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['city', 'state', 'country']) }} AS location_id,
    city,
    state,
    country,
    CURRENT_TIMESTAMP() AS created_at
FROM cleaned_locations