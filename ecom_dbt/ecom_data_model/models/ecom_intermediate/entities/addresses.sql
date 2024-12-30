WITH parsed_addresses AS (
    SELECT DISTINCT
        shipping_address AS address,
        TRIM(SPLIT_PART(shipping_address, ',', 1)) AS street_address,
        TRIM(SPLIT_PART(shipping_address, ',', -4)) AS postal_code,
        TRIM(SPLIT_PART(shipping_address, ',', -3)) AS city,
        TRIM(SPLIT_PART(shipping_address, ',', -2)) AS state,
        TRIM(SPLIT_PART(shipping_address, ',', -1)) AS country
    FROM {{ source('staging', 'stg_orders') }}
    WHERE shipping_address IS NOT NULL
    
    UNION DISTINCT
    
    SELECT DISTINCT
        billing_address AS address,
        TRIM(SPLIT_PART(billing_address, ',', 1)) AS street_address,
        TRIM(SPLIT_PART(billing_address, ',', -4)) AS postal_code,
        TRIM(SPLIT_PART(billing_address, ',', -3)) AS city,
        TRIM(SPLIT_PART(billing_address, ',', -2)) AS state,
        TRIM(SPLIT_PART(billing_address, ',', -1)) AS country
    FROM {{ source('staging', 'stg_orders') }}
    WHERE billing_address IS NOT NULL
),

validated_addresses AS (
    SELECT *
    FROM parsed_addresses
    WHERE city != ''
    AND state != ''
    AND country != ''
    AND city IS NOT NULL
    AND state IS NOT NULL
    AND country IS NOT NULL
)

SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['address']) }} AS address_id,
    street_address,
    postal_code,
    COALESCE(l.location_id, 
        {{ dbt_utils.generate_surrogate_key(['city', 'state', 'country']) }}
    ) AS location_id,
    current_timestamp() AS created_at
FROM validated_addresses
LEFT JOIN {{ ref('locations') }} l
    USING (city, state, country)
WHERE address IS NOT NULL