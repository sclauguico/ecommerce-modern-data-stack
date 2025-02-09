WITH parsed_addresses AS (
    SELECT DISTINCT
        shipping_address AS address,
        shipping_address AS street_address,
        c.city,
        c.state,
        c.country
    FROM {{ source('ecom_staging', 'stg_orders') }} o
    JOIN {{ source('ecom_staging', 'stg_customers') }} c 
        ON o.customer_id = c.customer_id
    WHERE shipping_address IS NOT NULL
    
    UNION DISTINCT
    
    SELECT DISTINCT
        billing_address AS address,
        billing_address AS street_address,
        c.city,
        c.state,
        c.country
    FROM {{ source('ecom_staging', 'stg_orders') }} o
    JOIN {{ source('ecom_staging', 'stg_customers') }} c 
        ON o.customer_id = c.customer_id
    WHERE billing_address IS NOT NULL
)

SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['address', 'city', 'state', 'country']) }} AS address_id,
    street_address,
    COALESCE(l.location_id, 
        {{ dbt_utils.generate_surrogate_key(['city', 'state', 'country']) }}
    ) AS location_id,
    current_timestamp() AS created_at
FROM parsed_addresses
LEFT JOIN {{ ref('locations') }} l
    USING (city, state, country)
WHERE address IS NOT NULL