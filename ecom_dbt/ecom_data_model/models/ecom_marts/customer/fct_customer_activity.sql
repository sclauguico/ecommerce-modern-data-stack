{{ config(
    materialized='table',
    tags=['marts', 'customer', 'fact']
) }}

SELECT
    c.customer_id,
    c.email,
    DATE(i.event_date) AS event_date, 
    
    -- Page Views
    COUNT(CASE WHEN i.event_type = 'view' THEN 1 END) AS total_views,
    COUNT(DISTINCT CASE WHEN i.event_type = 'view' THEN i.product_id END) AS unique_products_viewed,

    -- Cart Activity 
    COUNT(CASE WHEN i.event_type = 'cart_add' THEN 1 END) AS cart_adds,
    COUNT(DISTINCT CASE WHEN i.event_type = 'cart_add' THEN i.product_id END) AS unique_products_added,

    -- Purchase Activity
    COUNT(CASE WHEN i.event_type = 'purchase' THEN 1 END) AS purchases,
    COUNT(DISTINCT CASE WHEN i.event_type = 'purchase' THEN i.product_id END) AS unique_products_purchased,

    -- Session Info
    COUNT(DISTINCT i.session_id) AS total_sessions,
    COUNT(DISTINCT i.device_type) AS devices_used,

    -- Timestamps
    CURRENT_TIMESTAMP() AS updated_at
FROM {{ source('ecom_intermediate', 'customers_enriched') }} c
LEFT JOIN {{ source('ecom_intermediate', 'customer_interactions') }} i 
    ON c.customer_id = i.customer_id
WHERE DATE(i.event_date) IS NOT NULL
GROUP BY 
    c.customer_id, 
    c.email, 
    DATE(i.event_date)