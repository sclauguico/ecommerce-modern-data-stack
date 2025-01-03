{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

SELECT
    product_id,
    date_trunc('day', event_date) as event_day,
    COUNT(DISTINCT CASE WHEN event_type = 'view' THEN event_id END) as views,
    COUNT(DISTINCT CASE WHEN event_type = 'cart_add' THEN event_id END) as cart_adds,
    COUNT(DISTINCT CASE WHEN event_type = 'wishlist_add' THEN event_id END) as wishlist_adds,
    COUNT(DISTINCT customer_id) as unique_customers
FROM {{ source('ecom_intermediate', 'customer_interactions') }}
GROUP BY 1, 2