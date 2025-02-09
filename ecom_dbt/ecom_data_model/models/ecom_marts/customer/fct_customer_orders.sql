{{ config(
    materialized='table',
    tags=['marts', 'customers']
) }}

SELECT
    o.customer_id,
    o.order_id,
    DATE(o.order_date) as order_date,
    o.total_amount,
    COUNT(DISTINCT oi.product_id) AS unique_products,
    SUM(oi.quantity) AS total_items,
    o.total_amount / NULLIF(COUNT(DISTINCT oi.product_id), 0) AS avg_order_value,
    AVG(r.review_score) AS avg_review_score,
    CURRENT_TIMESTAMP() AS created_at
FROM {{ source('ecom_intermediate', 'orders') }} o
LEFT JOIN {{ source('ecom_intermediate', 'order_items') }} oi 
    ON o.order_id = oi.order_id
LEFT JOIN {{ source('ecom_intermediate', 'reviews_enriched') }} r 
    ON o.order_id = r.order_id
{% if is_incremental() %}
WHERE o.order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
GROUP BY 
    o.customer_id, 
    o.order_id, 
    o.order_date, 
    o.total_amount
