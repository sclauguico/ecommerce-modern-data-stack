{{ config(
    materialized='table',
    tags=['marts', 'customers']
) }}

SELECT
    customer_id,
    order_id,
    order_date,
    total_amount,
    COUNT(DISTINCT oi.product_id) as unique_products,
    SUM(oi.quantity) as total_items,
    AVG(r.review_score) as avg_review_score
FROM {{ source('ecom_intermediate', 'orders') }} o
LEFT JOIN {{ source('ecom_intermediate', 'order_items') }} oi USING (order_id)
LEFT JOIN {{ source('ecom_intermediate', 'reviews_enriched') }} r USING (order_id)
{% if is_incremental() %}
WHERE o.order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
GROUP BY 1, 2, 3, 4
