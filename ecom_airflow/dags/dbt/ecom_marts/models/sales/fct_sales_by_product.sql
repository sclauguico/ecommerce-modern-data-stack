{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

SELECT
    p.product_id,
    p.category_id,
    DATE(o.order_date) as order_date,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.quantity) as units_sold,
    SUM(oi.total_price) as revenue,
    AVG(oi.unit_price) as avg_selling_price,
    CURRENT_TIMESTAMP as created_at
FROM {{ source('ecom_intermediate', 'products_enriched') }} p
JOIN {{ source('ecom_intermediate', 'order_items') }} oi USING (product_id)
JOIN {{ source('ecom_intermediate', 'orders') }} o USING (order_id)
GROUP BY 1, 2, 3