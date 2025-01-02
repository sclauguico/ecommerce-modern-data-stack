{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

SELECT
    DATE_TRUNC('day', TO_DATE(order_date)) AS sale_date,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(total_amount) AS total_revenue,
    SUM(shipping_cost) AS total_shipping,
    SUM(total_amount) - SUM(shipping_cost) AS net_revenue,
    AVG(total_amount) AS avg_order_value,
    SUM(total_amount) / COUNT(DISTINCT customer_id) AS revenue_per_customer,
    CURRENT_TIMESTAMP AS updated_at
FROM {{ source('ecom_intermediate', 'orders') }}
GROUP BY sale_date