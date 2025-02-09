{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

SELECT
    DATE_TRUNC('DAY', order_date)::DATE as sale_date,
    COUNT(DISTINCT order_id) as total_orders,
    COUNT(DISTINCT customer_id) as unique_customers,
    SUM(total_amount) as total_revenue,
    SUM(shipping_cost) as total_shipping,
    SUM(total_amount - shipping_cost) as net_revenue,
    AVG(total_amount) as avg_order_value,
    SUM(total_amount) / NULLIF(COUNT(DISTINCT customer_id), 0) as revenue_per_customer,
    CURRENT_TIMESTAMP() as updated_at
FROM {{ source('ecom_intermediate', 'orders') }}
GROUP BY 1