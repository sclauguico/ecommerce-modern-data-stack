{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

WITH daily_sales AS (
    SELECT
        c.location_id,
        date_trunc('day', o.order_date)::date as date,
        COUNT(DISTINCT o.order_id) as total_orders,
        COUNT(DISTINCT o.customer_id) as customer_count,
        SUM(o.total_amount) as total_revenue,
        SUM(o.total_amount) / NULLIF(COUNT(DISTINCT o.order_id), 0) as avg_order_value
    FROM {{ source('ecom_intermediate', 'orders') }} o
    JOIN {{ source('ecom_intermediate', 'customers_enriched') }} c 
        ON o.customer_id = c.customer_id
    GROUP BY 1, 2
)

SELECT
    location_id,
    date,
    total_orders,
    total_revenue,
    avg_order_value,
    customer_count,
    current_timestamp() as created_at
FROM daily_sales