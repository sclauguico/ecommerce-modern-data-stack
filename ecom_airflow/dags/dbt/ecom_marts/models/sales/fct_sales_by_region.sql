{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

SELECT
    l.state,
    l.country,
    date_trunc('month', o.order_date) as sales_month,
    COUNT(DISTINCT o.order_id) as total_orders,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    SUM(o.total_amount) as total_revenue,
    AVG(o.total_amount) as avg_order_value
FROM {{ source('ecom_intermediate', 'orders') }} o
JOIN {{ source('ecom_intermediate', 'customers_enriched') }} c USING (customer_id)
JOIN {{ source('ecom_intermediate', 'locations') }} l ON c.location_id = l.location_id
GROUP BY 1, 2, 3