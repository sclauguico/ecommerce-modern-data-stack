{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

WITH daily_sales AS (
    SELECT
        p.product_id,
        o.order_date::date as date,
        oi.quantity,
        oi.total_price,
        p.base_price,
        p.stock_quantity
    FROM {{ source('ecom_intermediate', 'products_enriched') }} p
    LEFT JOIN {{ source('ecom_intermediate', 'order_items') }} oi 
        USING (product_id)
    LEFT JOIN {{ source('ecom_intermediate', 'orders') }} o 
        USING (order_id)
    WHERE o.order_date IS NOT NULL
),

daily_metrics AS (
    SELECT
        product_id,
        date,
        COUNT(DISTINCT quantity) as total_orders,
        SUM(quantity) as units_sold,
        SUM(total_price) as revenue,
        -- Calculate profit margin
        CASE 
            WHEN SUM(total_price) > 0 
            THEN ((SUM(total_price) - (SUM(quantity) * MIN(base_price))) / SUM(total_price) * 100)
            ELSE 0 
        END as profit_margin,
        -- Get stock level
        MIN(stock_quantity) as stock_level
    FROM daily_sales
    GROUP BY 1, 2
),

-- Calculate reorder flags in separate CTE
sales_velocity AS (
    SELECT 
        product_id,
        date,
        stock_level,
        AVG(units_sold) OVER (
            PARTITION BY product_id 
            ORDER BY date 
            ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
        ) as avg_daily_sales
    FROM daily_metrics
)

SELECT
    m.product_id,
    m.date,
    m.total_orders,
    m.units_sold,
    m.revenue,
    ROUND(m.profit_margin, 2) as profit_margin,
    m.stock_level,
    -- Set reorder flag based on stock level and sales velocity
    CASE 
        WHEN m.stock_level <= GREATEST(v.avg_daily_sales * 7, 10)
        THEN TRUE 
        ELSE FALSE 
    END as reorder_flag,
    current_timestamp() as created_at
FROM daily_metrics m
LEFT JOIN sales_velocity v 
    USING (product_id, date)