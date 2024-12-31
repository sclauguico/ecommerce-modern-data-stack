{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}


SELECT 
    -- Product Base Info
    p.product_id,
    p.product_name,
    p.description,
    p.base_price,
    p.sale_price,
    p.stock_quantity,
    -- Related Info Denormalized
    c.category_name,
    s.subcategory_name, 
    b.brand_name,
    -- Metrics
    COALESCE(po.total_orders, 0) AS total_orders,
    COALESCE(po.total_quantity_sold, 0) AS total_quantity_sold,
    COALESCE(po.total_revenue, 0) AS total_revenue,
    COALESCE(po.total_revenue / NULLIF(po.total_quantity_sold, 0), 0) AS avg_selling_price,
    COALESCE(p.review_count, 0) AS review_count,
    p.avg_review_score,
    p.is_active,
    p.created_at,
    CURRENT_TIMESTAMP() AS updated_at
FROM {{ source('ecom_intermediate', 'products_enriched') }} p
LEFT JOIN {{ source('ecom_intermediate', 'categories_enriched') }} c
    ON p.category_id = c.category_id
LEFT JOIN {{ source('ecom_intermediate', 'subcategories_enriched') }} s 
    ON p.subcategory_id = s.subcategory_id
LEFT JOIN {{ source('ecom_intermediate', 'brands') }} b
    ON p.brand_id = b.brand_id
LEFT JOIN (
    SELECT
        product_id,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(quantity) AS total_quantity_sold,
        SUM(total_price) AS total_revenue  
    FROM {{ source('ecom_intermediate', 'order_items') }}
    GROUP BY 1
) po
    ON p.product_id = po.product_id
