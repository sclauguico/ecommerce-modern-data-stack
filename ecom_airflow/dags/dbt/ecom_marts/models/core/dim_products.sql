{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

SELECT
    p.product_id,
    p.category_id,
    p.product_name,
    p.description,
    p.base_price,
    p.sale_price,
    p.stock_quantity,
    c.category_name,
    s.subcategory_name,
    b.brand_name,
    p.review_count,
    COALESCE(p.avg_review_score, 0) as avg_review_score,
    CASE
        WHEN p.stock_quantity = 0 THEN 'Out of Stock'
        WHEN p.stock_quantity < 10 THEN 'Low Stock'
        ELSE 'In Stock'
    END as inventory_status,
    p.is_active,
    p.created_at,
    CURRENT_TIMESTAMP() as updated_at
FROM {{ source('ecom_intermediate', 'products_enriched') }} p
LEFT JOIN {{ source('ecom_intermediate', 'categories_enriched') }} c 
    ON p.category_id = c.category_id
LEFT JOIN {{ source('ecom_intermediate', 'subcategories_enriched') }} s 
    ON p.subcategory_id = s.subcategory_id
LEFT JOIN {{ source('ecom_intermediate', 'brands') }} b 
    ON p.brand_id = b.brand_id