{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

SELECT
    c.category_id,
    c.category_name,
    COUNT(DISTINCT s.subcategory_id) as subcategories,
    COUNT(DISTINCT p.product_id) as total_products,
    SUM(oi.total_price) as category_revenue
FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
LEFT JOIN {{ source('ecom_intermediate', 'subcategories_enriched') }} s USING (category_id)
LEFT JOIN {{ source('ecom_intermediate', 'products_enriched') }} p USING (category_id)
LEFT JOIN {{ source('ecom_intermediate', 'order_items') }} oi USING (product_id)
GROUP BY 1, 2
