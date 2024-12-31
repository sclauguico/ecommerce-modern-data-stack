SELECT
    s.subcategory_id,
    s.subcategory_name,
    s.category_id,
    COUNT(DISTINCT p.product_id) AS product_count,
    s.created_at
FROM {{ source('ecom_staging', 'stg_subcategories') }} s
LEFT JOIN {{ source('ecom_staging', 'stg_products') }} p USING (subcategory_id)
GROUP BY 1, 2, 3, 5