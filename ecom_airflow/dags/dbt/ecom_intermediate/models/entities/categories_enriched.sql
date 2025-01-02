SELECT
    c.category_id,
    c.category_name,
    COUNT(DISTINCT s.subcategory_id) AS subcategory_count,
    COUNT(DISTINCT p.product_id) AS product_count,
    c.created_at
FROM {{ source('ecom_staging', 'stg_categories') }} c
LEFT JOIN {{ source('ecom_staging', 'stg_subcategories') }} s 
    USING (category_id)
LEFT JOIN {{ source('ecom_staging', 'stg_products') }} p 
    USING (category_id)
GROUP BY 1, 2, 5