SELECT
    r.product_id,
    r.order_id,
    r.customer_id,
    r.review_score,
    r.review_text,
    p.product_name,
    c.category_name,
    s.subcategory_name,
    b.brand_name,
    r.loaded_at AS created_at
FROM {{ source('ecom_staging', 'stg_reviews') }} r
LEFT JOIN {{ ref('products_enriched') }} p ON r.product_id = p.product_id 
LEFT JOIN {{ ref('categories_enriched') }} c ON p.category_id = c.category_id
LEFT JOIN {{ ref('subcategories_enriched') }} s ON p.subcategory_id = s.subcategory_id
LEFT JOIN {{ ref('brands') }} b ON p.brand_id = b.brand_id