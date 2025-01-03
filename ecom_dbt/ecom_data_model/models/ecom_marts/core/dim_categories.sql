{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

SELECT
    c.category_id,
    c.category_name,
    s.subcategory_name,
    p.product_name
FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
LEFT JOIN {{ source('ecom_intermediate', 'subcategories_enriched') }} s USING (category_id)
LEFT JOIN {{ source('ecom_intermediate', 'products_enriched') }} p USING (category_id)
LEFT JOIN {{ source('ecom_intermediate', 'order_items') }} oi USING (product_id)
GROUP BY 1, 2, 3, 4
