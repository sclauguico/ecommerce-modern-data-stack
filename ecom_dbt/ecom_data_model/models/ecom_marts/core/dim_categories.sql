{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

WITH category_hierarchy AS (
    SELECT DISTINCT
        c.category_id,
        c.category_name,
        LISTAGG(DISTINCT s.subcategory_name, ', ') 
            WITHIN GROUP (ORDER BY s.subcategory_name) as subcategories
    FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
    LEFT JOIN {{ source('ecom_intermediate', 'subcategories_enriched') }} s 
        USING (category_id)
    GROUP BY 
        c.category_id,
        c.category_name
)

SELECT
    CAST(c.category_id AS varchar(100)) as category_id,
    CAST(c.category_name AS varchar(100)) as category_name,
    CAST(ch.subcategories AS varchar(100)) as subcategories,
    CAST(p.product_name AS varchar(100)) as product_name,
    c.created_at
FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
LEFT JOIN category_hierarchy ch 
    USING (category_id)
LEFT JOIN {{ source('ecom_intermediate', 'products_enriched') }} p 
    USING (category_id)