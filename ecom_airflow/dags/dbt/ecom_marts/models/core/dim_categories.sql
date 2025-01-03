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
    category_id,
    category_name,
    subcategories,
    created_at
FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
LEFT JOIN category_hierarchy ch 
    USING (category_id)