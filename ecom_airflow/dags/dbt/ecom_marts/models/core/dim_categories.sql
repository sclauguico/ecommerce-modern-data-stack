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
),

product_aggregation AS (
    SELECT
        category_id,
        LISTAGG(DISTINCT product_name, ', ') 
            WITHIN GROUP (ORDER BY product_name) as product_names
    FROM {{ source('ecom_intermediate', 'products_enriched') }}
    GROUP BY category_id
)

SELECT
    CAST(c.category_id AS varchar(100)) as category_id,
    CAST(c.category_name AS varchar(100)) as category_name,
    CAST(ch.subcategories AS varchar(100)) as subcategories,
    CAST(pa.product_names AS varchar(100)) as product_names,
    c.created_at
FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
LEFT JOIN category_hierarchy ch 
    USING (category_id)
LEFT JOIN product_aggregation pa 
    USING (category_id)