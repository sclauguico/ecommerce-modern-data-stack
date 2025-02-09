{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

WITH category_hierarchy AS (
    SELECT
        subcategories_dedup.category_id, 
        subcategories_dedup.category_name,
        LEFT(LISTAGG(subcategories_dedup.subcategory_name, ', ') 
WITHIN GROUP (ORDER BY subcategories_dedup.subcategory_name), 1000) AS subcategories

    FROM (
        SELECT DISTINCT 
            c.category_id, 
            c.category_name, 
            s.subcategory_name
        FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
        LEFT JOIN {{ source('ecom_intermediate', 'subcategories_enriched') }} s 
            USING (category_id)
    ) subcategories_dedup
    GROUP BY subcategories_dedup.category_id, subcategories_dedup.category_name
),

product_aggregation AS (
    SELECT
        products_dedup.category_id,
        LEFT(LISTAGG(products_dedup.product_name, ', ') 
        WITHIN GROUP (ORDER BY products_dedup.product_name), 1000) AS product_names
    FROM (
        SELECT DISTINCT 
            category_id, 
            product_name
        FROM {{ source('ecom_intermediate', 'products_enriched') }}
    ) products_dedup
    GROUP BY products_dedup.category_id
)

SELECT
    CAST(c.category_id AS varchar(100)) AS category_id,
    CAST(c.category_name AS varchar(100)) AS category_name,
    CAST(ch.subcategories AS varchar(1000)) AS subcategories,
    CAST(pa.product_names AS varchar(1000)) AS product_names,
    c.created_at
FROM {{ source('ecom_intermediate', 'categories_enriched') }} c
LEFT JOIN category_hierarchy ch 
    USING (category_id)
LEFT JOIN product_aggregation pa 
    USING (category_id)
