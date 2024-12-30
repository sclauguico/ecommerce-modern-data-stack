{{
   config(
       materialized='table',
       tags=['transaction']
   )
}}

WITH order_items_base AS (
   SELECT * FROM {{ source('staging', 'stg_order_items') }}
   WHERE product_id IS NOT NULL
),

-- Ensure products exist
valid_products AS (
   SELECT DISTINCT product_id 
   FROM {{ ref('products_enriched') }}
),

-- Filter for valid products and deduplicate early
validated_items AS (
   SELECT DISTINCT oi.*
   FROM order_items_base oi
   INNER JOIN valid_products vp
       ON oi.product_id = vp.product_id
),

-- Get the first review score for each order_id/product_id combination
reviews_deduped AS (
    SELECT DISTINCT 
        order_id,
        product_id,
        FIRST_VALUE(review_score) OVER (
            PARTITION BY order_id, product_id 
            ORDER BY loaded_at DESC
        ) as review_score
    FROM {{ source('staging', 'stg_reviews') }}
)

SELECT DISTINCT
  oi.order_item_id,
  oi.order_id, 
  oi.product_id,
  o.customer_id,
  oi.quantity,
  oi.unit_price,
  oi.total_price,
  p.category_id,
  p.subcategory_id,
  p.brand_id,
  r.review_score,
  oi.created_at
FROM validated_items oi
LEFT JOIN {{ source('staging', 'stg_orders') }} o 
  USING (order_id)
LEFT JOIN {{ ref('products_enriched') }} p 
  USING (product_id)
LEFT JOIN reviews_deduped r 
  USING (order_id, product_id)