WITH source AS (
   SELECT * FROM {{ source('ecom_raw', 'products') }}
),
casted AS (
   SELECT
       CAST(product_id AS VARCHAR) as product_id,
       CAST(category_id AS VARCHAR) as category_id,
       CAST(subcategory_id AS VARCHAR) as subcategory_id,
       CAST(product_name AS VARCHAR) as product_name,
       CAST(description AS TEXT) as description,
       CAST(base_price AS DECIMAL(12,2)) as base_price,
       CAST(sale_price AS DECIMAL(12,2)) as sale_price,
       CAST(stock_quantity AS INTEGER) as stock_quantity,
       CAST(weight_kg AS DECIMAL(8,2)) as weight_kg,
       CAST(is_active AS BOOLEAN) as is_active,
       TRY_CAST(created_at AS TIMESTAMP) as created_at,
       CAST(brand AS VARCHAR) as brand,
       CAST(sku AS VARCHAR) as sku,
       CAST(rating AS DECIMAL(3,1)) as rating,
       CAST(review_count AS INTEGER) as review_count,
       CAST(data_source AS VARCHAR) as data_source,
       CAST(batch_id AS VARCHAR) as batch_id,
       TRY_CAST(loaded_at AS TIMESTAMP) as loaded_at
   FROM source
)
SELECT * FROM casted