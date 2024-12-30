WITH source AS (
   SELECT * FROM {{ source('raw', 'order_items') }}
),
casted AS (
   SELECT
       CAST(order_item_id AS VARCHAR) as order_item_id,
       CAST(order_id AS VARCHAR) as order_id,
       CAST(product_id AS VARCHAR) as product_id,
       CAST(quantity AS INTEGER) as quantity,
       CAST(unit_price AS DECIMAL(12,2)) as unit_price,
       CAST(total_price AS DECIMAL(12,2)) as total_price,
       TRY_CAST(created_at AS TIMESTAMP) as created_at,
       CAST(data_source AS VARCHAR) as data_source,
       CAST(batch_id AS VARCHAR) as batch_id,
       TRY_CAST(loaded_at AS TIMESTAMP) as loaded_at
   FROM source
)
SELECT * FROM casted