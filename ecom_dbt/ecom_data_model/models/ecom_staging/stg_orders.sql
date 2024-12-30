WITH source AS (
   SELECT * FROM {{ source('raw', 'orders') }}
),
casted AS (
   SELECT
       CAST(order_id AS VARCHAR) as order_id,
       CAST(customer_id AS VARCHAR) as customer_id,
       TRY_CAST(order_date AS TIMESTAMP) as order_date,
       CAST(status AS VARCHAR) as status,
       CAST(total_amount AS DECIMAL(12,2)) as total_amount,
       CAST(shipping_cost AS DECIMAL(12,2)) as shipping_cost,
       CAST(payment_method AS VARCHAR) as payment_method,
       CAST(shipping_address AS VARCHAR) as shipping_address,
       CAST(billing_address AS VARCHAR) as billing_address,
       TRY_CAST(created_at AS TIMESTAMP) as created_at,
       TRY_CAST(updated_at AS TIMESTAMP) as updated_at,
       CAST(data_source AS VARCHAR) as data_source,
       CAST(batch_id AS VARCHAR) as batch_id,
       TRY_CAST(loaded_at AS TIMESTAMP) as loaded_at
   FROM source
)
SELECT * FROM casted