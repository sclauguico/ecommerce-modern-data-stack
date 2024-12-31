WITH source AS (
   SELECT * FROM {{ source('ecom_raw', 'interactions') }}
),
casted AS (
   SELECT
       CAST(event_id AS VARCHAR) as event_id,
       CAST(customer_id AS VARCHAR) as customer_id,
       CAST(product_id AS VARCHAR) as product_id,
       CAST(event_type AS VARCHAR) as event_type,
       TRY_CAST(event_date AS TIMESTAMP) as event_date,
       CAST(device_type AS VARCHAR) as device_type,
       CAST(session_id AS VARCHAR) as session_id,
       TRY_CAST(created_at AS TIMESTAMP) as created_at,
       CAST(data_source AS VARCHAR) as data_source,
       CAST(batch_id AS VARCHAR) as batch_id,
       TRY_CAST(loaded_at AS TIMESTAMP) as loaded_at
   FROM source
)
SELECT * FROM casted