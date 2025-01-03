WITH source AS (
    SELECT * FROM {{ source('ecom_raw', 'reviews') }}
),
casted AS (
    SELECT
        CAST(review_id AS VARCHAR) as review_id,
        CAST(product_id AS VARCHAR) as product_id,
        CAST(order_id AS VARCHAR) as order_id,
        CAST(customer_id AS VARCHAR) as customer_id,
        CAST(review_score AS INTEGER) as review_score,
        CAST(review_text AS TEXT) as review_text,
        CAST(data_source AS VARCHAR) as data_source,
        CAST(batch_id AS VARCHAR) as batch_id,
        TRY_CAST(loaded_at AS TIMESTAMP) as loaded_at
    FROM source
)
SELECT * FROM casted