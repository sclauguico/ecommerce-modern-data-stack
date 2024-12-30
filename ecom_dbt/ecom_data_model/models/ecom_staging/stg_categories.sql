WITH source AS (
    SELECT * FROM {{ source('raw', 'categories') }}
),
casted AS (
    SELECT
        CAST(category_id AS VARCHAR) as category_id,
        CAST(category_name AS VARCHAR) as category_name,
        TRY_CAST(created_at AS TIMESTAMP) as created_at,
        CAST(data_source AS VARCHAR) as data_source,
        CAST(batch_id AS VARCHAR) as batch_id,
        TRY_CAST(loaded_at AS TIMESTAMP) as loaded_at
    FROM source
)
SELECT * FROM casted
