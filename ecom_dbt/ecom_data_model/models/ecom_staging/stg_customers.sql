WITH source AS (
    SELECT * FROM {{ source('raw', 'customers') }}
),
casted AS (
    SELECT
        CAST(customer_id AS VARCHAR) as customer_id,
        CAST(email AS VARCHAR) as email,
        CAST(first_name AS VARCHAR) as first_name,
        CAST(last_name AS VARCHAR) as last_name,
        CAST(age AS INTEGER) as age,
        CAST(gender AS VARCHAR) as gender,
        CAST(annual_income AS DECIMAL(12,2)) as annual_income,
        CAST(marital_status AS VARCHAR) as marital_status,
        CAST(education AS VARCHAR) as education,
        CAST(location_type AS VARCHAR) as location_type,
        CAST(city AS VARCHAR) as city,
        CAST(state AS VARCHAR) as state,
        CAST(country AS VARCHAR) as country,
        TRY_CAST(signup_date AS TIMESTAMP) as signup_date,
        TRY_CAST(last_login AS TIMESTAMP) as last_login,
        CAST(preferred_channel AS VARCHAR) as preferred_channel,
        CAST(is_active AS BOOLEAN) as is_active,
        CAST(data_source AS VARCHAR) as data_source,
        CAST(batch_id AS VARCHAR) as batch_id,
        TRY_CAST(loaded_at AS TIMESTAMP) as loaded_at
    FROM source
)
SELECT * FROM casted