{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

WITH date_spine AS (
    SELECT DISTINCT DATE_TRUNC('DAY', date)::DATE as date
    FROM (
        SELECT DISTINCT order_date as date FROM {{ source('ecom_intermediate', 'orders') }}
        UNION
        SELECT DISTINCT created_at as date FROM {{ source('ecom_intermediate', 'orders') }}
        UNION
        SELECT DISTINCT event_date as date FROM {{ source('ecom_intermediate', 'customer_interactions') }}
    )
)
SELECT
    date,
    EXTRACT(YEAR FROM date) as year,
    EXTRACT(MONTH FROM date) as month,
    EXTRACT(DAY FROM date) as day,
    EXTRACT(DOW FROM date) as day_of_week,
    DATE_TRUNC('MONTH', date)::DATE as first_day_of_month,
    LAST_DAY(date) as last_day_of_month,
    DATE_TRUNC('YEAR', date)::DATE as first_day_of_year,
    CASE 
        WHEN date = CURRENT_DATE() THEN 'Current'
        WHEN date > CURRENT_DATE() THEN 'Future'
        ELSE 'Past'
    END as date_status,
    CURRENT_TIMESTAMP() as created_at
FROM date_spine