{{ config(
    materialized='view',
    tags=['marts', 'dimensions']
) }}

SELECT DISTINCT
    date_day,
    EXTRACT(YEAR FROM date_day) as year,
    EXTRACT(MONTH FROM date_day) as month,
    EXTRACT(DOW FROM date_day) as day_of_week,
    date_trunc('month', date_day) as first_day_of_month,
    last_day(date_day) as last_day_of_month,
FROM (
    SELECT DISTINCT order_date as date_day 
    FROM {{ source('ecom_intermediate', 'orders') }}
    UNION
    SELECT DISTINCT event_date
    FROM {{ source('ecom_intermediate', 'customer_interactions') }}
)