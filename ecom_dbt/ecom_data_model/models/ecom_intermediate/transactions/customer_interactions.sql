SELECT
    i.event_id,
    i.customer_id,
    i.product_id,
    p.category_id,
    p.subcategory_id,
    i.event_type,
    i.event_date,
    i.device_type,
    i.session_id,
    i.created_at
FROM {{ source('ecom_staging', 'stg_interactions') }} i
LEFT JOIN {{ source('ecom_staging', 'stg_products') }} p 
    USING (product_id)