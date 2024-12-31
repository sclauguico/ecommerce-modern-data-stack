{{ config(
    materialized='table',
    tags=['marts', 'sales']
) }}

SELECT
    -- Order Info
    o.order_id,
    o.order_date,
    -- Customer Info
    c.customer_id,
    c.email,
    l_customer.city AS customer_city,
    l_customer.state AS customer_state,
    -- Product Info
    p.product_id,
    p.product_name,
    ca.category_name,
    b.brand_name,
    -- Order Status
    os.status_name,
    -- Payment Info
    pm.method_name AS payment_method,
    -- Address Info
    sa.street_address AS shipping_address,
    l_shipping.city AS shipping_city,
    l_shipping.state AS shipping_state,
    -- Transaction Info
    oi.quantity,
    oi.unit_price,
    oi.total_price AS item_total,
    o.shipping_cost,
    o.total_amount AS order_total,
    -- Review Info
    r.review_score,
    -- Timestamps
    o.created_at,
    CURRENT_TIMESTAMP() AS updated_at
FROM {{ source('ecom_intermediate', 'orders') }} o
JOIN {{ source('ecom_intermediate', 'order_items') }} oi 
    ON o.order_id = oi.order_id
JOIN {{ source('ecom_intermediate', 'customers_enriched') }} c 
    ON o.customer_id = c.customer_id
JOIN {{ source('ecom_intermediate', 'locations') }} l_customer
    ON c.location_id = l_customer.location_id
JOIN {{ source('ecom_intermediate', 'products_enriched') }} p 
    ON oi.product_id = p.product_id
JOIN {{ source('ecom_intermediate', 'categories_enriched') }} ca
    ON p.category_id = ca.category_id
JOIN {{ source('ecom_intermediate', 'brands') }} b
    ON p.brand_id = b.brand_id
LEFT JOIN {{ source('ecom_intermediate', 'order_statuses') }} os 
    ON o.status_id = os.status_id
LEFT JOIN {{ source('ecom_intermediate', 'payment_methods') }} pm 
    ON o.payment_method_id = pm.payment_method_id
LEFT JOIN {{ source('ecom_intermediate', 'addresses') }} sa 
    ON o.shipping_address_id = sa.address_id
LEFT JOIN {{ source('ecom_intermediate', 'locations') }} l_shipping
    ON sa.location_id = l_shipping.location_id
LEFT JOIN {{ source('ecom_intermediate', 'reviews_enriched') }} r 
    ON o.order_id = r.order_id 
    AND oi.product_id = r.product_id