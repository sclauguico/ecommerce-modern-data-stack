{{ config(
    materialized='table',
    tags=['marts', 'customers']
) }}

SELECT
    customer_id,
    product_id,
    order_id,
    review_score,
    review_text,
    p.category_id,
    p.subcategory_id
FROM {{ source('ecom_intermediate', 'reviews_enriched') }} r
JOIN {{ source('ecom_intermediate', 'products_enriched') }} p USING (product_id)
{% if is_incremental() %}
WHERE r.created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
