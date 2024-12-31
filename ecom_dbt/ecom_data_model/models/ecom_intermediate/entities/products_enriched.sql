WITH product_orders AS (
    SELECT
        product_id,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(quantity) AS total_quantity_sold,
        SUM(total_price) AS total_revenue
    FROM {{ source('ecom_staging', 'stg_order_items') }}
    GROUP BY product_id
),

product_reviews AS (
    SELECT
        product_id,
        COUNT(*) AS review_count,
        AVG(review_score) AS avg_review_score
    FROM {{ source('ecom_staging', 'stg_reviews') }}
    GROUP BY product_id
),

all_brands AS (
    SELECT DISTINCT
        TRIM(brand) as brand_name,
        {{ dbt_utils.generate_surrogate_key(['TRIM(brand)']) }} as brand_id
    FROM {{ source('ecom_staging', 'stg_products') }}
    WHERE brand IS NOT NULL
    AND TRIM(brand) != ''
)

SELECT
    p.product_id,
    p.product_name,
    p.description,
    p.base_price,
    p.sale_price,
    p.stock_quantity,
    p.weight_kg,
    p.is_active,
    COALESCE(b.brand_id, ab.brand_id) as brand_id,
    p.category_id,
    p.subcategory_id,
    COALESCE(po.total_orders, 0) AS total_orders,
    COALESCE(po.total_quantity_sold, 0) AS total_quantity_sold,
    COALESCE(po.total_revenue, 0) AS total_revenue,
    COALESCE(pr.review_count, 0) AS review_count,
    pr.avg_review_score,
    p.created_at
FROM {{ source('ecom_staging', 'stg_products') }} p
LEFT JOIN {{ ref('brands') }} b 
    ON TRIM(p.brand) = b.brand_name
LEFT JOIN all_brands ab
    ON TRIM(p.brand) = ab.brand_name
LEFT JOIN {{ source('ecom_staging', 'stg_categories') }} c 
    ON p.category_id = c.category_id
LEFT JOIN {{ source('ecom_staging', 'stg_subcategories') }} s 
    ON p.category_id = s.category_id 
    AND p.subcategory_id = s.subcategory_id
LEFT JOIN product_orders po 
    ON p.product_id = po.product_id
LEFT JOIN product_reviews pr 
    ON p.product_id = pr.product_id
WHERE p.product_id IS NOT NULL