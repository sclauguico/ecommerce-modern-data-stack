import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import boto3
import psycopg2
from io import StringIO
import pandas as pd
import json
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class RecentEcommerceDataGenerator:
    def __init__(self, id_offset=1000000):  # Add a large offset to some IDs
        # Set end date to today and start date to 30 days ago
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        self.fake = Faker()
        self.id_offset = id_offset  # Store the offset
        np.random.seed(42)
        random.seed(42)
        
        # Keep existing templates and aspects from the original code
        self.review_templates = {
            'positive': [
                "Great {product_type}! {positive_aspect}. Highly recommend!",
                "Really happy with this purchase. {positive_aspect} and {another_positive}.",
                "Excellent quality {product_type}. {positive_aspect}.",
                "Best {product_type} I've bought. {positive_aspect}.",
                "{positive_aspect}. Worth every penny!"
            ],
            'neutral': [
                "Decent {product_type}. {positive_aspect}, but {negative_aspect}.",
                "Average {product_type}. {neutral_comment}.",
                "Good enough for the price. {neutral_comment}.",
                "{positive_aspect}, however {negative_aspect}.",
                "Not bad, but not great. {neutral_comment}."
            ],
            'negative': [
                "Disappointed with this {product_type}. {negative_aspect}.",
                "Not worth the price. {negative_aspect}.",
                "Had issues with {negative_aspect}.",
                "Wouldn't recommend. {negative_aspect}.",
                "Poor quality {product_type}. {negative_aspect}."
            ]
        }
    
        self.positive_aspects = [
            "Excellent build quality",
            "Fast shipping",
            "Great customer service",
            "Perfect fit",
            "Amazing features",
            "Beautiful design",
            "Easy to use",
            "Incredible value",
            "Superior performance",
            "Long battery life"
        ]
        
        self.negative_aspects = [
            "Poor build quality",
            "Slow delivery",
            "Unresponsive customer service",
            "Doesn't fit well",
            "Missing features",
            "Unappealing design",
            "Complicated to use",
            "Overpriced",
            "Underperforms",
            "Short battery life"
        ]
        
        self.neutral_comments = [
            "Meets basic expectations",
            "Standard quality",
            "Average performance",
            "Fair for the price",
            "Could be better"
        ]

    def generate_product_categories(self):
        """Generate product categories with meaningful business metrics and attributes"""
        # Define comprehensive category characteristics
        categories = {
            'Electronics': {
                'subcategories': [
                    {
                        'name': 'Smartphones',
                        'margin_range': (0.15, 0.25),
                        'return_rate': 0.05,
                        'shipping_weight_factor': 0.8
                    },
                    {
                        'name': 'Laptops',
                        'margin_range': (0.12, 0.20),
                        'return_rate': 0.06,
                        'shipping_weight_factor': 2.0
                    },
                    {
                        'name': 'Accessories',
                        'margin_range': (0.40, 0.60),
                        'return_rate': 0.03,
                        'shipping_weight_factor': 0.3
                    },
                    {
                        'name': 'Tablets',
                        'margin_range': (0.20, 0.30),
                        'return_rate': 0.04,
                        'shipping_weight_factor': 1.0
                    },
                    {
                        'name': 'Wearables',
                        'margin_range': (0.30, 0.45),
                        'return_rate': 0.04,
                        'shipping_weight_factor': 0.4
                    }
                ],
                'seasonal_factors': {  # Monthly factors
                    1: 1.2,  # Post-holiday sales
                    2: 0.8,
                    3: 0.9,
                    4: 1.0,
                    5: 1.1,  # Graduation
                    6: 1.2,
                    7: 1.0,
                    8: 1.3,  # Back to school
                    9: 1.1,
                    10: 1.0,
                    11: 1.4,  # Black Friday
                    12: 1.5   # Holiday season
                },
                'growth_rate': 0.15,  # 15% annual growth
                'storage_cost': 2.5    # Storage cost per unit
            },
            'Fashion': {
                'subcategories': [
                    {
                        'name': "Men's Clothing",
                        'margin_range': (0.45, 0.65),
                        'return_rate': 0.12,
                        'shipping_weight_factor': 0.7
                    },
                    {
                        'name': "Women's Clothing",
                        'margin_range': (0.50, 0.70),
                        'return_rate': 0.15,
                        'shipping_weight_factor': 0.7
                    },
                    {
                        'name': "Children's Clothing",
                        'margin_range': (0.40, 0.60),
                        'return_rate': 0.08,
                        'shipping_weight_factor': 0.5
                    },
                    {
                        'name': 'Shoes',
                        'margin_range': (0.45, 0.65),
                        'return_rate': 0.10,
                        'shipping_weight_factor': 1.0
                    },
                    {
                        'name': 'Fashion Accessories',
                        'margin_range': (0.60, 0.80),
                        'return_rate': 0.07,
                        'shipping_weight_factor': 0.3
                    }
                ],
                'seasonal_factors': {
                    1: 0.8,   # Post-holiday lull
                    2: 0.9,
                    3: 1.2,   # Spring fashion
                    4: 1.1,
                    5: 1.2,   # Summer prep
                    6: 1.1,
                    7: 0.9,
                    8: 1.2,   # Fall fashion
                    9: 1.1,
                    10: 1.0,
                    11: 1.3,  # Black Friday
                    12: 1.4   # Holiday season
                },
                'growth_rate': 0.12,
                'storage_cost': 1.5
            },
            'Home & Living': {
                'subcategories': [
                    {
                        'name': 'Furniture',
                        'margin_range': (0.35, 0.55),
                        'return_rate': 0.06,
                        'shipping_weight_factor': 3.0
                    },
                    {
                        'name': 'Kitchen',
                        'margin_range': (0.40, 0.60),
                        'return_rate': 0.05,
                        'shipping_weight_factor': 1.5
                    },
                    {
                        'name': 'Decor',
                        'margin_range': (0.50, 0.70),
                        'return_rate': 0.08,
                        'shipping_weight_factor': 1.0
                    },
                    {
                        'name': 'Bedding',
                        'margin_range': (0.45, 0.65),
                        'return_rate': 0.04,
                        'shipping_weight_factor': 1.2
                    },
                    {
                        'name': 'Storage',
                        'margin_range': (0.40, 0.60),
                        'return_rate': 0.03,
                        'shipping_weight_factor': 2.0
                    }
                ],
                'seasonal_factors': {
                    1: 1.1,   # New Year organization
                    2: 0.9,
                    3: 1.2,   # Spring cleaning
                    4: 1.1,
                    5: 1.0,
                    6: 1.1,   # Wedding season
                    7: 1.0,
                    8: 1.2,   # Back to school
                    9: 0.9,
                    10: 1.0,
                    11: 1.2,  # Black Friday
                    12: 1.3   # Holiday season
                },
                'growth_rate': 0.10,
                'storage_cost': 3.0
            }
        }
        
        category_data = []
        subcategory_data = []
        
        for cat_id, (category_name, category_props) in enumerate(categories.items(), 1):
            # Calculate category metrics
            current_month = self.end_date.month
            seasonal_factor = category_props['seasonal_factors'][current_month]
            
            category_data.append({
                'category_id': cat_id,
                'category_name': category_name,
                'created_at': self.start_date,
                'seasonal_factor': seasonal_factor,
                'growth_rate': category_props['growth_rate'],
                'storage_cost': category_props['storage_cost'],
                'description': f"Category for all {category_name.lower()} products",
                'display_order': cat_id,
                'is_active': True,
                'target_margin': round(
                    sum(sub['margin_range'][0] for sub in category_props['subcategories']) / 
                    len(category_props['subcategories']), 
                    2
                ),
                'avg_return_rate': round(
                    sum(sub['return_rate'] for sub in category_props['subcategories']) / 
                    len(category_props['subcategories']), 
                    3
                )
            })
            
            for sub_id, subcategory in enumerate(category_props['subcategories'], 1):
                # Calculate subcategory-specific metrics
                margin_low, margin_high = subcategory['margin_range']
                target_margin = round(random.uniform(margin_low, margin_high), 2)
                
                subcategory_data.append({
                    'subcategory_id': (cat_id * 100) + sub_id,
                    'category_id': cat_id,
                    'subcategory_name': subcategory['name'],
                    'created_at': self.start_date,
                    'target_margin': target_margin,
                    'return_rate': subcategory['return_rate'],
                    'shipping_weight_factor': subcategory['shipping_weight_factor'],
                    'display_order': sub_id,
                    'is_active': True,
                    'handling_fee': round(5 * subcategory['shipping_weight_factor'], 2),
                    'min_order_quantity': 1,
                    'description': f"{subcategory['name']} within {category_name} category"
                })
        
        return pd.DataFrame(category_data), pd.DataFrame(subcategory_data)

    def generate_random_date(self):
        """Generate a random date within the last 30 days"""
        days_offset = random.randint(0, 30)
        return self.end_date - timedelta(days=days_offset)

    def generate_products(self, n_products=1000):
        """Generate product catalog with realistic patterns and pricing"""
        categories_df, subcategories_df = self.generate_product_categories()
        
        # Define category-specific price ranges and characteristics
        category_props = {
            1: {  # Electronics
                'price_range': (200, 2000),
                'margin_range': (0.15, 0.35),
                'seasonal_factor': 1.2,  # Higher demand in holiday season
                'stock_range': (50, 500),
                'weight_range': (0.3, 10.0)
            },
            2: {  # Fashion
                'price_range': (20, 300),
                'margin_range': (0.40, 0.70),
                'seasonal_factor': 1.1,
                'stock_range': (100, 1000),
                'weight_range': (0.1, 2.0)
            },
            3: {  # Home & Living
                'price_range': (50, 1500),
                'margin_range': (0.30, 0.60),
                'seasonal_factor': 1.0,
                'stock_range': (30, 300),
                'weight_range': (0.5, 25.0)
            },
            4: {  # Beauty
                'price_range': (10, 200),
                'margin_range': (0.45, 0.75),
                'seasonal_factor': 1.1,
                'stock_range': (200, 800),
                'weight_range': (0.1, 1.0)
            },
            5: {  # Sports
                'price_range': (30, 500),
                'margin_range': (0.25, 0.50),
                'seasonal_factor': 1.15,
                'stock_range': (50, 400),
                'weight_range': (0.2, 15.0)
            }
        }

        # Brand tiers and their characteristics
        brand_tiers = {
            'Premium': {
                'price_multiplier': (1.3, 1.8),
                'rating_boost': 0.5,
                'stock_multiplier': 0.7,  # Limited stock for premium items
                'weight': 0.2  # 20% of brands are premium
            },
            'Standard': {
                'price_multiplier': (0.9, 1.2),
                'rating_boost': 0.2,
                'stock_multiplier': 1.0,
                'weight': 0.5  # 50% of brands are standard
            },
            'Budget': {
                'price_multiplier': (0.6, 0.8),
                'rating_boost': -0.1,
                'stock_multiplier': 1.3,  # Higher stock for budget items
                'weight': 0.3  # 30% of brands are budget
            }
        }

        products = []
        for product_id in range(1, n_products + 1):
            category_id = random.randint(1, len(categories_df))
            cat_props = category_props[category_id]
            
            # Select brand tier
            brand_tier = random.choices(
                list(brand_tiers.keys()),
                weights=[t['weight'] for t in brand_tiers.values()]
            )[0]
            tier_props = brand_tiers[brand_tier]
            
            # Generate brand with consistent properties
            brand = f"{self.fake.company()} {brand_tier}"
            
            # Base price calculation
            base_price = random.uniform(*cat_props['price_range'])
            price_multiplier = random.uniform(*tier_props['price_multiplier'])
            final_base_price = base_price * price_multiplier
            
            # Calculate sale price with seasonal adjustment
            discount_factor = random.uniform(0.8, 0.95)
            sale_price = final_base_price * discount_factor * cat_props['seasonal_factor']
            
            # Stock quantity based on tier and category
            base_stock = random.randint(*cat_props['stock_range'])
            stock_quantity = int(base_stock * tier_props['stock_multiplier'])
            
            # Rating calculation
            base_rating = random.uniform(3.5, 4.5)
            adjusted_rating = min(5.0, base_rating + tier_props['rating_boost'])
            
            valid_subcats = subcategories_df[subcategories_df['category_id'] == category_id]
            subcategory_id = random.choice(valid_subcats['subcategory_id'].values)
            
            # Generate created_at date with more recent dates more likely
            days_ago = int(np.random.exponential(7))  # Most products created in last week
            days_ago = min(days_ago, 30)  # Cap at 30 days
            created_at = self.end_date - timedelta(days=days_ago)
            
            products.append({
                'product_id': product_id,
                'category_id': category_id,
                'subcategory_id': subcategory_id,
                'product_name': f"{brand} {self.fake.word().title()}",
                'description': self.generate_product_description(category_id, brand_tier),
                'base_price': round(final_base_price, 2),
                'sale_price': round(sale_price, 2),
                'stock_quantity': stock_quantity,
                'weight_kg': round(random.uniform(*cat_props['weight_range']), 2),
                'is_active': True if stock_quantity > 0 else False,
                'created_at': created_at,
                'brand': brand,
                'brand_tier': brand_tier,
                'sku': f"SKU-{category_id}{subcategory_id}-{random.randint(10000, 99999)}",
                'rating': round(adjusted_rating, 1),
                'review_count': random.randint(0, 100),
                'margin': round(random.uniform(*cat_props['margin_range']), 2)
            })
        
        return pd.DataFrame(products), categories_df, subcategories_df

    def generate_customers(self, n_customers=1000, historic_customers_file='de-ecommerce/data/customers.csv'):
        """Generate customer data with realistic segments and behaviors"""
        customers = []
        
        # Define customer segments with characteristics
        segments = {
            'High Value': {
                'income_range': (80000, 150000),
                'age_range': (35, 60),
                'purchase_frequency': (3, 8),
                'avg_order_value': (150, 500),
                'channel_weights': {'Web': 0.4, 'Mobile App': 0.4, 'Email': 0.2},
                'education_weights': {
                    'Bachelor': 0.4, 'Master': 0.3, 'PhD': 0.2, 
                    'Some College': 0.1
                },
                'weight': 0.2  # 20% of customers
            },
            'Mid Value': {
                'income_range': (50000, 79999),
                'age_range': (25, 45),
                'purchase_frequency': (2, 5),
                'avg_order_value': (50, 150),
                'channel_weights': {'Web': 0.3, 'Mobile App': 0.5, 'Email': 0.2},
                'education_weights': {
                    'High School': 0.2, 'Some College': 0.3, 'Bachelor': 0.4, 
                    'Master': 0.1
                },
                'weight': 0.5  # 50% of customers
            },
            'Low Value': {
                'income_range': (20000, 49999),
                'age_range': (18, 35),
                'purchase_frequency': (1, 3),
                'avg_order_value': (20, 50),
                'channel_weights': {'Web': 0.2, 'Mobile App': 0.6, 'Email': 0.2},
                'education_weights': {
                    'High School': 0.4, 'Some College': 0.4, 'Bachelor': 0.2
                },
                'weight': 0.3  # 30% of customers
            }
        }
        
        # Location probabilities (urban areas have more customers)
        location_weights = {
            'Urban': 0.6,
            'Suburban': 0.3,
            'Rural': 0.1
        }
        
        # Try to load historic customers
        try:
            historic_df = pd.read_csv(historic_customers_file)
            print(f"Loaded {len(historic_df)} historic customers")
            n_returning = min(int(len(historic_df) * 0.3), n_customers)
            n_new = n_customers - n_returning
            returning_customers = historic_df.sample(n=n_returning).copy()
            
            # Update returning customers
            for idx in returning_customers.index:
                last_login = self.generate_random_date()
                returning_customers.loc[idx, 'last_login'] = last_login
                returning_customers.loc[idx, 'is_active'] = True
                
            print(f"Processing {n_returning} returning customers")
        except FileNotFoundError:
            print("No historic customers found. Generating all new customers.")
            n_returning = 0
            n_new = n_customers
            returning_customers = pd.DataFrame()
        
        # Generate new customers
        print(f"Generating {n_new} new customers")
        start_id = self.id_offset if not len(returning_customers) else max(returning_customers['customer_id']) + 1
        
        for customer_id in range(start_id, start_id + n_new):
            # Select customer segment
            segment = random.choices(
                list(segments.keys()),
                weights=[s['weight'] for s in segments.values()]
            )[0]
            segment_props = segments[segment]
            
            # Generate signup date with exponential distribution (more recent dates more likely)
            days_ago = int(np.random.exponential(7))  # Most signups in last week
            days_ago = min(days_ago, 30)  # Cap at 30 days
            signup_date = self.end_date - timedelta(days=days_ago)
            
            # Generate last login with higher frequency for more recent signups
            days_since_signup = (self.end_date - signup_date).days
            login_gap = int(np.random.exponential(days_since_signup * 0.3))
            last_login = min(signup_date + timedelta(days=login_gap), self.end_date)
            
            # Generate age within segment range
            age = random.randint(*segment_props['age_range'])
            
            # Select preferred channel based on segment weights
            preferred_channel = random.choices(
                list(segment_props['channel_weights'].keys()),
                weights=list(segment_props['channel_weights'].values())
            )[0]
            
            # Select education based on segment weights
            education = random.choices(
                list(segment_props['education_weights'].keys()),
                weights=list(segment_props['education_weights'].values())
            )[0]
            
            # Select location type based on weights
            location_type = random.choices(
                list(location_weights.keys()),
                weights=list(location_weights.values())
            )[0]
            
            customers.append({
                'customer_id': customer_id,
                'email': self.fake.email(),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'age': age,
                'gender': random.choice(['M', 'F', 'Other']),
                'annual_income': random.randint(*segment_props['income_range']),
                'customer_segment': segment,
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
                'education': education,
                'location_type': location_type,
                'city': self.fake.city(),
                'state': self.fake.state(),
                'country': 'USA',
                'signup_date': signup_date,
                'last_login': last_login,
                'preferred_channel': preferred_channel,
                'is_active': True,
                'expected_purchase_frequency': random.randint(*segment_props['purchase_frequency']),
                'expected_order_value': round(random.uniform(*segment_props['avg_order_value']), 2)
            })
        
        # Combine new and returning customers
        new_customers_df = pd.DataFrame(customers)
        final_customers_df = pd.concat([returning_customers, new_customers_df], ignore_index=True)
        
        print(f"Generated total of {len(final_customers_df)} customers")
        print(f"- Returning customers: {len(returning_customers)}")
        print(f"- New customers: {len(new_customers_df)}")
        
        return final_customers_df

    def generate_orders(self, customers_df, products_df):
        """Generate order data for the last 30 days with realistic patterns"""
        orders = []
        order_items = []
        order_id = 1 + self.id_offset
        
        # Add daily pattern factors
        hourly_patterns = {
            0: 0.2,  # 12 AM
            1: 0.1,  # 1 AM
            2: 0.1,
            3: 0.1,
            4: 0.1,
            5: 0.2,
            6: 0.3,
            7: 0.5,
            8: 0.7,  # Morning rush
            9: 1.0,
            10: 1.2,
            11: 1.3,
            12: 1.4,  # Lunch peak
            13: 1.3,
            14: 1.2,
            15: 1.1,
            16: 1.2,
            17: 1.3,  # Evening rush
            18: 1.4,
            19: 1.5,  # Prime time
            20: 1.4,
            21: 1.2,
            22: 0.8,
            23: 0.4   # Late night
        }
        
        # Day of week factors
        daily_factors = {
            0: 0.7,  # Monday
            1: 0.8,
            2: 0.9,
            3: 1.0,
            4: 1.3,  # Friday boost
            5: 1.5,  # Saturday peak
            6: 1.1   # Sunday
        }
        
        # Special events/promotions (simulated for the last 30 days)
        special_events = {
            5: 1.3,   # Flash sale
            12: 1.4,  # Mid-month promotion
            15: 1.5,  # Payday effect
            25: 1.3,  # End-of-month sale
        }
        
        for _, customer in customers_df.iterrows():
            # Customer segment based frequency
            if 'customer_segment' in customer:
                base_frequency = {
                    'High Value': 3,
                    'Mid Value': 2,
                    'Low Value': 1
                }.get(customer['customer_segment'], 2)
            else:
                base_frequency = 2
                
            # Adjust frequency based on customer history
            if customer['signup_date'] < self.start_date:
                base_frequency *= 1.2  # Returning customers more likely to order
                
            num_orders = np.random.poisson(base_frequency)
            
            for _ in range(num_orders):
                # Generate order date with realistic patterns
                order_date = self.generate_random_date()
                hour = random.choices(
                    range(24),
                    weights=[hourly_patterns[h] for h in range(24)]
                )[0]
                
                order_date = order_date.replace(hour=hour)
                
                # Apply daily and special event factors
                day_factor = daily_factors[order_date.weekday()]
                special_factor = special_events.get(order_date.day, 1.0)
                
                # Status based on order date
                days_since_order = (self.end_date - order_date).days
                if days_since_order < 1:
                    status = 'Pending'
                elif days_since_order < 2:
                    status = 'Processing'
                elif days_since_order < 4:
                    status = 'Shipped'
                else:
                    status = 'Delivered'
                
                # Generate order items with realistic patterns
                base_items = np.random.poisson(1.5) + 1
                num_items = max(1, int(base_items * day_factor * special_factor))
                
                # Product selection logic
                if random.random() < 0.7:  # 70% chance of category affinity
                    preferred_category = random.randint(1, 5)
                    matching_products = products_df[
                        products_df['category_id'] == preferred_category
                    ]
                    if len(matching_products) > 0:
                        order_products = matching_products.sample(
                            n=min(num_items, len(matching_products)),
                            replace=True
                        )
                    else:
                        order_products = products_df.sample(
                            n=min(num_items, len(products_df)),
                            replace=True
                        )
                else:
                    order_products = products_df.sample(
                        n=min(num_items, len(products_df)),
                        replace=True
                    )
                
                shipping_cost = round(random.uniform(5, 20), 2)
                total_amount = shipping_cost
                
                for _, product in order_products.iterrows():
                    quantity = random.randint(1, 3)
                    
                    # Apply time-based pricing
                    base_price = product['sale_price']
                    time_factor = hourly_patterns[hour] * day_factor * special_factor
                    adjusted_price = base_price * (0.9 + (time_factor * 0.2))
                    
                    item_total = quantity * adjusted_price
                    total_amount += item_total
                    
                    order_items.append({
                        'order_item_id': len(order_items) + 1 + self.id_offset,
                        'order_id': order_id,
                        'product_id': product['product_id'],
                        'quantity': quantity,
                        'unit_price': round(adjusted_price, 2),
                        'total_price': round(item_total, 2),
                        'created_at': order_date
                    })
                
                # Payment method distribution
                payment_methods = {
                    'Credit Card': 0.6,
                    'PayPal': 0.3,
                    'Debit Card': 0.1
                }
                payment_method = random.choices(
                    list(payment_methods.keys()),
                    weights=list(payment_methods.values())
                )[0]
                
                orders.append({
                    'order_id': order_id,
                    'customer_id': customer['customer_id'],
                    'order_date': order_date,
                    'status': status,
                    'total_amount': round(total_amount, 2),
                    'shipping_cost': shipping_cost,
                    'payment_method': payment_method,
                    'shipping_address': self.fake.street_address(),
                    'billing_address': self.fake.street_address(),
                    'created_at': order_date,
                    'updated_at': order_date + timedelta(minutes=random.randint(5, 60))
                })
                
                order_id += 1
        
        return pd.DataFrame(orders), pd.DataFrame(order_items)

    def generate_customer_interactions(self, customers_df, products_df):
        """Generate customer interactions for the last 30 days with realistic patterns"""
        events = []
        
        # Define time-of-day patterns
        hourly_patterns = {
            'desktop': {  # Peak during work hours
                'distribution': [0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.11, 
                            0.10, 0.09, 0.08, 0.09, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 
                            0.02, 0.01, 0.01, 0.01]
            },
            'mobile': {   # More evening activity
                'distribution': [0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08,
                            0.07, 0.08, 0.09, 0.08, 0.07, 0.08, 0.09, 0.10, 0.12, 0.11,
                            0.09, 0.08, 0.05, 0.03]
            },
            'tablet': {   # Evening peak
                'distribution': [0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                            0.07, 0.08, 0.07, 0.08, 0.07, 0.08, 0.09, 0.11, 0.13, 0.12,
                            0.10, 0.08, 0.06, 0.03]
            }
        }

        # Event type transition probabilities
        event_flow = {
            'view': {'cart_add': 0.3, 'view': 0.6, 'search': 0.1},
            'cart_add': {'purchase': 0.4, 'cart_remove': 0.2, 'view': 0.4},
            'cart_remove': {'view': 0.7, 'search': 0.3},
            'search': {'view': 0.8, 'cart_add': 0.2},
            'purchase': {'view': 0.7, 'search': 0.3},
            'start': {'view': 0.7, 'search': 0.3}
        }

        for _, customer in customers_df.iterrows():
            # Determine base interaction frequency based on customer segment
            if 'customer_segment' in customer:
                base_frequency = {
                    'High Value': 15,
                    'Mid Value': 10,
                    'Low Value': 5
                }.get(customer['customer_segment'], 10)
            else:
                base_frequency = 10

            # Higher frequency for returning customers
            if customer['signup_date'] < self.start_date:
                base_frequency *= 1.2

            num_sessions = np.random.poisson(base_frequency)
            
            for _ in range(num_sessions):
                session_id = f"session_{random.randint(10000, 99999)}"
                
                # Device selection based on time and customer preferences
                device_weights = {'desktop': 0.4, 'mobile': 0.45, 'tablet': 0.15}
                if customer['preferred_channel'] == 'Mobile App':
                    device_weights = {'desktop': 0.2, 'mobile': 0.7, 'tablet': 0.1}
                
                device_type = random.choices(
                    list(device_weights.keys()),
                    weights=list(device_weights.values())
                )[0]
                
                # Generate session start time
                base_date = self.generate_random_date()
                hour_weights = hourly_patterns[device_type]['distribution']
                hour = random.choices(range(24), weights=hour_weights)[0]
                minute = random.randint(0, 59)
                
                session_start = base_date.replace(hour=hour, minute=minute)
                
                # Generate sequence of events
                current_event = 'start'
                event_time = session_start
                num_events = np.random.poisson(4) + 1  # At least one event per session
                
                for _ in range(num_events):
                    # Choose next event based on transition probabilities
                    next_event = random.choices(
                        list(event_flow[current_event].keys()),
                        weights=list(event_flow[current_event].values())
                    )[0]
                    
                    if next_event != 'start':
                        # Product selection logic
                        if random.random() < 0.7:  # 70% chance to view related products
                            recent_views = [e['product_id'] for e in events[-3:] 
                                        if e['event_type'] == 'view']
                            if recent_views:
                                related_category = products_df[
                                    products_df['product_id'].isin(recent_views)
                                ]['category_id'].iloc[0]
                                matching_products = products_df[
                                    products_df['category_id'] == related_category
                                ]
                            else:
                                matching_products = products_df
                        else:
                            matching_products = products_df
                        
                        product = matching_products.sample(n=1).iloc[0]
                        
                        events.append({
                            'event_id': len(events) + 1 + self.id_offset,
                            'customer_id': customer['customer_id'],
                            'product_id': product['product_id'],
                            'event_type': next_event,
                            'event_date': event_time,
                            'device_type': device_type,
                            'session_id': session_id,
                            'created_at': event_time
                        })
                    
                    # Increment time realistically
                    event_time += timedelta(
                        seconds=random.randint(10, 300)  # 10 seconds to 5 minutes
                    )
                    current_event = next_event
        
        return pd.DataFrame(events)
    
    def generate_review_text(self, rating, product_type):
        """Generate realistic review text based on rating"""
        if rating >= 4:
            template = random.choice(self.review_templates['positive'])
            aspect = random.choice(self.positive_aspects)
            another = random.choice(self.positive_aspects)
            return template.format(
                product_type=product_type,
                positive_aspect=aspect,
                another_positive=another
            )
        elif rating >= 3:
            template = random.choice(self.review_templates['neutral'])
            return template.format(
                product_type=product_type,
                positive_aspect=random.choice(self.positive_aspects),
                negative_aspect=random.choice(self.negative_aspects),
                neutral_comment=random.choice(self.neutral_comments)
            )
        else:
            template = random.choice(self.review_templates['negative'])
            return template.format(
                product_type=product_type,
                negative_aspect=random.choice(self.negative_aspects)
            )

    def generate_reviews(self, orders_df, order_items_df, products_df, customers_df):
        """Generate reviews with realistic patterns and timing"""
        reviews_data = []
        review_id = 1 + self.id_offset
        
        # Define review probability by customer segment
        review_prob = {
            'High Value': 0.8,
            'Mid Value': 0.6,
            'Low Value': 0.4
        }
        
        # Define rating distributions by price range
        def get_rating_distribution(price):
            if price > 500:  # High-end products
                return [0.02, 0.03, 0.10, 0.35, 0.50]  # Higher expectations
            elif price > 100:  # Mid-range products
                return [0.05, 0.10, 0.15, 0.40, 0.30]
            else:  # Budget products
                return [0.10, 0.15, 0.25, 0.30, 0.20]  # More varied ratings
        
        # Define review delay patterns
        def get_review_delay():
            delays = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
            weights = [0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]
            return random.choices(delays, weights=weights)[0]
        
        for _, order in orders_df.iterrows():
            if order['status'] != 'Delivered':
                continue
                
            customer = customers_df[
                customers_df['customer_id'] == order['customer_id']
            ].iloc[0]
            
            # Determine review probability
            base_prob = review_prob.get(
                customer.get('customer_segment', 'Mid Value'),
                0.5
            )
            
            # Adjust probability based on customer history
            if customer['signup_date'] < self.start_date:
                base_prob *= 1.2  # More likely to review if returning customer
            
            order_items = order_items_df[order_items_df['order_id'] == order['order_id']]
            
            for _, item in order_items.iterrows():
                if random.random() < base_prob:
                    product = products_df[
                        products_df['product_id'] == item['product_id']
                    ].iloc[0]
                    
                    # Get rating distribution based on product price
                    rating_dist = get_rating_distribution(product['base_price'])
                    review_score = random.choices(range(1, 6), weights=rating_dist)[0]
                    
                    # Calculate review date
                    review_delay = get_review_delay()
                    review_date = order['order_date'] + timedelta(days=review_delay)
                    
                    # Only include if review date is within our 30-day window
                    if review_date <= self.end_date:
                        review_text = self.generate_review_text(
                            review_score,
                            product['product_name']
                        )
                        
                        reviews_data.append({
                            'review_id': review_id,
                            'product_id': product['product_id'],
                            'order_id': order['order_id'],
                            'customer_id': customer['customer_id'],
                            'review_score': review_score,
                            'review_text': review_text,
                            'review_date': review_date,
                            'helpful_votes': np.random.poisson(2) if review_score != 3 else np.random.poisson(1),
                            'verified_purchase': True,
                            'created_at': review_date
                        })
                        
                        review_id += 1
                        
                        # Update product rating and review count
                        mask = products_df['product_id'] == product['product_id']
                        products_df.loc[mask, 'review_count'] += 1
                        current_rating = products_df.loc[mask, 'rating'].iloc[0]
                        current_count = products_df.loc[mask, 'review_count'].iloc[0]
                        new_rating = ((current_rating * (current_count - 1)) + review_score) / current_count
                        products_df.loc[mask, 'rating'] = round(new_rating, 1)
        
        return pd.DataFrame(reviews_data), products_df

    def generate_all_data(self, n_customers=1000, n_products=1000):
        """Generate all e-commerce data for the last 30 days"""
        output_formats = {
            'customers': 'json',
            'products': 'json',
            'orders': 'json',
            'reviews': 'json',
            'categories': 'csv',
            'subcategories': 'csv',
            'order_items': 'csv',
            'interactions': 'csv'
        }
        
        print(f"Generating data for period: {self.start_date.date()} to {self.end_date.date()}")
        
        print("Generating products...")
        products_df, categories_df, subcategories_df = self.generate_products(n_products)
        
        print("Generating customers...")
        customers_df = self.generate_customers(n_customers)
        
        print("Generating orders...")
        orders_df, order_items_df = self.generate_orders(customers_df, products_df)
        
        print("Generating reviews...")
        reviews_df, updated_products_df = self.generate_reviews(orders_df, order_items_df, products_df, customers_df)
        
        print("Generating customer interactions...")
        interactions_df = self.generate_customer_interactions(customers_df, updated_products_df)
        
        data_dict = {
            'customers': customers_df,
            'products': updated_products_df,
            'categories': categories_df,
            'subcategories': subcategories_df,
            'orders': orders_df,
            'order_items': order_items_df,
            'reviews': reviews_df,
            'interactions': interactions_df
        }
        
        # self.save_data(data_dict, output_formats)
        
        return data_dict
    
class EcommerceDataLoader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, 
                 pg_host, pg_port, pg_user, 
                 pg_password, pg_database):
        """Initialize connections to S3 and PostgreSQL"""
        # S3 client setup
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        self.bucket_name = bucket_name

        # PostgreSQL connection setup
        self.pg_conn_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        self.engine = create_engine(self.pg_conn_string)

    def upload_json_to_s3(self, data_dict, table_name):
        """Upload JSON data directly to S3"""
        try:
            # Convert DataFrame to JSON format with metadata
            json_data = {
                "metadata": {
                    "table": table_name,
                    "recordCount": len(data_dict),
                    "generatedAt": pd.Timestamp.now().isoformat(),
                    "version": "1.0"
                },
                "data": json.loads(data_dict.to_json(orient='records', date_format='iso'))
            }
            
            # Convert to JSON string
            json_str = json.dumps(json_data)
            
            # Upload to S3
            s3_key = f'json/{table_name}.json'
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str,
                ContentType='application/json'
            )
            print(f"Uploaded {table_name} to S3: s3://{self.bucket_name}/{s3_key}")
            
        except Exception as e:
            print(f"Error uploading {table_name} to S3: {str(e)}")
            raise

    def load_csv_to_postgres(self, df, table_name):
        """Load DataFrame directly to PostgreSQL"""
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        try:
            # Create copy buffer
            output = StringIO()
            df.to_csv(output, index=False, header=True)
            output.seek(0)
            
            # Create database connection
            with self.engine.connect() as connection:
                # Create table if it doesn't exist
                df.head(0).to_sql(table_name, connection, if_exists='replace', index=False)
                
                # Copy data
                raw_conn = connection.connection
                with raw_conn.cursor() as cursor:
                    cursor.copy_expert(
                        f"COPY {table_name} FROM STDIN WITH CSV HEADER",
                        output
                    )
                raw_conn.commit()
                
            print(f"Loaded {len(df)} rows to PostgreSQL table: {table_name}")
            
        except Exception as e:
            print(f"Error loading {table_name} to PostgreSQL: {str(e)}")
            raise

def save_data(data_dict):
    """Save data to S3 and PostgreSQL"""
    # AWS credentials (replace with your own)
    aws_access_key_id = os.environ.get("AWS_S3_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_S3_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("AWS_S3_LATEST_SYNTH")

    # Validate AWS credentials
    if not all([aws_access_key_id, aws_secret_access_key, bucket_name]):
        raise ValueError("Required AWS environment variables are not set")

    # PostgreSQL connection details
    pg_config = {
        'pg_host': os.environ.get("POSTGRES_HOST"),
        'pg_port': int(os.environ.get("POSTGRES_PORT")),
        'pg_user': os.environ.get("POSTGRES_USER"),
        'pg_password': os.environ.get("POSTGRES_PASSWORD"),
        'pg_database': os.environ.get("POSTGRES_DB")
    }

    # Initialize loader
    loader = EcommerceDataLoader(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=bucket_name,
        **pg_config
    )

    # Define which tables go to which destination
    json_tables = ['customers', 'products', 'orders', 'reviews']
    csv_tables_mapping = {
        'categories': 'latest_categories',
        'subcategories': 'latest_subcategories',
        'order_items': 'latest_order_items',
        'interactions': 'latest_interactions'
    }
    
    # Upload JSON files to S3
    for table_name in json_tables:
        if table_name in data_dict:
            loader.upload_json_to_s3(data_dict[table_name], table_name)

    # Load CSV files to PostgreSQL with the 'latest_' prefix
    for original_name, latest_name in csv_tables_mapping.items():
        if original_name in data_dict:
            print(f"Loading {original_name} to PostgreSQL as {latest_name}")
            loader.load_csv_to_postgres(data_dict[original_name], latest_name)
            
    output_dir = Path('generated_latest_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for table_name, df in data_dict.items():
        csv_path = output_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {table_name} to {csv_path}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, df in data_dict.items():
        csv_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {table_name} to {csv_path}")

if __name__ == "__main__":
    # Generate data
    generator = RecentEcommerceDataGenerator()
    data = generator.generate_all_data()
    
    # Save data directly to S3 and PostgreSQL
    save_data(data)
    
    # Print summary
    print("\nData Generation and Loading Summary:")
    for table_name, df in data.items():
        print(f"\n{table_name.upper()} Table:")
        print(f"Total records: {len(df)}")
        
        # Different tables have different date columns
        date_columns = {
            'customers': 'signup_date',
            'products': 'created_at',
            'orders': 'order_date',
            'latest_order_items': 'created_at',
            'reviews': 'created_at',
            'latest_interactions': 'event_date',
            'latest_categories': 'created_at',
            'latest_subcategories': 'created_at'
        }
        
        # Get the appropriate date column for this table
        date_col = date_columns.get(table_name)
        if date_col and date_col in df.columns:
            print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        
        destination = 'S3' if table_name in ['customers', 'products', 'orders', 'reviews'] else 'PostgreSQL'
        print(f"Loaded to: {destination}")
