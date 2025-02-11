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

class EcommerceDataGenerator:
    def __init__(self, start_date=datetime(2023, 1, 1), end_date=datetime(2025, 1, 31)):
        self.start_date = start_date
        self.end_date = end_date
        self.fake = Faker()
        np.random.seed(42)
        random.seed(42)
        
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
        """Generate product categories with meaningful attributes"""
        categories = {
            'Electronics': {
                'subcategories': ['Smartphones', 'Laptops', 'Accessories', 'Tablets', 'Wearables'],
                'margin_range': (0.15, 0.35),
                'return_rate': 0.08,
                'shipping_weight_factor': 1.2
            },
            'Fashion': {
                'subcategories': ['Men\'s Clothing', 'Women\'s Clothing', 'Children\'s Clothing', 'Shoes', 'Accessories'],
                'margin_range': (0.40, 0.70),
                'return_rate': 0.12,
                'shipping_weight_factor': 0.7
            },
            'Home & Living': {
                'subcategories': ['Furniture', 'Kitchen', 'Decor', 'Bedding', 'Storage'],
                'margin_range': (0.30, 0.60),
                'return_rate': 0.05,
                'shipping_weight_factor': 2.0
            },
            'Beauty': {
                'subcategories': ['Skincare', 'Makeup', 'Haircare', 'Fragrances', 'Tools'],
                'margin_range': (0.45, 0.75),
                'return_rate': 0.03,
                'shipping_weight_factor': 0.5
            },
            'Sports': {
                'subcategories': ['Exercise Equipment', 'Sportswear', 'Outdoor Gear', 'Accessories', 'Footwear'],
                'margin_range': (0.25, 0.50),
                'return_rate': 0.06,
                'shipping_weight_factor': 1.5
            }
        }
        
        category_data = []
        subcategory_data = []
        
        for cat_id, (category, props) in enumerate(categories.items(), 1):
            margin_range = props['margin_range']
            category_data.append({
                'category_id': cat_id,
                'category_name': category,
                'created_at': self.start_date,
                'target_margin': round(random.uniform(*margin_range), 2),
                'return_rate': props['return_rate'],
                'shipping_weight_factor': props['shipping_weight_factor'],
                'description': f"Category for all {category.lower()} products",
                'display_order': cat_id,
                'is_active': True
            })
            
            for sub_id, subcategory in enumerate(props['subcategories'], 1):
                # Vary margin slightly from parent category
                sub_margin = random.uniform(
                    margin_range[0] * 0.9,
                    margin_range[1] * 1.1
                )
                
                subcategory_data.append({
                    'subcategory_id': (cat_id * 100) + sub_id,
                    'category_id': cat_id,
                    'subcategory_name': subcategory,
                    'created_at': self.start_date,
                    'target_margin': round(sub_margin, 2),
                    'return_rate': props['return_rate'] * random.uniform(0.8, 1.2),
                    'display_order': sub_id,
                    'is_active': True
                })
        
        return pd.DataFrame(category_data), pd.DataFrame(subcategory_data)

    def generate_products(self, n_products=1000):
        """Generate product catalog with realistic pricing"""
        categories_df, subcategories_df = self.generate_product_categories()
        
        # Define category-specific price ranges
        category_prices = {
            1: {'range': (500, 2000), 'name': 'Electronics'},  # Electronics
            2: {'range': (20, 200), 'name': 'Fashion'},   # Fashion
            3: {'range': (50, 1000), 'name': 'Home & Living'},  # Home & Living
            4: {'range': (10, 100), 'name': 'Beauty'},    # Beauty
            5: {'range': (30, 300), 'name': 'Sports'}     # Sports
        }
        
        products = []
        for product_id in range(1, n_products + 1):
            category_id = random.randint(1, len(categories_df))
            valid_subcats = subcategories_df[subcategories_df['category_id'] == category_id]
            subcategory_id = random.choice(valid_subcats['subcategory_id'].values)
            
            # Get category-specific price range
            price_range = category_prices[category_id]['range']
            base_price = random.uniform(*price_range)
            
            # Add price variations based on subcategory
            subcategory_factor = 0.8 + (0.4 * random.random())  # 0.8 to 1.2
            base_price *= subcategory_factor
            
            # Calculate sale price with seasonal factors
            sale_price = base_price * random.uniform(0.8, 1.0)
            
            products.append({
                'product_id': product_id,
                'category_id': category_id,
                'subcategory_id': subcategory_id,
                'product_name': f"{self.fake.company()} {self.fake.word().title()}",
                'description': self.fake.text(max_nb_chars=200),
                'base_price': round(base_price, 2),
                'sale_price': round(sale_price, 2),
                'stock_quantity': random.randint(0, 1000),
                'weight_kg': round(random.uniform(0.1, 20.0), 2),
                'is_active': random.random() > 0.1,
                'created_at': self.start_date + timedelta(days=random.randint(0, 30)),
                'brand': self.fake.company(),
                'sku': f"SKU-{random.randint(10000, 99999)}",
                'rating': round(random.uniform(3.0, 5.0), 1),
                'review_count': random.randint(0, 1000)
            })
        
        return pd.DataFrame(products), categories_df, subcategories_df

    def generate_customers(self, n_customers=1000):
        """Generate customer data with segments"""
        customers = []
        
        # Define customer segments
        segments = {
            'High Value': {'income_range': (80000, 150000), 'age_range': (35, 60), 'weight': 0.2},
            'Mid Value': {'income_range': (50000, 79999), 'age_range': (25, 45), 'weight': 0.5},
            'Low Value': {'income_range': (20000, 49999), 'age_range': (18, 35), 'weight': 0.3}
        }
        
        for customer_id in range(1, n_customers + 1):
            # Select segment
            segment = random.choices(list(segments.keys()), 
                                weights=[s['weight'] for s in segments.values()])[0]
            segment_props = segments[segment]
            
            # Generate segment-appropriate values
            age = random.randint(*segment_props['age_range'])
            income = random.randint(*segment_props['income_range'])
            
            signup_date = self.start_date + timedelta(days=random.randint(0, (self.end_date - self.start_date).days))
            
            customers.append({
                'customer_id': customer_id,
                'email': self.fake.email(),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'age': age,
                'gender': random.choice(['M', 'F', 'Other']),
                'annual_income': income,
                'customer_segment': segment,
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
                'education': random.choice(['High School', 'Some College', 'Bachelor', 'Master', 'PhD']),
                'location_type': random.choice(['Urban', 'Suburban', 'Rural']),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'country': 'USA',
                'signup_date': signup_date,
                'last_login': signup_date + timedelta(days=random.randint(0, 30)),
                'preferred_channel': random.choice(['Web', 'Mobile App', 'Email']),
                'is_active': random.random() > 0.1
            })
        
        return pd.DataFrame(customers)

    def generate_orders(self, customers_df, products_df):
        """Generate order data with realistic patterns"""
        orders = []
        order_items = []
        order_id = 1
        
        # Add seasonality factors (monthly)
        seasonality = {
            1: 1.2,  # January (post-holiday sales)
            2: 0.8,  # February (slow month)
            3: 0.9,  # March
            4: 1.0,  # April (spring shopping)
            5: 1.1,  # May
            6: 1.15, # June (summer start)
            7: 1.05, # July
            8: 1.1,  # August (back to school)
            9: 1.0,  # September
            10: 1.1, # October
            11: 1.3, # November (Black Friday)
            12: 1.5  # December (holiday season)
        }
        
        # Add day-of-week factors
        daily_factors = {
            0: 0.7,  # Monday
            1: 0.8,  # Tuesday
            2: 0.9,  # Wednesday
            3: 1.0,  # Thursday
            4: 1.3,  # Friday
            5: 1.5,  # Saturday
            6: 1.1   # Sunday
        }
        
        # Add yearly growth trend
        base_growth_rate = 1.15  # 15% annual growth
        
        for _, customer in customers_df.iterrows():
            # Number of orders influenced by customer demographics
            income_factor = min(2.0, max(0.5, customer['annual_income'] / 65000))
            customer_frequency = np.random.poisson(5 * income_factor)
            
            for _ in range(customer_frequency):
                max_days = (self.end_date - customer['signup_date']).days
                if max_days <= 0:
                    order_date = customer['signup_date']
                else:
                    order_date = customer['signup_date'] + timedelta(days=random.randint(0, max_days))
                
                # Apply seasonality and trends
                month_factor = seasonality[order_date.month]
                day_factor = daily_factors[order_date.weekday()]
                days_since_start = (order_date - self.start_date).days
                years_since_start = days_since_start / 365.0
                growth_factor = base_growth_rate ** years_since_start
                
                # Combined factor for order value
                total_factor = month_factor * day_factor * growth_factor
                
                # Order status based on recency
                if order_date + timedelta(days=7) > self.end_date:
                    status = random.choice(['Pending', 'Processing', 'Shipped'])
                else:
                    status = 'Delivered'
                
                shipping_cost = round(random.uniform(5, 20) * growth_factor, 2)
                
                # Generate order items with product affinity
                num_items = np.random.poisson(2) + 1
                category_preference = random.randint(1, 5)  # Simulate customer category preference
                preferred_products = products_df[products_df['category_id'] == category_preference]
                
                if len(preferred_products) > 0 and random.random() < 0.7:  # 70% chance to buy from preferred category
                    order_products = preferred_products.sample(n=min(num_items, len(preferred_products)), replace=True)
                else:
                    order_products = products_df.sample(n=min(num_items, len(products_df)), replace=True)
                
                total_amount = shipping_cost
                for _, product in order_products.iterrows():
                    quantity = np.random.poisson(1.5) + 1  # More realistic quantity distribution
                    price = product['sale_price'] * (1 + random.uniform(-0.1, 0.1))  # Price variation
                    item_total = quantity * price * total_factor
                    total_amount += item_total
                    
                    order_items.append({
                        'order_item_id': len(order_items) + 1,
                        'order_id': order_id,
                        'product_id': product['product_id'],
                        'quantity': quantity,
                        'unit_price': round(price, 2),
                        'total_price': round(item_total, 2),
                        'created_at': order_date
                    })
                
                orders.append({
                    'order_id': order_id,
                    'customer_id': customer['customer_id'],
                    'order_date': order_date,
                    'status': status,
                    'total_amount': round(total_amount, 2),
                    'shipping_cost': shipping_cost,
                    'payment_method': random.choice(['Credit Card', 'PayPal', 'Debit Card']),
                    'shipping_address': self.fake.street_address(),
                    'billing_address': self.fake.street_address(),
                    'created_at': order_date,
                    'updated_at': order_date + timedelta(days=random.randint(0, 5))
                })
                
                order_id += 1
        
        return pd.DataFrame(orders), pd.DataFrame(order_items)

    def generate_customer_interactions(self, customers_df, products_df):
        """Generate customer interaction events with realistic patterns"""
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
            # Number of sessions varies by customer segment
            if 'customer_segment' in customer:
                if customer['customer_segment'] == 'High Value':
                    num_sessions = np.random.poisson(15)
                elif customer['customer_segment'] == 'Mid Value':
                    num_sessions = np.random.poisson(10)
                else:
                    num_sessions = np.random.poisson(5)
            else:
                num_sessions = np.random.poisson(10)
            
            for _ in range(num_sessions):
                session_id = f"session_{random.randint(10000, 99999)}"
                device_type = random.choices(
                    ['desktop', 'mobile', 'tablet'], 
                    weights=[0.4, 0.45, 0.15]
                )[0]
                
                # Generate session start time
                base_date = customer['signup_date'] + timedelta(
                    days=random.randint(0, (self.end_date - customer['signup_date']).days)
                )
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
                        # Select product based on customer preferences
                        if random.random() < 0.7 and 'customer_segment' in customer:
                            # 70% chance to view products matching customer segment
                            if customer['customer_segment'] == 'High Value':
                                matching_products = products_df[products_df['base_price'] > 100]
                            elif customer['customer_segment'] == 'Mid Value':
                                matching_products = products_df[
                                    (products_df['base_price'] >= 50) & 
                                    (products_df['base_price'] <= 100)
                                ]
                            else:
                                matching_products = products_df[products_df['base_price'] < 50]
                            
                            if len(matching_products) > 0:
                                product = matching_products.sample(n=1).iloc[0]
                            else:
                                product = products_df.sample(n=1).iloc[0]
                        else:
                            product = products_df.sample(n=1).iloc[0]
                        
                        events.append({
                            'event_id': len(events) + 1,
                            'customer_id': customer['customer_id'],
                            'product_id': product['product_id'],
                            'event_type': next_event,
                            'event_date': event_time,
                            'device_type': device_type,
                            'session_id': session_id,
                            'created_at': event_time
                        })
                    
                    # Increment time for next event (1-5 minutes between events)
                    event_time += timedelta(minutes=random.randint(1, 5))
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
        """Generate reviews with realistic patterns and sentiment"""
        reviews_data = []
        
        # Define sentiment patterns by product price range
        def get_rating_distribution(price):
            if price > 500:  # Expensive products
                return [0.02, 0.03, 0.10, 0.35, 0.50]  # Higher expectations
            elif price > 100:  # Mid-range products
                return [0.05, 0.10, 0.15, 0.40, 0.30]
            else:  # Budget products
                return [0.10, 0.15, 0.25, 0.30, 0.20]  # More varied ratings
        
        # Define review probability by customer segment
        review_prob = {
            'High Value': 0.8,
            'Mid Value': 0.6,
            'Low Value': 0.4
        }
        
        # Define time patterns for review submission
        def get_review_delay():
            # Most reviews come within first week
            delays = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
            weights = [0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]
            return random.choices(delays, weights=weights)[0]
        
        for _, order in orders_df.iterrows():
            customer = customers_df[customers_df['customer_id'] == order['customer_id']].iloc[0]
            order_items = order_items_df[order_items_df['order_id'] == order['order_id']]
            
            # Determine if customer will leave reviews
            review_probability = review_prob.get(
                customer.get('customer_segment', 'Mid Value'), 
                0.5
            )
            
            if random.random() < review_probability:
                for _, item in order_items.iterrows():
                    product = products_df[products_df['product_id'] == item['product_id']].iloc[0]
                    
                    # Get rating distribution based on product price
                    rating_dist = get_rating_distribution(product['base_price'])
                    review_score = random.choices(range(1, 6), weights=rating_dist)[0]
                    
                    # Generate review text
                    review_text = self.generate_review_text(
                        review_score, 
                        product['category_id']
                    )
                    
                    # Calculate review date
                    review_date = order['order_date'] + timedelta(
                        days=get_review_delay()
                    )
                    
                    reviews_data.append({
                        'product_id': product['product_id'],
                        'order_id': order['order_id'],
                        'customer_id': customer['customer_id'],
                        'review_score': review_score,
                        'review_text': review_text,
                        'review_date': review_date,
                        'helpful_votes': np.random.poisson(2) if review_score != 3 else np.random.poisson(1),
                        'verified_purchase': True
                    })
        
        reviews_df = pd.DataFrame(reviews_data)
        
        # Update product ratings
        for pid in products_df['product_id'].unique():
            product_reviews = reviews_df[reviews_df['product_id'] == pid]
            if len(product_reviews) > 0:
                products_df.loc[products_df['product_id'] == pid, 'rating'] = \
                    round(product_reviews['review_score'].mean(), 1)
                products_df.loc[products_df['product_id'] == pid, 'review_count'] = \
                    len(product_reviews)
        
        return reviews_df, products_df

    def format_datetime(self, dt):
        """Format datetime objects for JSON serialization"""
        return dt.isoformat() if isinstance(dt, datetime) else dt

    def df_to_json_records(self, df):
        """Convert DataFrame to JSON-serializable records with datetime handling"""
        return json.loads(df.to_json(orient='records', date_format='iso'))

    def generate_all_data(self, n_customers=1000, n_products=1000):
        """Generate all e-commerce data with specified output formats"""
        # Define output format for each table
        output_formats = {
            'customers': 'json',  # Complex nested data, good for JSON
            'products': 'json',   # Product details often needed in API format
            'orders': 'json',     # Order details benefit from JSON structure
            'reviews': 'json',    # Review text better handled in JSON
            'categories': 'csv',  # Simple structural data good for CSV
            'subcategories': 'csv',
            'order_items': 'csv', # Transactional data good for CSV
            'interactions': 'csv' # High-volume data better in CSV
        }
        
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
        
        # Prepare data dictionary
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
        
        # # Save data in appropriate formats
        # print("Saving data in specified formats...")
        # self.save_data(data_dict, output_formats)
        
        return data_dict
    
class EcommerceDataLoader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, 
                 pg_host, pg_port, pg_user, 
                 pg_password, pg_database):
        """Initialize connections to S3 and PostgreSQL"""
        # Validate required parameters
        if not all([aws_access_key_id, aws_secret_access_key, bucket_name]):
            raise ValueError("AWS credentials and bucket name are required")

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

    def upload_json_to_s3(self, data_df, table_name):
        """Upload JSON data directly to S3"""
        if not isinstance(data_df, pd.DataFrame):
            raise ValueError("data_df must be a pandas DataFrame")

        try:
            # Convert DataFrame to JSON format with metadata
            json_data = {
                "metadata": {
                    "table": table_name,
                    "recordCount": len(data_df),
                    "generatedAt": pd.Timestamp.now().isoformat(),
                    "version": "1.0"
                },
                "data": json.loads(data_df.to_json(orient='records', date_format='iso'))
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
            print(f"Successfully uploaded {table_name} to S3: s3://{self.bucket_name}/{s3_key}")
            
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
                
            print(f"Successfully loaded {len(df)} rows to PostgreSQL table: {table_name}")
            
        except Exception as e:
            print(f"Error loading {table_name} to PostgreSQL: {str(e)}")
            raise

def save_data(data_dict):
    """Save data to S3 and PostgreSQL"""
    # Get AWS credentials from environment variables
    aws_access_key_id = os.environ.get("AWS_S3_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_S3_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("AWS_S3_HISTORIC_SYNTH")

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
    csv_tables = ['categories', 'subcategories', 'order_items', 'interactions']

    # Validate data_dict
    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dictionary containing DataFrames")

    # Upload JSON files to S3
    for table_name in json_tables:
        if table_name in data_dict:
            loader.upload_json_to_s3(data_dict[table_name], table_name)

    # Load CSV files to PostgreSQL
    for table_name in csv_tables:
        if table_name in data_dict:
            loader.load_csv_to_postgres(data_dict[table_name], table_name)

    output_dir = Path('generated_historic_data')
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
    try:
        generator = EcommerceDataGenerator()
        data = generator.generate_all_data()
        
        # Save data directly to S3 and PostgreSQL
        save_data(data)        
        # Print sample data and format information
        print("\nData Generation Summary:")
        for table_name, df in data.items():
            print(f"\n{table_name.upper()} Table:")
            print(f"Total records: {len(df)}")
            print(f"Output format: {'JSON' if table_name in ['customers', 'products', 'orders', 'reviews'] else 'CSV'}")
            print("\nSample data:")
            print(df.head())
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise
