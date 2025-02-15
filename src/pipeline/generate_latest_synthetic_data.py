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
            """Generate product categories with realistic distributions"""
            categories = {
                'Electronics': ['Smartphones', 'Laptops', 'Accessories', 'Tablets', 'Wearables'],
                'Fashion': ['Men\'s Clothing', 'Women\'s Clothing', 'Children\'s Clothing', 'Shoes', 'Accessories'],
                'Home & Living': ['Furniture', 'Kitchen', 'Decor', 'Bedding', 'Storage'],
                'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrances', 'Tools'],
                'Sports': ['Exercise Equipment', 'Sportswear', 'Outdoor Gear', 'Accessories', 'Footwear']
            }
            
            category_data = []
            subcategory_data = []
            
            for cat_id, (category, subcategories) in enumerate(categories.items(), 1):
                category_data.append({
                    'category_id': cat_id,
                    'category_name': category,
                    'created_at': self.start_date
                })
                
                for sub_id, subcategory in enumerate(subcategories, 1):
                    subcategory_data.append({
                        'subcategory_id': (cat_id * 100) + sub_id,
                        'category_id': cat_id,
                        'subcategory_name': subcategory,
                        'created_at': self.start_date
                    })
            
            return pd.DataFrame(category_data), pd.DataFrame(subcategory_data)

    def generate_random_date(self):
        """Generate a random date within the last 30 days"""
        days_offset = random.randint(0, 30)
        return self.end_date - timedelta(days=days_offset)

    def generate_products(self, n_products=1000):
        """Generate product catalog with realistic patterns"""
        categories_df, subcategories_df = self.generate_product_categories()
        
        # Price tier probabilities and characteristics
        price_tiers = {
            'luxury': {'multiplier': (2.0, 3.0), 'stock': (10, 50), 'weight': 0.1},
            'premium': {'multiplier': (1.5, 2.0), 'stock': (20, 100), 'weight': 0.2},
            'standard': {'multiplier': (1.0, 1.5), 'stock': (50, 200), 'weight': 0.4},
            'budget': {'multiplier': (0.6, 1.0), 'stock': (100, 500), 'weight': 0.3}
        }
        
        # Brand quality tiers
        brand_tiers = {
            'luxury': {'rating_base': (4.3, 5.0), 'weight': 0.1},
            'premium': {'rating_base': (4.0, 4.7), 'weight': 0.2},
            'standard': {'rating_base': (3.5, 4.3), 'weight': 0.4},
            'budget': {'rating_base': (3.0, 4.0), 'weight': 0.3}
        }
        
        products = []
        for product_id in range(1, n_products + 1):
            category_id = random.randint(1, len(categories_df))
            valid_subcats = subcategories_df[subcategories_df['category_id'] == category_id]
            subcategory_id = random.choice(valid_subcats['subcategory_id'].values)
            
            # Select price and brand tiers
            price_tier = random.choices(
                list(price_tiers.keys()),
                weights=[t['weight'] for t in price_tiers.values()]
            )[0]
            brand_tier = random.choices(
                list(brand_tiers.keys()),
                weights=[t['weight'] for t in brand_tiers.values()]
            )[0]
            
            # Generate base price using lognormal distribution
            base_price = np.random.lognormal(
                mean=np.log(100),
                sigma=0.7
            )
            
            # Apply tier multipliers
            tier_multiplier = random.uniform(
                *price_tiers[price_tier]['multiplier']
            )
            final_base_price = base_price * tier_multiplier
            
            # Generate sale price with seasonal factors
            month_factor = 1.0
            if self.end_date.month in [11, 12]:  # Holiday season
                month_factor = random.uniform(0.7, 0.9)  # Bigger discounts
            elif self.end_date.month in [1, 7]:  # New Year and Summer sales
                month_factor = random.uniform(0.8, 0.95)
            
            sale_price = final_base_price * month_factor
            
            # Stock quantity with tier-based variation
            stock_range = price_tiers[price_tier]['stock']
            stock_quantity = random.randint(*stock_range)
            
            # Rating with brand tier influence
            rating_range = brand_tiers[brand_tier]['rating_base']
            base_rating = random.uniform(*rating_range)
            
            # Add some random variation
            rating = round(min(5.0, max(1.0, np.random.normal(base_rating, 0.2))), 1)
            
            # Generate creation date with exponential distribution
            days_ago = int(np.random.exponential(7))  # Most products added recently
            days_ago = min(days_ago, 30)
            created_at = self.end_date - timedelta(days=days_ago)
            
            products.append({
                'product_id': product_id,
                'category_id': category_id,
                'subcategory_id': subcategory_id,
                'product_name': f"{self.fake.company()} {self.fake.word().title()}",
                'description': self.fake.text(max_nb_chars=200),
                'base_price': round(final_base_price, 2),
                'sale_price': round(sale_price, 2),
                'stock_quantity': stock_quantity,
                'weight_kg': round(random.uniform(0.1, 20.0), 2),
                'is_active': stock_quantity > 0,
                'created_at': created_at,
                'brand': self.fake.company(),
                'sku': f"SKU-{random.randint(10000, 99999)}",
                'rating': rating,
                'review_count': random.randint(0, 100)
            })
        
        return pd.DataFrame(products), categories_df, subcategories_df

    def generate_customers(self, n_customers=1000, historic_customers_file='de-ecommerce/data/customers.csv'):
        """
        Generate customer data with a mix of returning and new customers
        
        Parameters:
            n_customers: Number of total customers to generate
            historic_customers_file: Path to historic customers CSV file
            
        Returns:
            DataFrame with customer data
        """
        # Try to load historic customers
        try:
            historic_df = pd.read_csv(historic_customers_file)
            print(f"Loaded {len(historic_df)} historic customers")
            
            # Calculate number of returning customers (30% of historic customers)
            n_returning = min(int(len(historic_df) * 0.3), n_customers)
            n_new = n_customers - n_returning
            
            print(f"Generating {n_returning} returning customers and {n_new} new customers")
            
            # Select random returning customers
            returning_customers = historic_df.sample(n=n_returning).copy()
            
            # Update their activity for the recent period
            for idx in returning_customers.index:
                # Generate new login date within the last 30 days
                last_login = self.generate_random_date()
                returning_customers.loc[idx, 'last_login'] = last_login
                
                # Maybe update some other fields that might change
                returning_customers.loc[idx, 'annual_income'] = max(15000, int(np.random.normal(65000, 30000)))
                returning_customers.loc[idx, 'marital_status'] = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
                returning_customers.loc[idx, 'location_type'] = random.choice(['Urban', 'Suburban', 'Rural'])
                returning_customers.loc[idx, 'preferred_channel'] = random.choice(['Web', 'Mobile App', 'Email'])
                returning_customers.loc[idx, 'is_active'] = True  # They're active since they're returning
            
        except FileNotFoundError:
            print("Historic customers file not found. Generating all new customers.")
            n_returning = 0
            n_new = n_customers
            returning_customers = pd.DataFrame()
        
        # Generate new customers
        new_customers = []
        start_id = self.id_offset if not len(returning_customers) else max(returning_customers['customer_id']) + 1
        
        for customer_id in range(start_id, start_id + n_new):
            signup_date = self.generate_random_date()
            last_login = min(signup_date + timedelta(days=random.randint(0, 5)), self.end_date)
            
            new_customers.append({
                'customer_id': customer_id,
                'email': self.fake.email(),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'age': max(18, min(90, int(np.random.normal(45, 15)))),
                'gender': random.choice(['M', 'F', 'Other']),
                'annual_income': max(15000, int(np.random.normal(65000, 30000))),
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
                'education': random.choice(['High School', 'Some College', 'Bachelor', 'Master', 'PhD']),
                'location_type': random.choice(['Urban', 'Suburban', 'Rural']),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'country': 'USA',
                'signup_date': signup_date,
                'last_login': last_login,
                'preferred_channel': random.choice(['Web', 'Mobile App', 'Email']),
                'is_active': random.random() > 0.1
            })
        
        # Combine returning and new customers
        new_customers_df = pd.DataFrame(new_customers)
        final_customers_df = pd.concat([returning_customers, new_customers_df], ignore_index=True)
        
        print(f"Generated total of {len(final_customers_df)} customers")
        print(f"- Returning customers: {len(returning_customers)}")
        print(f"- New customers: {len(new_customers_df)}")
        
        return final_customers_df

    def generate_orders(self, customers_df, products_df):
        """Generate order data for the last 30 days"""
        orders = []
        order_items = []
        order_id = 1 + self.id_offset
        
        # Seasonality patterns
        seasonality = {
            1: 1.2,   # Post-holiday sales
            2: 0.8,   # Slow month
            3: 0.9,   # Spring beginning
            4: 1.0,   # Regular
            5: 1.1,   # Pre-summer
            6: 1.15,  # Summer start
            7: 1.05,  # Mid-summer
            8: 1.1,   # Back to school
            9: 1.0,   # Regular
            10: 1.1,  # Pre-holiday
            11: 1.3,  # Black Friday
            12: 1.5   # Holiday season
        }
        
        # Day of week patterns
        daily_factors = {
            0: 0.7,  # Monday
            1: 0.8,  # Tuesday
            2: 0.9,  # Wednesday
            3: 1.0,  # Thursday
            4: 1.3,  # Friday
            5: 1.5,  # Saturday
            6: 1.1   # Sunday
        }
        
        for _, customer in customers_df.iterrows():
            # Order frequency based on customer segment
            base_frequency = {
                'High Value': 8,
                'Mid Value': 5,
                'Low Value': 3
            }.get(customer['customer_segment'], 5)
            
            # Reduce frequency for 30-day period
            base_frequency = max(1, int(base_frequency * (30/365)))
            
            # Add randomness to frequency
            order_count = np.random.poisson(base_frequency)
            
            for _ in range(order_count):
                # Generate order date with patterns
                order_date = max(
                    customer['signup_date'],
                    customer['signup_date'] + timedelta(
                        days=random.randint(0, (self.end_date - customer['signup_date']).days)
                    )
                )
                
                # Apply seasonality and day of week factors
                season_factor = seasonality[order_date.month]
                day_factor = daily_factors[order_date.weekday()]
                
                # Generate order items
                num_items = np.random.poisson(2) + 1
                
                # Product selection with category affinity
                if random.random() < 0.7:  # 70% chance to buy from preferred category
                    preferred_category = random.randint(1, 5)
                    matching_products = products_df[
                        products_df['category_id'] == preferred_category
                    ]
                    if len(matching_products) > 0:
                        order_products = matching_products.sample(
                            n=min(num_items, len(matching_products))
                        )
                    else:
                        order_products = products_df.sample(n=num_items)
                else:
                    order_products = products_df.sample(n=num_items)
                
                shipping_cost = round(random.uniform(5, 20), 2)
                total_amount = shipping_cost
                
                # Generate order items
                for _, product in order_products.iterrows():
                    quantity = np.random.poisson(1.5) + 1
                    price = product['sale_price']
                    item_total = quantity * price
                    total_amount += item_total
                    
                    order_items.append({
                        'order_item_id': len(order_items) + 1 + self.id_offset,
                        'order_id': order_id,
                        'product_id': product['product_id'],
                        'quantity': quantity,
                        'unit_price': round(price, 2),
                        'total_price': round(item_total, 2),
                        'created_at': order_date
                    })
                
                # Order status based on date
                days_since_order = (self.end_date - order_date).days
                if days_since_order < 1:
                    status = 'Pending'
                elif days_since_order < 2:
                    status = 'Processing'
                elif days_since_order < 4:
                    status = 'Shipped'
                else:
                    status = 'Delivered'
                
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
                    'updated_at': order_date + timedelta(minutes=random.randint(5, 60))
                })
                
                order_id += 1
        
        return pd.DataFrame(orders), pd.DataFrame(order_items)

    def generate_customer_interactions(self, customers_df, products_df):
        """Generate customer interactions for the last 30 days"""
        events = []
        
        # Time-of-day patterns
        hourly_weights = [
            0.01, 0.01, 0.005, 0.005, 0.01, 0.02,  # 0-5
            0.03, 0.05, 0.07, 0.08, 0.09, 0.10,    # 6-11
            0.11, 0.10, 0.09, 0.08, 0.07, 0.06,    # 12-17
            0.08, 0.09, 0.08, 0.05, 0.03, 0.02     # 18-23
        ]
        
        # Device preferences by hour
        device_by_hour = {
            'morning': {'desktop': 0.4, 'mobile': 0.5, 'tablet': 0.1},
            'workday': {'desktop': 0.6, 'mobile': 0.3, 'tablet': 0.1},
            'evening': {'desktop': 0.2, 'mobile': 0.6, 'tablet': 0.2}
        }
        
        # Event flow with realistic browsing patterns
        event_flow = {
            'view': {'cart_add': 0.2, 'view': 0.7, 'search': 0.1},
            'cart_add': {'purchase': 0.3, 'cart_remove': 0.2, 'view': 0.5},
            'cart_remove': {'view': 0.8, 'search': 0.2},
            'search': {'view': 0.9, 'cart_add': 0.1}
        }
        
        for _, customer in customers_df.iterrows():
            # Vary session count by customer segment
            if customer['customer_segment'] == 'High Value':
                session_count = np.random.poisson(12)
            elif customer['customer_segment'] == 'Mid Value':
                session_count = np.random.poisson(8)
            else:
                session_count = np.random.poisson(4)
            
            for _ in range(session_count):
                session_id = f"session_{random.randint(10000, 99999)}"
                
                # Generate session start time
                hour = random.choices(range(24), weights=hourly_weights)[0]
                
                # Select device based on time
                if 6 <= hour <= 11:
                    device_weights = device_by_hour['morning']
                elif 12 <= hour <= 17:
                    device_weights = device_by_hour['workday']
                else:
                    device_weights = device_by_hour['evening']
                
                device_type = random.choices(
                    ['desktop', 'mobile', 'tablet'],
                    weights=[device_weights[d] for d in ['desktop', 'mobile', 'tablet']]
                )[0]
                
                # Generate session events
                event_time = customer['signup_date'] + timedelta(
                    days=random.randint(0, (self.end_date - customer['signup_date']).days),
                    hours=hour,
                    minutes=random.randint(0, 59)
                )
                
                current_event = 'view'
                viewed_products = []
                
                # Generate sequence of events
                num_events = np.random.poisson(4) + 1
                
                for _ in range(num_events):
                    # Product selection with category affinity
                    if viewed_products and random.random() < 0.7:
                        # 70% chance to view related product
                        last_product = products_df[
                            products_df['product_id'] == viewed_products[-1]
                        ].iloc[0]
                        related_products = products_df[
                            products_df['category_id'] == last_product['category_id']
                        ]
                        product = related_products.sample().iloc[0]
                    else:
                        product = products_df.sample().iloc[0]
                    
                    viewed_products.append(product['product_id'])
                    
                    events.append({
                        'event_id': len(events) + 1 + self.id_offset,
                        'customer_id': customer['customer_id'],
                        'product_id': product['product_id'],
                        'event_type': current_event,
                        'event_date': event_time,
                        'device_type': device_type,
                        'session_id': session_id,
                        'created_at': event_time
                    })
                    
                    # Choose next event
                    next_event_probs = event_flow[current_event]
                    current_event = random.choices(
                        list(next_event_probs.keys()),
                        weights=list(next_event_probs.values())
                    )[0]
                    
                    # Increment time realistically
                    event_time += timedelta(minutes=random.randint(1, 10))
        
        return pd.DataFrame(events)
    
    def generate_review_text(self, rating, product_info, days_to_review):
        """Generate review text with appropriate context for recent reviews"""
        category_specific_aspects = {
            'Electronics': {
                'positive': ["battery life is impressive", "interface is intuitive", "fast performance", 
                            "build quality is premium", "setup was easy", "great features"],
                'negative': ["battery drains quickly", "confusing interface", "slow performance", 
                            "feels cheaply made", "difficult setup", "missing features"]
            },
            'Fashion': {
                'positive': ["perfect fit", "high quality material", "beautiful design", 
                            "comfortable to wear", "well made", "true to size"],
                'negative': ["poor fit", "cheap material", "looks different from picture", 
                            "uncomfortable", "poorly made", "sizing is off"]
            },
            'Home & Living': {
                'positive': ["great quality", "looks beautiful", "easy to assemble", 
                            "perfect size", "well designed", "durable"],
                'negative': ["poor quality", "looks cheap", "difficult assembly", 
                            "wrong size", "bad design", "broke easily"]
            }
        }

        # Time-based context phrases
        time_context = {
            'quick': ["Just got this", "After quick testing", "First impression", 
                    "Day one review", "Initial thoughts"],
            'short': ["After a few days", "Short term use", "Early review", 
                    "Quick update", "First week thoughts"]
        }

        # Select appropriate context
        time_phrase = random.choice(time_context['quick'] if days_to_review <= 2 else time_context['short'])
        
        # Get category-specific aspects
        category = product_info.get('category_name', 'Electronics')
        aspects = category_specific_aspects.get(category, category_specific_aspects['Electronics'])
        
        if rating >= 4:
            templates = [
                "{time_phrase}: {positive_aspect}. {conclusion}",
                "{time_phrase} and I'm impressed. {positive_aspect} and {another_positive}. {conclusion}",
                "{positive_aspect}. {time_phrase} and {another_positive}. {conclusion}",
                "Great purchase! {time_phrase} - {positive_aspect}. {conclusion}"
            ]
            positive_aspects = aspects['positive']
            conclusions = ["Highly recommend!", "Very satisfied!", "Great product!", "Excellent purchase!"]
            
            return random.choice(templates).format(
                time_phrase=time_phrase,
                positive_aspect=random.choice(positive_aspects),
                another_positive=random.choice(positive_aspects),
                conclusion=random.choice(conclusions)
            )
        
        elif rating >= 3:
            templates = [
                "{time_phrase} and it's decent. {positive_aspect}, but {negative_aspect}. {conclusion}",
                "{time_phrase}: {positive_aspect}. However, {negative_aspect}. {conclusion}",
                "Mixed feelings. {time_phrase} - {positive_aspect}, though {negative_aspect}. {conclusion}"
            ]
            conclusions = ["It's okay.", "Might work for others.", "Still evaluating.", "We'll see how it holds up."]
            
            return random.choice(templates).format(
                time_phrase=time_phrase,
                positive_aspect=random.choice(aspects['positive']),
                negative_aspect=random.choice(aspects['negative']),
                conclusion=random.choice(conclusions)
            )
        
        else:
            templates = [
                "{time_phrase} and I'm disappointed. {negative_aspect}. {conclusion}",
                "Not happy with this purchase. {time_phrase} and {negative_aspect}. {conclusion}",
                "{time_phrase}: {negative_aspect}. Would not recommend. {conclusion}"
            ]
            conclusions = ["Returning this.", "Save your money.", "Looking for alternatives.", "Not worth it."]
            
            return random.choice(templates).format(
                time_phrase=time_phrase,
                negative_aspect=random.choice(aspects['negative']),
                conclusion=random.choice(conclusions)
            )

    def generate_reviews(self, orders_df, order_items_df, products_df, customers_df):
        """Generate reviews with rich text content for recent data"""
        reviews_data = []
        review_id = 1 + self.id_offset
        
        # Review probability by customer segment and price tier
        review_prob = {
            'High Value': {'base': 0.8, 'price_sensitivity': 0.1},
            'Mid Value': {'base': 0.6, 'price_sensitivity': 0.2},
            'Low Value': {'base': 0.4, 'price_sensitivity': 0.3}
        }
        
        # Rating distributions by price range and customer segment
        def get_rating_params(price, segment):
            if segment == 'High Value':
                return (4.2, 0.8) if price > 500 else (4.0, 0.7)
            elif segment == 'Mid Value':
                return (4.0, 0.9) if price > 500 else (3.8, 0.8)
            else:
                return (3.8, 1.0) if price > 500 else (3.5, 0.9)
        
        for _, order in orders_df.iterrows():
            if order['status'] != 'Delivered':
                continue
                
            customer = customers_df[customers_df['customer_id'] == order['customer_id']].iloc[0]
            segment = customer.get('customer_segment', 'Mid Value')
            
            order_items = order_items_df[order_items_df['order_id'] == order['order_id']]
            
            for _, item in order_items.iterrows():
                product = products_df[products_df['product_id'] == item['product_id']].iloc[0]
                
                # Calculate review probability
                base_prob = review_prob[segment]['base']
                price_factor = 1 + (review_prob[segment]['price_sensitivity'] * 
                                (1 if product['base_price'] > 100 else -1))
                final_prob = min(0.95, base_prob * price_factor)
                
                if random.random() < final_prob:
                    # Generate rating based on customer segment and price
                    mean, std = get_rating_params(product['base_price'], segment)
                    review_score = round(min(5, max(1, np.random.normal(mean, std))), 1)
                    
                    # Calculate review date with realistic delay for recent data
                    days_to_review = int(np.random.exponential(3))  # Most reviews within 3 days for recent data
                    days_to_review = min(days_to_review, 7)  # Cap at 7 days for recent data
                    review_date = order['order_date'] + timedelta(days=days_to_review)
                    
                    if review_date <= self.end_date:
                        review_text = self.generate_review_text(
                            review_score,
                            product,
                            days_to_review
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