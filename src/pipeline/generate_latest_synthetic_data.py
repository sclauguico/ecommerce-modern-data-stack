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

def set_all_seeds(seed=42):
    """
    Set all seeds to ensure reproducibility across all random number generators
    """
    import random
    import numpy as np
    import torch
    import os
    
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Faker (if used)
    from faker import Faker
    Faker.seed(seed)
    
    # PyTorch (if used)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

class RecentEcommerceDataGenerator:
    def __init__(self, id_offset=1000000, seed=42):  # Add a large offset to some IDs
        # Set all seeds first
        set_all_seeds(seed)
        
        # Store seed for reference
        self.seed = seed
        
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

    def generate_categories(self):
        """Generate product categories with simplified output structure"""
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
            # Generate category record
            category_data.append({
                'category_id': cat_id,
                'category_name': category,
                'created_at': datetime(2023, 1, 1)
            })
            
            # Generate subcategory records
            for sub_id, subcategory in enumerate(subcategories, 1):
                subcategory_data.append({
                    'subcategory_id': (cat_id * 100) + sub_id,
                    'category_id': cat_id,
                    'subcategory_name': subcategory,
                    'created_at': datetime(2023, 1, 1)
                })
        
        return pd.DataFrame(category_data), pd.DataFrame(subcategory_data)

    def generate_random_date(self):
        """Generate a random date within the last 30 days"""
        days_offset = random.randint(0, 30)
        return self.end_date - timedelta(days=days_offset)

    def generate_products(self, n_products=50):
        """Generate product catalog with recent dates and sophisticated pricing"""
        categories_df, subcategories_df = self.generate_categories()
        
        # Category-specific price ranges
        category_patterns = {
            'Electronics': {
                'name_patterns': [
                    "{brand} {model} {type}",
                    "{brand} {type} {series}",
                    "{brand} {type} ({feature})"
                ],
                'types': {
                    'Smartphones': ['Smartphone', 'Phone', 'Mobile Phone', '5G Phone', 'Foldable Phone'],
                    'Laptops': ['Laptop', 'Notebook', 'Ultrabook', 'Gaming Laptop', 'Workstation'],
                    'Accessories': ['Charger', 'Case', 'Screen Protector', 'Stand', 'Adapter'],
                    'Tablets': ['Tablet', 'iPad', 'Android Tablet', 'Drawing Tablet', 'Mini Tablet'],
                    'Wearables': ['Smartwatch', 'Fitness Tracker', 'Smart Band', 'Sport Watch', 'Health Monitor']
                },
                'features': ['Pro', 'Ultra', 'Max', 'Lite', 'Plus', '5G', 'Wireless', 'Premium'],
                'models': ['X', 'Pro', 'Elite', 'Max', 'Ultra', 'SE', 'Air'],
                'series': ['Series ' + str(i) for i in range(1, 10)] + ['2023 Edition', '2024 Edition']
            },
            'Fashion': {
                'name_patterns': [
                    "{brand} {material} {type}",
                    "{brand} {style} {type}",
                    "{brand} {type} in {color}"
                ],
                'types': {
                    "Men's Clothing": ['T-Shirt', 'Jacket', 'Jeans', 'Sweater', 'Polo Shirt', 'Hoodie'],
                    "Women's Clothing": ['Dress', 'Blouse', 'Skirt', 'Leggings', 'Cardigan', 'Top'],
                    "Children's Clothing": ['T-Shirt', 'Pants', 'Dress', 'Pajamas', 'Sweater'],
                    'Shoes': ['Sneakers', 'Boots', 'Sandals', 'Loafers', 'Running Shoes'],
                    'Accessories': ['Belt', 'Scarf', 'Hat', 'Bag', 'Wallet']
                },
                'materials': ['Cotton', 'Leather', 'Denim', 'Silk', 'Wool', 'Linen'],
                'styles': ['Casual', 'Formal', 'Sport', 'Classic', 'Modern', 'Vintage'],
                'colors': ['Black', 'Navy', 'Brown', 'Grey', 'White', 'Beige']
            },
            'Home & Living': {
                'name_patterns': [
                    "{brand} {type} {material}",
                    "{brand} {style} {type}",
                    "{brand} {type} {collection}"
                ],
                'types': {
                    'Furniture': ['Sofa', 'Chair', 'Table', 'Bed', 'Desk', 'Shelf'],
                    'Kitchen': ['Pot Set', 'Pan', 'Knife Set', 'Blender', 'Coffee Maker'],
                    'Decor': ['Vase', 'Wall Art', 'Mirror', 'Lamp', 'Cushion'],
                    'Bedding': ['Duvet Cover', 'Sheet Set', 'Pillow', 'Blanket', 'Comforter'],
                    'Storage': ['Cabinet', 'Organizer', 'Box Set', 'Basket', 'Shelf Unit']
                },
                'materials': ['Wood', 'Metal', 'Glass', 'Ceramic', 'Bamboo', 'Cotton'],
                'styles': ['Modern', 'Classic', 'Rustic', 'Minimalist', 'Contemporary'],
                'collections': ['Home', 'Essential', 'Premium', 'Designer', 'Classic']
            },
            'Beauty': {
                'name_patterns': [
                    "{brand} {type} {variant}",
                    "{brand} {benefit} {type}",
                    "{brand} {type} with {ingredient}"
                ],
                'types': {
                    'Skincare': ['Cleanser', 'Moisturizer', 'Serum', 'Mask', 'Toner'],
                    'Makeup': ['Foundation', 'Lipstick', 'Mascara', 'Eyeshadow', 'Blush'],
                    'Haircare': ['Shampoo', 'Conditioner', 'Hair Mask', 'Hair Oil', 'Styling Cream'],
                    'Fragrances': ['Perfume', 'Eau de Toilette', 'Body Mist', 'Cologne'],
                    'Tools': ['Brush Set', 'Hair Dryer', 'Curling Iron', 'Facial Roller']
                },
                'benefits': ['Hydrating', 'Anti-aging', 'Brightening', 'Purifying', 'Nourishing'],
                'ingredients': ['Vitamin C', 'Hyaluronic Acid', 'Retinol', 'Collagen', 'Aloe Vera'],
                'variants': ['Premium', 'Natural', 'Sensitive', 'Advanced', 'Professional']
            },
            'Sports': {
                'name_patterns': [
                    "{brand} {type} {level}",
                    "{brand} {sport} {type}",
                    "{brand} {type} {technology}"
                ],
                'types': {
                    'Exercise Equipment': ['Treadmill', 'Bike', 'Weights', 'Yoga Mat', 'Resistance Bands'],
                    'Sportswear': ['T-Shirt', 'Shorts', 'Leggings', 'Jacket', 'Tank Top'],
                    'Outdoor Gear': ['Tent', 'Backpack', 'Sleeping Bag', 'Camping Stove'],
                    'Accessories': ['Water Bottle', 'Fitness Tracker', 'Sports Bag', 'Gloves'],
                    'Footwear': ['Running Shoes', 'Training Shoes', 'Hiking Boots', 'Cleats']
                },
                'levels': ['Pro', 'Elite', 'Amateur', 'Professional', 'Competition'],
                'sports': ['Running', 'Training', 'Yoga', 'Basketball', 'Football'],
                'technology': ['Air', 'Flex', 'Tech', 'Lite', 'Pro']
            }
        }

        # Category-specific price ranges
        category_prices = {
            1: {'range': (500, 2000), 'name': 'Electronics'},  # Electronics
            2: {'range': (20, 200), 'name': 'Fashion'},   # Fashion
            3: {'range': (50, 1000), 'name': 'Home & Living'},  # Home & Living
            4: {'range': (10, 100), 'name': 'Beauty'},    # Beauty
            5: {'range': (30, 300), 'name': 'Sports'}     # Sports
        }

        # Price tier definitions
        price_tiers = {
            'budget': {'weight': 0.3, 'multiplier': (0.6, 0.8)},
            'mid_range': {'weight': 0.5, 'multiplier': (0.9, 1.2)},
            'premium': {'weight': 0.2, 'multiplier': (1.3, 2.0)}
        }

        # Context data for descriptions
        context_data = {
            'Electronics': {
                'use_cases': ['daily use', 'professional work', 'gaming', 'content creation', 'business'],
                'features': ['latest processor', 'high-resolution display', 'long battery life', 'fast charging', '5G connectivity']
            },
            'Fashion': {
                'occasions': ['casual wear', 'formal events', 'office wear', 'outdoor activities', 'special occasions']
            },
            'Home & Living': {
                'rooms': ['living room', 'bedroom', 'home office', 'kitchen', 'dining room']
            },
            'Beauty': {
                'skin_types': ['all skin types', 'sensitive skin', 'dry skin', 'oily skin', 'combination skin']
            },
            'Sports': {
                'sports': ['running', 'training', 'yoga', 'hiking', 'gym workouts']
            }
        }
        
        # Description templates with simplified parameter usage
        description_templates = {
            'Electronics': [
                "Premium {product_type}. {feature_desc}",
                "High-performance {product_type}. {feature_desc}",
                "Advanced {product_type}. {feature_desc}"
            ],
            'Fashion': [
                "Stylish {product_type} made from {material}. Perfect for any occasion.",
                "Premium quality {product_type} in {color}. Versatile and comfortable.",
                "Comfortable {product_type} with {style} design. Modern and fashionable."
            ],
            'Home & Living': [
                "Beautiful {product_type} in {material}. Perfect addition to your home.",
                "{style} {product_type} that adds elegance to any room.",
                "High-quality {product_type} from our {collection} collection."
            ],
            'Beauty': [
                "{benefit} {product_type} enriched with {ingredient}. For all skin types.",
                "Professional-grade {product_type} featuring {ingredient}.",
                "Advanced {product_type} with {benefit} properties."
            ],
            'Sports': [
                "Professional-grade {product_type}. Features {technology}.",
                "High-performance {product_type}. {technology} technology.",
                "Premium {product_type}. Enhanced with {technology} for optimal performance."
            ]
        }

        # Feature descriptions for electronics
        feature_descriptions = [
            "Features advanced {feature} technology for enhanced performance",
            "Includes {feature} capabilities for professional use",
            "Enhanced with {feature} for superior functionality",
            "Built with {feature} for optimal performance",
            "Integrated {feature} technology for better results"
        ]

        products = []
        existing_brands = set()

        for product_id in range(1, n_products + 1):
            created_at = self.generate_random_date()
            category_id = random.randint(1, len(categories_df))
            category_name = category_prices[category_id]['name']
            category_pattern = category_patterns[category_name]
            
            valid_subcats = subcategories_df[subcategories_df['category_id'] == category_id]
            subcategory = random.choice(valid_subcats['subcategory_name'].values)
            subcategory_id = valid_subcats[valid_subcats['subcategory_name'] == subcategory]['subcategory_id'].iloc[0]

            # Brand generation with 70% reuse rate for consistency
            if random.random() < 0.7 and existing_brands:
                brand = random.choice(list(existing_brands))
            else:
                brand = f"{self.fake.company()} {random.choice(['', 'Pro', 'Elite', 'Premium'])}"
                existing_brands.add(brand)

            # Product name generation
            name_pattern = random.choice(category_pattern['name_patterns'])
            product_type = random.choice(category_pattern['types'][subcategory])
            
            name_params = {
                'brand': brand,
                'type': product_type,
                'material': random.choice(category_pattern.get('materials', [''])),
                'style': random.choice(category_pattern.get('styles', [''])),
                'color': random.choice(category_pattern.get('colors', [''])),
                'feature': random.choice(category_pattern.get('features', [''])),
                'model': random.choice(category_pattern.get('models', [''])),
                'series': random.choice(category_pattern.get('series', [''])),
                'collection': random.choice(category_pattern.get('collections', [''])),
                'variant': random.choice(category_pattern.get('variants', [''])),
                'level': random.choice(category_pattern.get('levels', [''])),
                'sport': random.choice(category_pattern.get('sports', [''])),
                'technology': random.choice(category_pattern.get('technology', [''])),
                'benefit': random.choice(category_pattern.get('benefits', [''])),
                'ingredient': random.choice(category_pattern.get('ingredients', ['']))
            }
            
            product_name = name_pattern.format(**{k: v for k, v in name_params.items() if v})

            # Description generation
            desc_template = random.choice(description_templates[category_name])
            desc_params = {
                'product_type': product_type,
                'material': name_params.get('material', ''),
                'style': name_params.get('style', ''),
                'color': name_params.get('color', ''),
                'technology': name_params.get('technology', ''),
                'benefit': name_params.get('benefit', ''),
                'ingredient': name_params.get('ingredient', ''),
                'collection': name_params.get('collection', '')
            }
            
            if category_name == 'Electronics':
                feature_template = random.choice(feature_descriptions)
                desc_params['feature_desc'] = feature_template.format(
                    feature=name_params.get('feature', 'advanced')
                )
            
            description = desc_template.format(**{k: v for k, v in desc_params.items() if v})

            # Price calculations with seasonal adjustments for recent period
            price_tier = random.choices(
                list(price_tiers.keys()),
                weights=[t['weight'] for t in price_tiers.values()]
            )[0]
            tier_props = price_tiers[price_tier]
            
            # Base price calculation
            price_range = category_prices[category_id]['range']
            base_price = random.uniform(*price_range)
            price_multiplier = random.uniform(*tier_props['multiplier'])
            final_base_price = base_price * price_multiplier
            
            # Seasonal adjustments for current month
            current_month = created_at.month
            seasonal_boost = 1.0
            if current_month in [11, 12]:  # Holiday season
                seasonal_boost = random.uniform(1.1, 1.3)
            elif current_month in [7, 8]:  # Summer sales
                seasonal_boost = random.uniform(0.7, 0.9)
            
            sale_price = final_base_price * random.uniform(0.8, 0.95) * seasonal_boost

            # Stock quantity based on tier and recent period
            stock_quantity = {
                'premium': random.randint(10, 200),
                'mid_range': random.randint(50, 500),
                'budget': random.randint(100, 1000)
            }[price_tier]

            # Ratings and reviews adjusted for 30-day window
            base_rating = np.random.normal(4.3 if price_tier == 'premium' else 4.0, 0.4)
            rating = round(min(5.0, max(1.0, base_rating)), 1)
            # Reduced review count for 30-day period
            review_count = random.randint(0, 100) if rating >= 4.0 else random.randint(0, 30)

            products.append({
                'product_id': product_id + self.id_offset,
                'category_id': category_id,
                'subcategory_id': subcategory_id,
                'product_name': product_name.strip(),
                'description': description.strip(),
                'base_price': round(final_base_price, 2),
                'sale_price': round(sale_price, 2),
                'stock_quantity': stock_quantity,
                'weight_kg': round(random.uniform(0.1, 20.0), 2),
                'is_active': random.random() > 0.1,
                'created_at': created_at,
                'brand': brand,
                'sku': f"SKU-{category_id}{subcategory_id}-{random.randint(10000, 99999)}",
                'rating': rating,
                'review_count': review_count
            })
        
        return pd.DataFrame(products), categories_df, subcategories_df

    def generate_customers(self, n_customers=200):
        """Generate new customer data for the 30-day window"""
        
        # Define age and demographic segments
        age_segments = {
            'Gen Z': {'range': (18, 25), 'weight': 0.20, 'income_range': (20000, 45000)},
            'Young Millennials': {'range': (26, 32), 'weight': 0.25, 'income_range': (35000, 60000)},
            'Older Millennials': {'range': (33, 40), 'weight': 0.20, 'income_range': (50000, 90000)},
            'Gen X': {'range': (41, 56), 'weight': 0.20, 'income_range': (60000, 120000)},
            'Boomers': {'range': (57, 75), 'weight': 0.15, 'income_range': (45000, 100000)}
        }

        # Education distribution by age segment
        education_dist = {
            'Gen Z': {'High School': 0.2, 'Some College': 0.4, 'Bachelor': 0.3, 'Master': 0.1},
            'Young Millennials': {'High School': 0.15, 'Some College': 0.3, 'Bachelor': 0.4, 'Master': 0.15},
            'Older Millennials': {'High School': 0.1, 'Some College': 0.25, 'Bachelor': 0.45, 'Master': 0.2},
            'Gen X': {'High School': 0.2, 'Some College': 0.3, 'Bachelor': 0.35, 'Master': 0.15},
            'Boomers': {'High School': 0.25, 'Some College': 0.35, 'Bachelor': 0.3, 'Master': 0.1}
        }

        # Location type distribution
        location_dist = {
            'Urban': 0.45,
            'Suburban': 0.40,
            'Rural': 0.15
        }

        # Channel preferences by age segment
        channel_dist = {
            'Gen Z': {'Mobile App': 0.6, 'Web': 0.3, 'Email': 0.1},
            'Young Millennials': {'Mobile App': 0.5, 'Web': 0.4, 'Email': 0.1},
            'Older Millennials': {'Mobile App': 0.4, 'Web': 0.4, 'Email': 0.2},
            'Gen X': {'Mobile App': 0.3, 'Web': 0.4, 'Email': 0.3},
            'Boomers': {'Mobile App': 0.2, 'Web': 0.5, 'Email': 0.3}
        }

        customers = []
        for customer_id in range(1, n_customers + 1):
            # Select age segment
            age_segment = random.choices(
                list(age_segments.keys()),
                weights=[s['weight'] for s in age_segments.values()]
            )[0]
            segment_props = age_segments[age_segment]
            
            # Generate age within segment
            age = random.randint(*segment_props['range'])
            
            # Generate income with normal distribution within range
            income_range = segment_props['income_range']
            income = int(np.random.normal(
                (income_range[0] + income_range[1]) / 2,
                (income_range[1] - income_range[0]) / 4
            ))
            income = max(income_range[0], min(income_range[1], income))
            
            # Select education based on age segment
            education = random.choices(
                list(education_dist[age_segment].keys()),
                weights=list(education_dist[age_segment].values())
            )[0]
            
            # Select location type
            location_type = random.choices(
                list(location_dist.keys()),
                weights=list(location_dist.values())
            )[0]
            
            # Select preferred channel based on age segment
            channel_weights = channel_dist[age_segment]
            preferred_channel = random.choices(
                list(channel_weights.keys()),
                weights=list(channel_weights.values())
            )[0]
            
            # Generate signup date
            days_range = (self.end_date - self.start_date).days
            signup_date = self.start_date + timedelta(days=random.randint(0, days_range))

            # Calculate segment but don't include in final data
            segment = 'High Value' if income > 80000 else ('Mid Value' if income > 50000 else 'Low Value')
            
            customers.append({
                'customer_id': customer_id + self.id_offset,
                'email': self.fake.email(),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'age': age,
                'gender': random.choice(['M', 'F', 'Other']),
                'annual_income': income,
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
                'education': education,
                'location_type': random.choice(['Urban', 'Suburban', 'Rural']),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'country': 'USA',
                'signup_date': signup_date,
                'last_login': signup_date + timedelta(days=random.randint(0, min(30, (self.end_date - signup_date).days))),
                'preferred_channel': preferred_channel,
                'is_active': True  # All new customers are active
            })
        
        return pd.DataFrame(customers)

    def generate_orders(self, customers_df, products_df):
        """Generate order data for the last 30 days"""
        orders = []
        order_items = []
        order_id = 1 + self.id_offset
        
        # Try to load historic customers
        try:
            root_dir = os.getcwd()
            historic_customers_file = os.path.join(root_dir, "generated_historic_data", "customers.csv")
            historic_df = pd.read_csv(historic_customers_file)
                        
            # Select active historic customers (20% of historic customers)
            n_historic_active = int(len(historic_df) * 0.2)
            active_historic_customers = historic_df.sample(n=n_historic_active)
            
            # Combine with new customers
            all_customers = pd.concat([customers_df, active_historic_customers], ignore_index=True)
        except Exception as e:
            print(f"Could not load historic customers: {str(e)}")
            print("Using only new customers")
            all_customers = customers_df
                
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
        
        # Add yearly growth trend
        base_growth_rate = 1.15  # 15% annual growth
        
        for _, customer in all_customers.iterrows():
            # Determine if customer is new or returning
            is_new = customer['customer_id'] >= self.id_offset
            
            # Calculate available date range for orders
            signup_date = pd.to_datetime(customer['signup_date'])
            
            # Adjust order frequency based on customer type
            income_factor = min(2.0, max(0.5, customer['annual_income'] / 65000))
            if is_new:
                # New customers have more variable engagement
                customer_frequency = np.random.poisson(3 * income_factor)  # Lower initial order frequency
            else:
                # Returning customers have more established patterns
                customer_frequency = np.random.poisson(7 * income_factor)  # Higher order frequency
            
            for _ in range(customer_frequency):
                # Generate order date based on customer type
                if signup_date >= self.start_date:
                    # New customer - orders start after signup
                    available_days = (self.end_date - signup_date).days
                    order_date = signup_date + timedelta(days=random.randint(0, max(1, available_days)))
                else:
                    # Returning customer - orders within the 30-day window
                    order_date = self.start_date + timedelta(days=random.randint(0, 29))
                
                # Apply seasonality and trends
                month_factor = seasonality[order_date.month]
                day_factor = daily_factors[order_date.weekday()]
                days_since_start = (order_date - self.start_date).days
                years_since_start = max(0.1, days_since_start / 365.0)
                growth_factor = base_growth_rate ** years_since_start
                
                # Combined factor for order value
                total_factor = month_factor * day_factor * growth_factor
                
                # Order status based on recency
                days_since_order = (self.end_date - order_date).days
                if days_since_order < 2:
                    status = 'Pending'
                elif days_since_order < 4:
                    status = 'Processing'
                elif days_since_order < 7:
                    status = 'Shipped'
                else:
                    status = 'Delivered'
                
                shipping_cost = round(random.uniform(5, 20) * growth_factor, 2)
                
                # Generate order items with product affinity
                # New customers tend to buy fewer items
                base_items = 2 if is_new else 3
                num_items = np.random.poisson(base_items) + 1
                
                # Category preference more pronounced for returning customers
                category_preference = random.randint(1, 5)
                preferred_products = products_df[products_df['category_id'] == category_preference]
                
                # Returning customers more likely to buy from preferred category
                category_probability = 0.5 if is_new else 0.8
                if len(preferred_products) > 0 and random.random() < category_probability:
                    order_products = preferred_products.sample(n=min(num_items, len(preferred_products)), replace=True)
                else:
                    order_products = products_df.sample(n=min(num_items, len(products_df)), replace=True)
                
                total_amount = shipping_cost
                for _, product in order_products.iterrows():
                    # Returning customers tend to buy in larger quantities
                    base_quantity = 1.2 if is_new else 1.8
                    quantity = np.random.poisson(base_quantity) + 1
                    
                    price = product['sale_price'] * (1 + random.uniform(-0.1, 0.1))
                    item_total = quantity * price * total_factor
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
                
                # Payment method preferences
                if is_new:
                    payment_methods = ['Credit Card', 'PayPal', 'Debit Card']
                    weights = [0.5, 0.3, 0.2]  # New customers prefer credit cards
                else:
                    payment_methods = ['Credit Card', 'PayPal', 'Debit Card']
                    weights = [0.4, 0.4, 0.2]  # Returning customers more likely to use PayPal
                
                payment_method = random.choices(payment_methods, weights=weights)[0]
                
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

    def generate_interactions(self, customers_df, products_df):
        """Generate customer interactions for the last 30 days for both new and historic customers"""
        events = []
        event_id = 1 + self.id_offset

        # Try to load historic customers from S3
        try:
            root_dir = os.getcwd()
            historic_customers_file = os.path.join(root_dir, "generated_historic_data", "customers.csv")
            historic_df = pd.read_csv(historic_customers_file)
            
            # print(f"Loaded {len(historic_df)} historic")
            
            # Select active historic customers (20% of historic customers)
            n_historic_active = int(len(historic_df) * 0.2)
            active_historic_customers = historic_df.sample(n=n_historic_active)
            
            # Combine with new customers
            all_customers = pd.concat([customers_df, active_historic_customers], ignore_index=True)
        except Exception as e:
            print(f"Could not load historic customers: {str(e)}")
            print("Using only new customers")
            all_customers = customers_df

        # Time-of-day patterns
        hourly_patterns = {
            'desktop': {  # Peak during work hours
                'distribution': [0.01, 0.01, 0.005, 0.005, 0.01, 0.02,  # 0-5
                            0.03, 0.05, 0.07, 0.08, 0.09, 0.10,    # 6-11
                            0.11, 0.10, 0.09, 0.08, 0.07, 0.06,    # 12-17
                            0.08, 0.09, 0.08, 0.05, 0.03, 0.02]    # 18-23
            },
            'mobile': {   # More evening activity
                'distribution': [0.02, 0.01, 0.01, 0.01, 0.01, 0.02,
                            0.04, 0.06, 0.07, 0.08, 0.07, 0.08,
                            0.09, 0.08, 0.07, 0.08, 0.09, 0.10,
                            0.12, 0.11, 0.09, 0.08, 0.05, 0.03]
            },
            'tablet': {   # Evening peak
                'distribution': [0.02, 0.01, 0.01, 0.01, 0.01, 0.02,
                            0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                            0.07, 0.08, 0.07, 0.08, 0.09, 0.11,
                            0.13, 0.12, 0.10, 0.08, 0.06, 0.03]
            }
        }
        
        # Event flow patterns - different for new vs returning customers
        new_customer_flow = {
            'view': {'cart_add': 0.2, 'view': 0.7, 'search': 0.1},  # More browsing
            'cart_add': {'view': 0.4, 'cart_remove': 0.3, 'purchase': 0.1, 'search': 0.2},  # More hesitant
            'cart_remove': {'view': 0.8, 'search': 0.2},
            'search': {'view': 0.8, 'cart_add': 0.2},
            'purchase': {'view': 0.7, 'search': 0.3},
            'start': {'view': 0.6, 'search': 0.4}  # More likely to start with search
        }
        
        returning_customer_flow = {
            'view': {'cart_add': 0.4, 'view': 0.5, 'search': 0.1},  # More decisive
            'cart_add': {'view': 0.2, 'cart_remove': 0.1, 'purchase': 0.5, 'search': 0.2},  # More likely to purchase
            'cart_remove': {'view': 0.6, 'search': 0.4},
            'search': {'view': 0.7, 'cart_add': 0.3},
            'purchase': {'view': 0.6, 'search': 0.4},
            'start': {'view': 0.8, 'search': 0.2}  # More likely to go straight to products
        }
        
        # Device preferences - different for new vs returning
        new_device_weights = [0.3, 0.6, 0.1]  # More mobile-heavy
        returning_device_weights = [0.4, 0.45, 0.15]  # More balanced
        
        for _, customer in all_customers.iterrows():
            # Determine if customer is new or returning
            is_new = customer['customer_id'] >= self.id_offset
            
            # Select appropriate event flow and device weights
            event_flow = new_customer_flow if is_new else returning_customer_flow
            device_weights = new_device_weights if is_new else returning_device_weights
            
            # Determine base number of sessions
            if is_new:
                # New customers have more variable engagement
                base_sessions = np.random.poisson(10)  # Higher initial engagement
            else:
                # Returning customers have more predictable patterns
                base_sessions = max(3, min(12, int(customer['annual_income'] / 15000)))
                
            
            num_sessions = int(base_sessions)
            
            for _ in range(num_sessions):
                session_id = f"session_{random.randint(10000, 99999)}"
                device_type = random.choices(
                    ['desktop', 'mobile', 'tablet'],
                    weights=device_weights
                )[0]
                
                # Generate session start time based on customer signup
                signup_date = pd.to_datetime(customer['signup_date'])
                if signup_date >= self.start_date:
                    # New customer - sessions start after signup
                    available_days = (self.end_date - signup_date).days
                    base_date = signup_date + timedelta(days=random.randint(0, max(0, available_days)))
                else:
                    # Returning customer - sessions within the 30-day window
                    base_date = self.start_date + timedelta(days=random.randint(0, 29))
                
                # Apply time-of-day patterns
                hour_weights = hourly_patterns[device_type]['distribution']
                hour = random.choices(range(24), weights=hour_weights)[0]
                minute = random.randint(0, 59)
                session_start = base_date.replace(hour=hour, minute=minute)
                
                # Generate sequence of events
                current_event = 'start'
                event_time = session_start
                
                # New customers tend to have longer sessions
                base_events = 4 if is_new else 3
                num_events = np.random.poisson(base_events) + 1
                
                # Track cart items for the session
                cart_items = set()
                
                for _ in range(num_events):
                    next_event_probs = event_flow.get(current_event, event_flow['start'])
                    next_event = random.choices(
                        list(next_event_probs.keys()),
                        weights=list(next_event_probs.values())
                    )[0]
                    
                    if next_event != 'start':
                        # Product selection logic
                        if cart_items and next_event in ['view', 'search']:
                            # 30% chance to view related products
                            if random.random() < 0.3:
                                related_category = products_df[
                                    products_df['product_id'].isin(cart_items)
                                ]['category_id'].iloc[0]
                                potential_products = products_df[
                                    products_df['category_id'] == related_category
                                ]
                                if not potential_products.empty:
                                    product = potential_products.sample(n=1).iloc[0]
                                else:
                                    product = products_df.sample(n=1).iloc[0]
                            else:
                                product = products_df.sample(n=1).iloc[0]
                        else:
                            product = products_df.sample(n=1).iloc[0]
                        
                        # Update cart items
                        if next_event == 'cart_add':
                            cart_items.add(product['product_id'])
                        elif next_event == 'cart_remove' and cart_items:
                            removed_item = random.choice(list(cart_items))
                            cart_items.remove(removed_item)
                            product = products_df[
                                products_df['product_id'] == removed_item
                            ].iloc[0]
                        
                        events.append({
                            'event_id': event_id,
                            'customer_id': customer['customer_id'],
                            'product_id': product['product_id'],
                            'event_type': next_event,
                            'event_date': event_time,
                            'device_type': device_type,
                            'session_id': session_id,
                            'created_at': event_time
                        })
                        event_id += 1
                    
                    current_event = next_event
                    
                    # Increment time with realistic gaps
                    if next_event in ['cart_add', 'purchase']:
                        # These actions take longer
                        event_time += timedelta(minutes=random.randint(2, 5))
                    else:
                        event_time += timedelta(minutes=random.randint(1, 3))
        
        return pd.DataFrame(events)
    
    def get_category_name(self, category_id):
        """Map category ID to category name"""
        category_names = {
            1: 'Electronics',
            2: 'Fashion',
            3: 'Home & Living',
            4: 'Beauty',
            5: 'Sports'
        }
        return category_names.get(category_id, 'General')
    
    def generate_review_text(self, rating, product_info):
        """Generate realistic review text based on rating and product info"""
        
        # Map category_id to category name
        category = self.get_category_name(product_info['category_id'])
        
        # Category-specific aspects
        category_specific_aspects = {
            'Electronics': {
                'positive': [
                    "battery life is impressive", "interface is intuitive", "setup was straightforward",
                    "performance is lightning fast", "build quality is premium", "features are well-thought-out"
                ],
                'negative': [
                    "battery drains quickly", "interface is confusing", "setup was complicated",
                    "performance is sluggish", "build quality feels cheap", "missing basic features"
                ]
            },
            'Fashion': {
                'positive': [
                    "fits perfectly", "material is high quality", "stitching is excellent",
                    "color is vibrant", "style is trendy", "comfortable to wear"
                ],
                'negative': [
                    "sizing runs small", "material feels cheap", "stitching came loose",
                    "color faded after washing", "style looks dated", "uncomfortable to wear"
                ]
            },
            'Home & Living': {
                'positive': [
                    "easy to assemble", "looks elegant", "high-quality materials",
                    "perfect size", "very durable", "great value"
                ],
                'negative': [
                    "difficult to assemble", "looks cheap", "materials feel flimsy",
                    "too big/small", "broke easily", "overpriced"
                ]
            },
            'Beauty': {
                'positive': [
                    "gentle on skin", "noticeable results", "pleasant fragrance",
                    "absorbs quickly", "long-lasting", "great texture"
                ],
                'negative': [
                    "caused irritation", "no visible results", "strong chemical smell",
                    "feels greasy", "wears off quickly", "texture is unpleasant"
                ]
            },
            'Sports': {
                'positive': [
                    "comfortable during workouts", "durable construction", "excellent grip",
                    "moisture-wicking", "lightweight", "versatile use"
                ],
                'negative': [
                    "uncomfortable during exercise", "falls apart easily", "poor grip",
                    "gets sweaty", "too heavy", "limited use"
                ]
            }
        }

        # Select appropriate template and aspects based on rating
        if rating >= 4:
            template = random.choice(self.review_templates['positive'])
            aspects = category_specific_aspects[category]['positive']
        elif rating >= 3:
            template = random.choice(self.review_templates['neutral'])
            aspects = self.positive_aspects + self.negative_aspects
        else:
            template = random.choice(self.review_templates['negative'])
            aspects = category_specific_aspects[category]['negative']

        # Generate review text
        return template.format(
            product_type=product_info['product_name'],
            positive_aspect=random.choice(aspects),
            another_positive=random.choice(aspects),
            negative_aspect=random.choice(category_specific_aspects[category]['negative']),
            neutral_comment=random.choice(self.neutral_comments)
        )

    def generate_reviews(self, orders_df, order_items_df, products_df, customers_df):
        """Generate reviews with realistic patterns and sentiment"""
        reviews = []
        review_id = 1 + self.id_offset
        
        # Review probability by customer segment
        review_prob = {
            'High Value': 0.8,
            'Mid Value': 0.6,
            'Low Value': 0.4
        }
        
        # Define rating distributions by price range
        def get_rating_distribution(price, is_new_customer):
            if price > 500:  # Expensive products
                base_dist = [0.02, 0.03, 0.10, 0.35, 0.50]  # Higher expectations
            elif price > 100:  # Mid-range products
                base_dist = [0.05, 0.10, 0.15, 0.40, 0.30]
            else:  # Budget products
                base_dist = [0.10, 0.15, 0.25, 0.30, 0.20]  # More varied ratings
                
            # New customers tend to give more extreme ratings
            if is_new_customer:
                # Increase weights of 1 and 5 star ratings
                base_dist[0] *= 1.5
                base_dist[4] *= 1.5
                # Normalize weights
                total = sum(base_dist)
                base_dist = [w/total for w in base_dist]
                
            return base_dist
        
        # Define time patterns for review submission
        def get_review_delay(is_new_customer):
            if is_new_customer:
                delays = [1, 2, 3, 4, 5, 6, 7, 14]  # New customers review more quickly
                weights = [0.35, 0.25, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]
            else:
                delays = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
                weights = [0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]
            return random.choices(delays, weights=weights)[0]
        
        # Try to load historic customers
        try:
            root_dir = os.getcwd()
            historic_customers_file = os.path.join(root_dir, "generated_historic_data", "customers.csv")
            historic_df = pd.read_csv(historic_customers_file)
            
            # Combine with new customers
            all_customers = pd.concat([customers_df, historic_df], ignore_index=True)
        except Exception as e:
            print(f"Could not load historic customers for reviews: {str(e)}")
            print("Using only new customers for reviews")
            all_customers = customers_df
        
        # Process only delivered orders
        delivered_orders = orders_df[orders_df['status'] == 'Delivered']
        
        for _, order in delivered_orders.iterrows():
            try:
                # Find customer in combined customer dataframe
                customer = all_customers[all_customers['customer_id'] == order['customer_id']].iloc[0]
                
                # Determine if customer is new or returning
                is_new = customer['customer_id'] >= self.id_offset
                
                # Get order items
                order_items = order_items_df[order_items_df['order_id'] == order['order_id']]
                
                # Determine customer segment and review probability
                segment = 'High Value' if customer['annual_income'] > 80000 else \
                        ('Mid Value' if customer['annual_income'] > 50000 else 'Low Value')
                base_review_prob = review_prob.get(segment, 0.5)
                
                # New customers more likely to review
                final_review_prob = base_review_prob * (1.3 if is_new else 1.0)
                
                if random.random() < final_review_prob:
                    for _, item in order_items.iterrows():
                        product = products_df[products_df['product_id'] == item['product_id']].iloc[0]
                        
                        # Get rating distribution based on product price and customer type
                        rating_dist = get_rating_distribution(product['base_price'], is_new)
                        review_score = random.choices(range(1, 6), weights=rating_dist)[0]
                        
                        # Generate review text with appropriate length
                        review_length = "short" if random.random() < (0.6 if is_new else 0.3) else "long"
                        
                        # Get product details for review text
                        product_name = (product['product_name'] if 'product_name' in product 
                                    else product['name'] if 'name' in product 
                                    else f"Product {product['product_id']}")
                        
                        review_text = self.generate_review_text(review_score, product_name, review_length)
                        
                        # Calculate review date with delay based on customer type
                        review_delay = get_review_delay(is_new)
                        review_date = pd.to_datetime(order['order_date']) + timedelta(days=review_delay)
                        
                        if review_date <= self.end_date:
                            # More helpful votes for extreme ratings
                            helpful_votes = np.random.poisson(2) if review_score in [1, 5] else \
                                        np.random.poisson(1) if review_score != 3 else 0
                            
                            reviews.append({
                                'review_id': review_id,
                                'product_id': product['product_id'],
                                'order_id': order['order_id'],
                                'customer_id': customer['customer_id'],
                                'review_score': review_score,
                                'review_text': review_text,
                                'review_date': review_date,
                                'helpful_votes': helpful_votes,
                                'verified_purchase': True,
                                'created_at': review_date,
                            })
                            review_id += 1
            
            except IndexError:
                print(f"Warning: Customer {order['customer_id']} not found in customer data")
                continue
            except Exception as e:
                print(f"Warning: Error processing order {order['order_id']}: {str(e)}")
                continue
        
        reviews_df = pd.DataFrame(reviews)
        
        # Update product ratings in place
        for pid in products_df['product_id'].unique():
            product_reviews = reviews_df[reviews_df['product_id'] == pid]
            if len(product_reviews) > 0:
                products_df.loc[products_df['product_id'] == pid, 'rating'] = \
                    round(product_reviews['review_score'].mean(), 1)
                products_df.loc[products_df['product_id'] == pid, 'review_count'] = \
                    len(product_reviews)
        
        # Return only the reviews DataFrame since we've updated products_df in place
        return reviews_df

    def generate_review_text(self, rating, product_name, length="short"):
        """Generate realistic review text based on rating"""
        positive_phrases = [
            "Excellent product", "Great quality", "Highly recommend",
            "Exceeded expectations", "Perfect fit", "Amazing value",
            "Outstanding performance", "Very satisfied", "Fantastic purchase"
        ]
        
        negative_phrases = [
            "Disappointed", "Poor quality", "Not worth the price",
            "Wouldn't recommend", "Didn't meet expectations", "Waste of money",
            "Stopped working", "Frustrating experience", "Save your money"
        ]
        
        neutral_phrases = [
            "Okay product", "Decent quality", "As expected",
            "Nothing special", "Gets the job done", "Average quality",
            "Fair price", "Basic functionality", "Standard product"
        ]
        
        if rating >= 4:
            base_phrases = positive_phrases
        elif rating <= 2:
            base_phrases = negative_phrases
        else:
            base_phrases = neutral_phrases
        
        # Protect against None or empty product names
        safe_product_name = str(product_name) if product_name else "This product"
        
        if length == "short":
            review = f"{random.choice(base_phrases)}. {safe_product_name} {random.choice(['is', 'was'])} "
            if rating >= 4:
                review += random.choice(["great", "excellent", "fantastic", "very good"])
            elif rating <= 2:
                review += random.choice(["disappointing", "poor", "not good", "below average"])
            else:
                review += random.choice(["okay", "decent", "average", "fair"])
        else:
            phrases = random.sample(base_phrases, 2)
            review = f"{phrases[0]}. {safe_product_name} {random.choice(['is', 'was'])} "
            if rating >= 4:
                review += random.choice(["great", "excellent", "fantastic", "very good"])
            elif rating <= 2:
                review += random.choice(["disappointing", "poor", "not good", "below average"])
            else:
                review += random.choice(["okay", "decent", "average", "fair"])
            review += f". {phrases[1]}. "
            review += random.choice([
                "Would definitely buy again.",
                "Not sure if I would buy again.",
                "Might consider other options next time.",
                "Looking forward to using it more.",
                "Hope this helps other buyers.",
                "Just wanted to share my experience."
            ])
        
        return review.strip()

    def generate_all_data(self, n_customers=200, n_products=50):
        """Generate all e-commerce data for the last 30 days"""
        print(f"Generating data for period: {self.start_date.date()} to {self.end_date.date()}")
        
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
        reviews_df = self.generate_reviews(orders_df, order_items_df, products_df, customers_df)
        
        print("Generating customer interactions...")
        interactions_df = self.generate_interactions(customers_df, products_df)
        
        # Ensure all date columns are datetime
        date_columns = {
            'customers': ['signup_date', 'last_login'],
            'products': ['created_at'],
            'orders': ['order_date', 'created_at', 'updated_at'],
            'order_items': ['created_at'],
            'reviews': ['review_date', 'created_at'],
            'interactions': ['event_date', 'created_at'],
            'categories': ['created_at'],
            'subcategories': ['created_at']
        }
        
        data_dict = {
            'customers': customers_df,
            'products': products_df,
            'categories': categories_df,
            'subcategories': subcategories_df,
            'orders': orders_df,
            'order_items': order_items_df,
            'reviews': reviews_df,
            'interactions': interactions_df
        }
        
        # self.save_data(data_dict, output_formats)
        for table_name, df in data_dict.items():
            if table_name in date_columns:
                for col in date_columns[table_name]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
        
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
    # Set master seed
    MASTER_SEED = 42
    
    # Generate data
    generator = RecentEcommerceDataGenerator(seed=MASTER_SEED)
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