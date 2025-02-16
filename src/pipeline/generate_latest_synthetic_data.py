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

    def generate_products(self, n_products=1000):
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
                'product_id': product_id,
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

    def generate_customers(self, n_customers=1000):
        root_dir = os.getcwd()  # Gets the current working directory (assumed to be the project root)
        historic_customers_file = os.path.join(root_dir, "generated_historic_data", "customers.csv")

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

        # Channel preferences by age segment
        channel_dist = {
            'Gen Z': {'Mobile App': 0.6, 'Web': 0.3, 'Email': 0.1},
            'Young Millennials': {'Mobile App': 0.5, 'Web': 0.4, 'Email': 0.1},
            'Older Millennials': {'Mobile App': 0.4, 'Web': 0.4, 'Email': 0.2},
            'Gen X': {'Mobile App': 0.3, 'Web': 0.4, 'Email': 0.3},
            'Boomers': {'Mobile App': 0.2, 'Web': 0.5, 'Email': 0.3}
        }

        # Try to load historic customers
        try:
            historic_df = pd.read_csv(historic_customers_file)
            print(f"Loaded {len(historic_df)} historic customers")
            
            # Calculate number of returning customers (70% of historic customers)
            n_returning = min(int(len(historic_df) * 0.7), int(n_customers * 0.3))
            n_new = n_customers - n_returning
            
            print(f"Generating {n_returning} returning customers and {n_new} new customers")
            
            # Select random returning customers
            returning_customers = historic_df.sample(n=n_returning).copy()
            
            # Update their activity for the recent period
            for idx in returning_customers.index:
                # Determine age segment based on age
                age = returning_customers.loc[idx, 'age']
                segment = next((seg for seg, props in age_segments.items() 
                            if props['range'][0] <= age <= props['range'][1]), 'Gen X')
                
                # Generate new login date within the last 30 days
                last_login = self.generate_random_date()
                returning_customers.loc[idx, 'last_login'] = last_login
                
                # Update fields with segment-appropriate values
                income_range = age_segments[segment]['income_range']
                returning_customers.loc[idx, 'annual_income'] = int(np.random.normal(
                    (income_range[0] + income_range[1]) / 2,
                    (income_range[1] - income_range[0]) / 4
                ))
                
                returning_customers.loc[idx, 'preferred_channel'] = random.choices(
                    list(channel_dist[segment].keys()),
                    weights=list(channel_dist[segment].values())
                )[0]
                
                # Update other fields
                returning_customers.loc[idx, 'marital_status'] = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
                returning_customers.loc[idx, 'location_type'] = random.choice(['Urban', 'Suburban', 'Rural'])
                returning_customers.loc[idx, 'is_active'] = True
            
        except FileNotFoundError:
            print("Historic customers file not found. Generating all new customers.")
            n_returning = 0
            n_new = n_customers
            returning_customers = pd.DataFrame()
        
        # Generate new customers
        new_customers = []
        start_id = self.id_offset if not len(returning_customers) else max(returning_customers['customer_id']) + 1
        
        for customer_id in range(start_id, start_id + n_new):
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
            
            # Select preferred channel based on age segment
            channel_weights = channel_dist[age_segment]
            preferred_channel = random.choices(
                list(channel_weights.keys()),
                weights=list(channel_weights.values())
            )[0]
            
            # Generate recent signup date
            signup_date = self.generate_random_date()
            last_login = min(signup_date + timedelta(days=random.randint(0, 5)), self.end_date)
            
            # Determine customer segment based on income
            if income > 80000:
                segment = 'High Value'
            elif income > 50000:
                segment = 'Mid Value'
            else:
                segment = 'Low Value'
            
            new_customers.append({
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
                'last_login': last_login,
                'preferred_channel': preferred_channel,
                'is_active': True
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
            signup_date = max(pd.to_datetime(customer['signup_date']), self.start_date)
            num_orders = np.random.poisson(2)
            
            # Order frequency based on customer segment
            base_frequency = {
                'High Value': 4,
                'Mid Value': 2,
                'Low Value': 1
            }.get(customer.get('customer_segment', 'Mid Value'), 2)
            
            # Add some randomness to frequency
            num_orders = np.random.poisson(base_frequency)
            
            for _ in range(num_orders):
                # Generate order date with patterns
                order_date = max(
                    signup_date,
                    self.generate_random_date()
                )
                
                # Apply seasonality and day of week factors
                season_factor = seasonality[order_date.month]
                day_factor = daily_factors[order_date.weekday()]
                
                # Order status based on recency
                days_since_order = (self.end_date - order_date).days
                if days_since_order < 1:
                    status = 'Pending'
                elif days_since_order < 2:
                    status = 'Processing'
                elif days_since_order < 4:
                    status = 'Shipped'
                else:
                    status = 'Delivered'
                
                # Generate order items with product affinity
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
                
                # Shipping cost with seasonal factor
                base_shipping = random.uniform(5, 20)
                shipping_cost = round(base_shipping * season_factor, 2)
                total_amount = shipping_cost
                
                # Generate order items
                for _, product in order_products.iterrows():
                    # Quantity based on customer segment and price
                    if customer.get('customer_segment') == 'High Value':
                        quantity = np.random.poisson(2) + 1
                    else:
                        quantity = np.random.poisson(1) + 1
                    
                    # Price with seasonal and daily factors
                    price = product['sale_price'] * season_factor * day_factor
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
                
                # Payment method based on customer segment
                payment_methods = {
                    'High Value': ['Credit Card', 'PayPal'],
                    'Mid Value': ['Credit Card', 'PayPal', 'Debit Card'],
                    'Low Value': ['Debit Card', 'PayPal']
                }
                segment = customer.get('customer_segment', 'Mid Value')
                payment_method = random.choice(payment_methods[segment])
                
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
        """Generate customer interactions for the last 30 days"""
        events = []
        event_id = 1 + self.id_offset
        
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
        
        # Event flow with realistic browsing patterns
        event_flow = {
            'view': {'cart_add': 0.3, 'view': 0.6, 'search': 0.1},
            'cart_add': {'view': 0.3, 'cart_remove': 0.2, 'purchase': 0.2, 'search': 0.3},
            'cart_remove': {'view': 0.7, 'search': 0.3},
            'search': {'view': 0.8, 'cart_add': 0.2},
            'purchase': {'view': 0.6, 'search': 0.4},
            'start': {'view': 0.7, 'search': 0.3}
        }
        
        for _, customer in customers_df.iterrows():
            # Base number of sessions - using income as a rough proxy for engagement
            base_sessions = max(5, min(15, int(customer['annual_income'] / 10000)))
            num_sessions = np.random.poisson(base_sessions)
            
            for _ in range(num_sessions):
                session_id = f"session_{random.randint(10000, 99999)}"
                device_type = random.choices(
                    ['desktop', 'mobile', 'tablet'], 
                    weights=[0.4, 0.45, 0.15]
                )[0]
                
                # Generate session start time
                # Convert signup_date to datetime if it isn't already
                signup_date = pd.to_datetime(customer['signup_date'])
                available_days = (self.end_date - signup_date).days
                
                if available_days <= 0:
                    # If signup_date is already at or after end_date, use signup_date
                    base_date = signup_date
                else:
                    # Otherwise, generate a random date between signup and end_date
                    days_offset = random.randint(0, available_days)
                    base_date = signup_date + timedelta(days=days_offset)
                
                hour_weights = hourly_patterns[device_type]['distribution']
                hour = random.choices(range(24), weights=hour_weights)[0]
                minute = random.randint(0, 59)
                session_start = base_date.replace(hour=hour, minute=minute)
                
                # Generate sequence of events
                current_event = 'start'
                event_time = session_start
                num_events = np.random.poisson(4) + 1  # At least one event per session
                
                for _ in range(num_events):
                    if current_event in event_flow:
                        next_event_probs = event_flow[current_event]
                        next_event = random.choices(
                            list(next_event_probs.keys()),
                            weights=list(next_event_probs.values())
                        )[0]
                    else:
                        next_event = 'view'
                    
                    if next_event != 'start':
                        # Select product based on price range affinity
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
                    
                    # Choose next event based on flow probabilities
                    if current_event in event_flow:
                        next_event_probs = event_flow[current_event]
                        current_event = random.choices(
                            list(next_event_probs.keys()),
                            weights=list(next_event_probs.values())
                        )[0]
                    else:
                        current_event = 'view'
                    
                    # Increment time realistically
                    event_time += timedelta(minutes=random.randint(1, 10))
        
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
    
    def generate_review_text(self, rating, product_info, days_to_review):
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

        # Time-based context phrases
        time_context = {
            'quick': [
                "Just got this", "After quick testing", "First impression", 
                "Day one review", "Initial thoughts"
            ],
            'short': [
                "After a few days", "Short term use", "Early review", 
                "Quick update", "First week thoughts"
            ]
        }

        # Get category and time phrase
        # Add time-based context phrases for recent reviews
        time_phrases = {
            'recent': [
                "Just received", "Got this yesterday", "Ordered last week",
                "Fresh out of the box", "First impression", "Early review"
            ]
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
        reviews_data = []
        review_id = 1 + self.id_offset
        
        # Review probability by customer segment
        review_prob = {
            'High Value': 0.8,
            'Mid Value': 0.6,
            'Low Value': 0.4
        }
        
        # Define rating distributions by price range
        def get_rating_distribution(price):
            if price > 500:  # Expensive products
                return [0.02, 0.03, 0.10, 0.35, 0.50]  # Higher expectations
            elif price > 100:  # Mid-range products
                return [0.05, 0.10, 0.15, 0.40, 0.30]
            else:  # Budget products
                return [0.10, 0.15, 0.25, 0.30, 0.20]  # More varied ratings
        
        # Define time patterns for review submission
        def get_review_delay():
            # Most reviews come within first week for recent data
            delays = [1, 2, 3, 4, 5, 6, 7]
            weights = [0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]
            return random.choices(delays, weights=weights)[0]
        
        for _, order in orders_df.iterrows():
            if order['status'] != 'Delivered':
                continue
                
            customer = customers_df[customers_df['customer_id'] == order['customer_id']].iloc[0]
            order_items = order_items_df[order_items_df['order_id'] == order['order_id']]
            
            # Determine customer segment and review probability
            segment = 'High Value' if customer['annual_income'] > 80000 else ('Mid Value' if customer['annual_income'] > 50000 else 'Low Value')
            review_probability = review_prob.get(segment, 0.5)
            
            if random.random() < review_probability:
                for _, item in order_items.iterrows():
                    product = products_df[products_df['product_id'] == item['product_id']].iloc[0]
                    
                    # Get rating distribution based on product price
                    rating_dist = get_rating_distribution(product['base_price'])
                    review_score = random.choices(range(1, 6), weights=rating_dist)[0]
                    
                    # Calculate review delay and date
                    review_delay = get_review_delay()
                    review_date = pd.to_datetime(order['order_date']) + timedelta(days=review_delay)
                    
                    # Generate review text with delay information
                    review_text = self.generate_review_text(
                        review_score,
                        product,
                        review_delay  # Pass the review_delay as days_to_review
                    )
                    
                    if review_date <= self.end_date:
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
        reviews_df, updated_products_df = self.generate_reviews(orders_df, order_items_df, products_df, customers_df)
        
        print("Generating customer interactions...")
        interactions_df = self.generate_interactions(customers_df, updated_products_df)
        
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
            'products': updated_products_df,
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