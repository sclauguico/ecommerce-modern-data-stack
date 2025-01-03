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
    def __init__(self, start_date=datetime(2023, 1, 1), end_date=datetime(2024, 12, 15)):
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
        """Generate product categories and subcategories"""
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

    def generate_products(self, n_products=1000):
        """Generate product catalog"""
        categories_df, subcategories_df = self.generate_product_categories()
        
        products = []
        for product_id in range(1, n_products + 1):
            category_id = random.randint(1, len(categories_df))
            valid_subcats = subcategories_df[subcategories_df['category_id'] == category_id]
            subcategory_id = random.choice(valid_subcats['subcategory_id'].values)
            
            base_price = random.uniform(10, 1000)
            
            products.append({
                'product_id': product_id,
                'category_id': category_id,
                'subcategory_id': subcategory_id,
                'product_name': f"{self.fake.company()} {self.fake.word().title()}",
                'description': self.fake.text(max_nb_chars=200),
                'base_price': round(base_price, 2),
                'sale_price': round(base_price * random.uniform(0.8, 1.0), 2),
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
        """Generate customer data"""
        customers = []
        for customer_id in range(1, n_customers + 1):
            age = np.random.normal(45, 15)
            age = max(18, min(90, int(age)))
            signup_date = self.start_date + timedelta(days=random.randint(0, (self.end_date - self.start_date).days))
            
            customers.append({
                'customer_id': customer_id,
                'email': self.fake.email(),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'age': age,
                'gender': random.choice(['M', 'F', 'Other']),
                'annual_income': max(15000, int(np.random.normal(65000, 30000))),
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
        """Generate order data"""
        orders = []
        order_items = []
        order_id = 1
        
        for _, customer in customers_df.iterrows():
            # Number of orders influenced by customer demographics
            income_factor = min(2.0, max(0.5, customer['annual_income'] / 65000))
            num_orders = np.random.poisson(5 * income_factor)
            
            for _ in range(num_orders):
                # Ensure order_date is within the date range
                max_days = (self.end_date - customer['signup_date']).days
                if max_days <= 0:
                    order_date = customer['signup_date']
                else:
                    order_date = customer['signup_date'] + timedelta(days=random.randint(0, max_days))
                
                # Order status flow
                if order_date + timedelta(days=7) > self.end_date:
                    status = random.choice(['Pending', 'Processing', 'Shipped'])
                else:
                    status = 'Delivered'
                
                shipping_cost = round(random.uniform(5, 20), 2)
                
                # Generate order items
                num_items = np.random.poisson(2) + 1
                order_products = products_df.sample(n=min(num_items, len(products_df)), replace=True)
                
                total_amount = shipping_cost
                for _, product in order_products.iterrows():
                    quantity = random.randint(1, 3)
                    price = product['sale_price']
                    item_total = quantity * price
                    total_amount += item_total
                    
                    order_items.append({
                        'order_item_id': len(order_items) + 1,
                        'order_id': order_id,
                        'product_id': product['product_id'],
                        'quantity': quantity,
                        'unit_price': price,
                        'total_price': item_total,
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
        """Generate customer interaction events"""
        events = []
        
        for _, customer in customers_df.iterrows():
            num_interactions = np.random.poisson(20)
            
            for _ in range(num_interactions):
                # Ensure event_date is within the date range
                max_days = (self.end_date - customer['signup_date']).days
                if max_days <= 0:
                    event_date = customer['signup_date']
                else:
                    event_date = customer['signup_date'] + timedelta(days=random.randint(0, max_days))
                
                product = products_df.sample(n=1).iloc[0]
                
                event_type = random.choice(['view', 'cart_add', 'cart_remove', 'wishlist_add', 'search'])
                
                events.append({
                    'event_id': len(events) + 1,
                    'customer_id': customer['customer_id'],
                    'product_id': product['product_id'],
                    'event_type': event_type,
                    'event_date': event_date,
                    'device_type': random.choice(['desktop', 'mobile', 'tablet']),
                    'session_id': f"session_{random.randint(10000, 99999)}",
                    'created_at': event_date
                })
        
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
        """
        Generate synthetic reviews for products based on order and customer data.
        
        Parameters:
            orders_df (DataFrame): DataFrame containing order details.
            order_items_df (DataFrame): DataFrame containing order items.
            products_df (DataFrame): DataFrame containing product details.
            customers_df (DataFrame): DataFrame containing customer details.

        Returns:
            reviews_df (DataFrame): DataFrame containing generated reviews.
            updated_products_df (DataFrame): DataFrame with updated product review counts.
        """
        # Ensure 'review_count' exists in products_df
        if 'review_count' not in products_df.columns:
            products_df['review_count'] = 0
            print("Added 'review_count' column to products_df with default value 0.")

        # Ensure 'review_count' column is properly filled and cast to int
        products_df['review_count'] = products_df['review_count'].fillna(0).astype(int)

        # Logic to generate synthetic reviews
        reviews_data = []
        for _, row in orders_df.iterrows():
            order_id = row['order_id']
            customer_id = row['customer_id']
            # Get items for this order
            order_items = order_items_df[order_items_df['order_id'] == order_id]

            for _, item in order_items.iterrows():
                product_id = item['product_id']
                # Create synthetic review (e.g., random score and text)
                review_score = np.random.randint(1, 6)  # Random score between 1 and 5
                review_text = f"Review for product {product_id}"  # Placeholder text

                reviews_data.append({
                    'product_id': product_id,
                    'order_id': order_id,
                    'customer_id': customer_id,
                    'review_score': review_score,
                    'review_text': review_text
                })

                # Increment review count for the product
                products_df.loc[products_df['product_id'] == product_id, 'review_count'] += 1

        # Create DataFrame for reviews
        reviews_df = pd.DataFrame(reviews_data)

        # Return reviews and updated products DataFrame
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