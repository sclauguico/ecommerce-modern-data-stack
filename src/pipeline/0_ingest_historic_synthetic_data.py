import os
import pandas as pd
import boto3
import json
from sqlalchemy import create_engine
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import psycopg2
import tempfile
from pandas import json_normalize
from datetime import datetime
import uuid
import io
from pathlib import Path

load_dotenv()

class InitialHistoricETL:
    def __init__(self):
        # Initialize connections
        self._init_postgres()
        self._init_s3()
        self._init_snowflake()
        
        # Define source mappings
        self.postgres_tables = ['categories', 'subcategories', 'order_items', 'interactions']
        self.s3_tables = ['customers', 'products', 'orders', 'reviews']
        
        # Track batch information
        self.batch_id = str(uuid.uuid4())
        self.batch_timestamp = datetime.now()
    
    def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        self.pg_conn_string = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        self.pg_engine = create_engine(self.pg_conn_string)

    def _init_s3(self):
        """Initialize S3 clients"""
        s3_credentials = {
            'aws_access_key_id': os.getenv('AWS_S3_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_S3_SECRET_ACCESS_KEY')
        }
        
        self.s3_client = boto3.client('s3', **s3_credentials)
        self.historic_bucket = os.getenv('AWS_S3_HISTORIC_SYNTH')
        self.latest_bucket = os.getenv('AWS_S3_LATEST_SYNTH')

    def _init_snowflake(self):
        """Initialize Snowflake connection"""
        try:
            self.snow_conn = snowflake.connector.connect(
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
                role=os.getenv('SNOWFLAKE_ROLE')
            )
            
            self.snow_cursor = self.snow_conn.cursor()
            
            # Setup database and schema
            for cmd in [
                f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}",
                f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}",
                f"USE SCHEMA {os.getenv('SNOWFLAKE_RAW_SCHEMA')}"
            ]:
                self.snow_cursor.execute(cmd)
                
        except Exception as e:
            print(f"Snowflake initialization error: {str(e)}")
            raise

    def extract_from_postgres(self, table_name, data_source):
        """Extract data from PostgreSQL"""
        try:
            table_prefix = 'latest_' if data_source == 'latest' else ''
            query = f"SELECT * FROM {table_prefix}{table_name}"
            df = pd.read_sql(query, self.pg_engine)
            return df
        except Exception as e:
            print(f"PostgreSQL extraction error for {table_name}: {str(e)}")
            raise

    def extract_from_s3(self, table_name, data_source):
        """Extract data from S3"""
        try:
            bucket = self.latest_bucket if data_source == 'latest' else self.historic_bucket
            response = self.s3_client.get_object(
                Bucket=bucket,
                Key=f'json/{table_name}.json'
            )
            
            json_content = json.loads(response['Body'].read().decode('utf-8'))
            return pd.DataFrame(json_content['data'])
        except Exception as e:
            print(f"S3 extraction error for {table_name}: {str(e)}")
            raise
        
    def get_primary_keys(self, table_name):
        """Return primary key columns for each table"""
        pk_mapping = {
            'customers': ['CUSTOMER_ID'],
            'orders': ['ORDER_ID'],
            'products': ['PRODUCT_ID'],
            'order_items': ['ORDER_ITEM_ID'],
            'categories': ['CATEGORY_ID'],
            'subcategories': ['SUBCATEGORY_ID'],
            'reviews': ['REVIEW_ID'],
            'interactions': ['EVENT_ID']
        }
        return pk_mapping.get(table_name.replace('latest_', ''), ['id'])

    def remove_duplicate_primary_keys(self, df, table_name):
        """Remove rows with duplicate primary keys, defaulting to the first column if necessary."""
        try:
            # Get primary keys for the table
            primary_keys = self.get_primary_keys(table_name)

            if not primary_keys:
                print(f"Warning: No primary keys found or defined for {table_name}. Skipping deduplication.")
                return df

            # Ensure all primary keys exist in the DataFrame
            missing_keys = [key for key in primary_keys if key not in df.columns]
            if missing_keys:
                print(f"Error: Missing primary keys {missing_keys} in table {table_name}.")
                return df

            # Deduplicate based on primary keys
            if "LOADED_AT" in df.columns:
                df = df.sort_values(by="LOADED_AT", ascending=False)
            else:
                print(f"Warning: No valid date column found for {table_name}. Proceeding without sorting.")
            df = df.drop_duplicates(subset=primary_keys, keep="first")
            return df
        except Exception as e:
            print(f"Error removing duplicate primary keys for {table_name}: {str(e)}")
            raise

    def transform_data(self, df, table_name, data_source):
        """Transform data with added metadata columns"""
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Get primary key before transformation
            primary_keys = self.get_primary_keys(table_name)
            primary_keys = [pk.lower() for pk in primary_keys]
            if primary_keys:
                original_ids = set(df[primary_keys[0]].unique())
            
            # Flatten JSON if needed
            df = self.flatten_json_df(df, table_name)
            
            # Add metadata columns
            df['DATA_SOURCE'] = data_source
            df['BATCH_ID'] = self.batch_id
            df['LOADED_AT'] = self.batch_timestamp
            
            # Standard transformations
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df = df.replace({pd.NA: None, pd.NaT: None})
            
            # Convert column names to uppercase for Snowflake
            df.columns = [col.upper() for col in df.columns]
            
            # Verify no IDs were lost during transformation
            if primary_keys:
                transformed_ids = set(df[primary_keys[0].upper()].unique())
                if len(original_ids) != len(transformed_ids):
                    lost_ids = original_ids - set(int(id) for id in transformed_ids)
                    print(f"Warning: Lost {len(lost_ids)} IDs during transformation")
                    print(f"Sample of lost IDs: {sorted(lost_ids)[:5]}")
            
            return df
            
        except Exception as e:
            print(f"Transform error for {table_name}: {str(e)}")
            raise
    
    def flatten_json_df(self, df, table_name):
        """Flatten nested JSON structures"""
        try:
            json_columns = [
                col for col in df.columns 
                if df[col].dtype == 'object' and 
                isinstance(df[col].dropna().iloc[0] if not df[col].isna().all() else None, (dict, list))
            ]
            
            if not json_columns:
                return df

            flat_df = df.copy()
            
            for col in json_columns:
                try:
                    if isinstance(df[col].dropna().iloc[0], dict):
                        flattened = pd.json_normalize(df[col].dropna(), sep='_')
                        flat_df = flat_df.drop(columns=[col])
                        for new_col in flattened.columns:
                            flat_df[f"{col}_{new_col}"] = flattened[new_col]
                    elif isinstance(df[col].dropna().iloc[0], list):
                        flat_df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
                except Exception as e:
                    print(f"Warning: Could not flatten column {col}: {str(e)}")
                    continue
            
            return flat_df
        except Exception as e:
            print(f"JSON flattening error for {table_name}: {str(e)}")
            raise

    def save_to_s3_historic(self, df, table_name, metadata):
        """Save transformed data and metadata to S3 historic bucket"""
        try:
            
            output_dir = Path('ingested_data')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            # Save to historic data folder
            self.s3_client.put_object(
                Bucket=self.historic_bucket,
                Key=f'csv/{table_name}.csv',
                Body=csv_buffer.getvalue().encode('utf-8')
            )
            
            # Save metadata
            self.s3_client.put_object(
                Bucket=self.historic_bucket,
                Key=f'csv/{table_name}_metadata.json',
                Body=json.dumps(metadata, default=str)
            )
            
            print(f"Successfully saved {table_name} to historic bucket as CSV with metadata")
            
        except Exception as e:
            print(f"Error saving to S3 historic bucket: {str(e)}")
            raise

    def load_to_snowflake(self, df, table_name):
        """Load data to Snowflake"""
        try:
            database = os.getenv('SNOWFLAKE_DATABASE')
            schema = os.getenv('SNOWFLAKE_RAW_SCHEMA')
            full_table_name = f"{database}.{schema}.{table_name.upper()}"
            
            # Create table if not exists
            column_definitions = []
            for col_name, dtype in df.dtypes.items():
                sf_type = "VARCHAR"
                if pd.api.types.is_integer_dtype(dtype):
                    sf_type = "NUMBER"
                elif pd.api.types.is_float_dtype(dtype):
                    sf_type = "FLOAT"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sf_type = "TIMESTAMP"
                column_definitions.append(f'"{col_name.upper()}" {sf_type}')

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} (
                {', '.join(column_definitions)}
            )
            """
            
            self.snow_cursor.execute(create_table_sql)
            
            # Write data
            success, num_chunks, num_rows, _ = write_pandas(
                conn=self.snow_conn,
                df=df,
                table_name=table_name.upper(),
                database=database,
                schema=schema,
                quote_identifiers=False
            )

            if not success:
                raise Exception(f"Failed to load {table_name} to Snowflake")

            print(f"Successfully loaded {num_rows} rows to {full_table_name}")

        except Exception as e:
            print(f"Error loading to Snowflake: {str(e)}")
            raise

    def find_valid_date_column(self, df, table_name):
        """Find the first valid date column from the possible options"""
        possible_columns = self.get_date_column(table_name)
        
        for col in possible_columns:
            if col in df.columns:
                return col
                
        return None

    def get_date_column(self, table_name):
        """Return the appropriate date column for each table with fallbacks"""
        date_columns = {
            'customers': ['signup_date', 'last_login', 'created_at'],
            'orders': ['order_date', 'created_at', 'updated_at'],
            'products': ['created_at'],
            'order_items': ['created_at', 'order_date'],
            'reviews': ['created_at', 'order_date'],
            'interactions': ['event_date', 'created_at'],
            'categories': ['created_at'],
            'subcategories': ['created_at']
        }
        
        clean_table_name = table_name.replace('latest_', '')
        return date_columns.get(clean_table_name, ['created_at'])
    
    def get_primary_date_column(self, table_name):
        """Return the primary date column for each table"""
        primary_date_columns = {
            'customers': 'signup_date',      # When customer joined
            'orders': 'order_date',          # When order was placed
            'products': 'created_at',        # When product was added
            'order_items': 'created_at',     # When order item was created
            'reviews': 'created_at',         # When review was submitted
            'interactions': 'event_date',    # When interaction occurred
            'categories': 'created_at',      # When category was created
            'subcategories': 'created_at'    # When subcategory was created
        }
        
        clean_table_name = table_name.replace('latest_', '')
        return primary_date_columns.get(clean_table_name)

    def run_initial_load(self):
        """Execute ETL process for initial historic data setup"""
        try:
            print("\nStarting initial historic data load...")
            
            # Create local directory for CSV files if it doesn't exist
            os.makedirs("ingested_data", exist_ok=True)
            
            for table in self.postgres_tables + self.s3_tables:
                print(f"\nProcessing {table}")
                
                try:
                    # Extract data from both sources
                    historic_df = (self.extract_from_s3(table, 'historic') 
                                if table in self.s3_tables 
                                else self.extract_from_postgres(table, 'historic'))
                    
                    latest_df = (self.extract_from_s3(table, 'latest') 
                            if table in self.s3_tables 
                            else self.extract_from_postgres(table, 'latest'))
                    
                    # Get primary keys
                    primary_keys = self.get_primary_keys(table)
                    primary_keys = [pk.lower() for pk in primary_keys]
                    
                    # Print ID ranges before processing
                    if primary_keys:
                        primary_key = primary_keys[0]
                        if primary_key in historic_df.columns and primary_key in latest_df.columns:
                            print(f"\nBefore processing:")
                            print(f"Historic {primary_key} range: {historic_df[primary_key].min()} to {historic_df[primary_key].max()}")
                            print(f"Latest {primary_key} range: {latest_df[primary_key].min()} to {latest_df[primary_key].max()}")
                    
                    # Get primary date column for this table
                    date_column = self.get_primary_date_column(table)
                    
                    if date_column and date_column in historic_df.columns and date_column in latest_df.columns:
                        try:
                            # Convert date columns to datetime
                            historic_df[date_column] = pd.to_datetime(historic_df[date_column])
                            latest_df[date_column] = pd.to_datetime(latest_df[date_column])
                            
                            min_latest_date = latest_df[date_column].min()
                            max_historic_date = historic_df[date_column].max()
                            
                            print(f"\nDate ranges for {date_column}:")
                            print(f"Historic: up to {max_historic_date}")
                            print(f"Latest: starting from {min_latest_date}")
                            
                            if max_historic_date >= min_latest_date:
                                print(f"Trimming historic data before {min_latest_date}")
                                historic_df = historic_df[historic_df[date_column] < min_latest_date]
                        except Exception as e:
                            print(f"Warning: Error processing dates for {table}: {str(e)}")
                            date_column = None
                    else:
                        print(f"No primary date column found for {table}")
                        date_column = None
                    
                    # Transform data after date handling
                    historic_df = self.transform_data(historic_df, table, 'historic')
                    latest_df = self.transform_data(latest_df, table, 'latest')
                    
                    # Combine data
                    combined_df = pd.concat([historic_df, latest_df], ignore_index=True)
                    
                    # Sort by ID and primary date column
                    sort_columns = []
                    if primary_keys:
                        sort_columns.extend([col.upper() for col in primary_keys])
                    if date_column:
                        sort_columns.append(date_column.upper())
                    
                    if sort_columns:
                        combined_df = combined_df.sort_values(by=sort_columns)
                    
                    # Print final ID range and check for gaps
                    if primary_keys:
                        primary_key = primary_keys[0].upper()
                        if primary_key in combined_df.columns:
                            print(f"\nFinal {primary_key} range: {combined_df[primary_key].min()} to {combined_df[primary_key].max()}")
                            
                            # Count any gaps in IDs
                            all_ids = set(combined_df[primary_key].unique())
                            expected_range = set(range(int(min(all_ids)), int(max(all_ids)) + 1))
                            missing_ids = sorted(expected_range - all_ids)
                            if missing_ids:
                                print(f"Found {len(missing_ids)} gaps in {primary_key}")
                                print(f"First few missing IDs: {missing_ids[:5]}...")
                    
                    # Save to local CSV
                    combined_path = f"ingested_data/{table}_combined.csv"
                    combined_df.to_csv(combined_path, index=False)
                    print(f"Saved combined data to {combined_path}")
                    
                    # Prepare metadata
                    metadata = {
                        'table_name': table,
                        'batch_id': self.batch_id,
                        'timestamp': self.batch_timestamp,
                        'historic_records': len(historic_df),
                        'latest_records': len(latest_df),
                        'total_records': len(combined_df),
                        'primary_date_column': date_column,
                        'min_latest_date': min_latest_date.isoformat() if date_column and min_latest_date else None,
                        'columns': combined_df.columns.tolist(),
                        'data_types': combined_df.dtypes.astype(str).to_dict()
                    }
                    # Save to S3 as historic data
                    self.save_to_s3_historic(combined_df, table, metadata)
                    
                    # Load to Snowflake
                    # self.load_to_snowflake(combined_df, table)
                    
                    print(f"""
                    Completed processing for {table}:
                    - Historic records: {len(historic_df)}
                    - Latest records: {len(latest_df)}
                    - Total records processed: {len(combined_df)}
                    - Date column used: {date_column if date_column else 'None'}
                    - Data saved locally to: {combined_path}
                    - Data saved to S3 historic bucket
                    """)

                except Exception as e:
                    print(f"Error processing table {table}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Initial load process error: {str(e)}")
            raise
        finally:
            self.snow_cursor.close()
            self.snow_conn.close()
            print("\nInitial historic data load completed. Connections closed.")

if __name__ == "__main__":
    try:
        etl = InitialHistoricETL()
        etl.run_initial_load()
    except Exception as e:
        print(f"Main execution error: {str(e)}")
        raise