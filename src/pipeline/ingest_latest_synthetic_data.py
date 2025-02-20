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

load_dotenv()

class IncrementalETL:
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

    def extract_historic_from_s3(self, table_name):
        """Extract historic data from S3 CSV files"""
        try:
            # Get CSV file from historic bucket
            response = self.s3_client.get_object(
                Bucket=self.historic_bucket,
                Key=f'csv/{table_name}.csv'
            )
            
            # Read CSV content
            csv_content = response['Body'].read().decode('utf-8')
            return pd.read_csv(io.StringIO(csv_content))
            
        except Exception as e:
            print(f"Historic S3 extraction error for {table_name}: {str(e)}")
            raise

    def extract_latest_from_postgres(self, table_name):
        """Extract latest data from PostgreSQL"""
        try:
            query = f"SELECT * FROM latest_{table_name}"
            df = pd.read_sql(query, self.pg_engine)
            return df
        except Exception as e:
            print(f"PostgreSQL extraction error for {table_name}: {str(e)}")
            raise

    def extract_latest_from_s3(self, table_name):
        """Extract latest data from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.latest_bucket,
                Key=f'json/{table_name}.json'
            )
            
            json_content = json.loads(response['Body'].read().decode('utf-8'))
            return pd.DataFrame(json_content['data'])
        except Exception as e:
            print(f"Latest S3 extraction error for {table_name}: {str(e)}")
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
        """Transform data with enhanced column handling and debugging"""
        try:
            print(f"\nDETAILED COLUMN ANALYSIS FOR {table_name}")
            print("=" * 50)
            
            if df.empty:
                print(f"No data to transform for {table_name}")
                return df
            
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Convert column names to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            # Print initial column info
            print("Initial columns and their types:")
            for col in df.columns:
                print(f"- {col}: {df[col].dtype}")
            
            # Get primary key before transformation
            primary_keys = self.get_primary_keys(table_name)
            primary_keys = [pk.lower() for pk in primary_keys]
            
            # Check for missing primary keys
            missing_keys = [pk for pk in primary_keys if pk not in df.columns]
            if missing_keys:
                print(f"Warning: Missing primary keys {missing_keys} in {table_name}")
                # Add missing primary key columns with auto-incrementing values
                start_id = 1
                for pk in missing_keys:
                    df[pk] = range(start_id, start_id + len(df))
                    start_id += len(df)
                    print(f"Added auto-incrementing {pk} column")
            
            if primary_keys:
                original_ids = set(df[primary_keys[0]].unique())
                print(f"Found {len(original_ids)} unique IDs in {primary_keys[0]}")
            
            # Flatten JSON if needed
            df = self.flatten_json_df(df, table_name)
            
            # Add metadata columns
            df['data_source'] = data_source
            df['batch_id'] = self.batch_id
            df['loaded_at'] = self.batch_timestamp
            
            # Standard transformations
            date_columns = [col for col in df.columns if 'date' in col or col.endswith('_at')]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {str(e)}")
            
            df = df.replace({pd.NA: None, pd.NaT: None})
            
            # Convert column names to uppercase for Snowflake
            df.columns = [col.upper() for col in df.columns]
            
            # Verify no IDs were lost during transformation
            if primary_keys:
                transformed_ids = set(df[primary_keys[0].upper()].unique())
                if len(original_ids) != len(transformed_ids):
                    lost_ids = original_ids - set(int(id) for id in transformed_ids if str(id).isdigit())
                    print(f"Warning: Lost {len(lost_ids)} IDs during transformation")
                    print(f"Sample of lost IDs: {sorted(lost_ids)[:5]}")
            
            print(f"Successfully transformed {len(df)} records")
            print(f"Final columns: {df.columns.tolist()}")
            return df
                
        except Exception as e:
            print(f"Transform error for {table_name}: {str(e)}")
            print(f"DataFrame info:")
            print(df.info())
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
        """Incrementally load data to Snowflake handling duplicates properly"""
        try:
            database = os.getenv('SNOWFLAKE_DATABASE')
            schema = os.getenv('SNOWFLAKE_RAW_SCHEMA')
            full_table_name = f"{database}.{schema}.{table_name.upper()}"
            temp_table_name = f"{database}.{schema}.TEMP_{table_name.upper()}"
            
            # Get primary keys
            primary_keys = [pk.upper() for pk in self.get_primary_keys(table_name)]

            # Create temporary table for staging the new data
            column_definitions = []
            for col_name, dtype in df.dtypes.items():
                sf_type = "VARCHAR"
                if pd.api.types.is_integer_dtype(dtype):
                    sf_type = "NUMBER"
                elif pd.api.types.is_float_dtype(dtype):
                    sf_type = "FLOAT"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sf_type = "TIMESTAMP"
                column_definitions.append(f'"{col_name}" {sf_type}')

            # Drop temp table if exists
            self.snow_cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

            # Create temp table
            create_temp_table_sql = f"""
            CREATE TEMPORARY TABLE {temp_table_name} (
                {', '.join(column_definitions)}
            )
            """
            self.snow_cursor.execute(create_temp_table_sql)

            # Load data into temp table
            success, num_chunks, num_rows, _ = write_pandas(
                conn=self.snow_conn,
                df=df,
                table_name=f"TEMP_{table_name.upper()}",
                database=database,
                schema=schema,
                quote_identifiers=False
            )

            if not success:
                raise Exception(f"Failed to load data to temporary table for {table_name}")

            # Create target table if it doesn't exist
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} (
                {', '.join(column_definitions)}
            )
            """
            self.snow_cursor.execute(create_table_sql)

            # Get all columns except metadata for comparison
            compare_columns = [col for col in df.columns if col.upper() not in ['DATA_SOURCE', 'BATCH_ID', 'LOADED_AT']]
            
            # Build comparison conditions for each non-metadata column
            comparison_conditions = []
            for col in compare_columns:
                condition = f"""
                (
                    target."{col}" != source."{col}"
                    OR (target."{col}" IS NULL AND source."{col}" IS NOT NULL)
                    OR (target."{col}" IS NOT NULL AND source."{col}" IS NULL)
                )
                """
                comparison_conditions.append(condition)

            # Join all comparison conditions with OR
            comparison_clause = " OR ".join(comparison_conditions)

            # Construct MERGE statement
            merge_sql = f"""
            MERGE INTO {full_table_name} target
            USING (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY {', '.join([f'"{pk}"' for pk in primary_keys])}
                        ORDER BY "LOADED_AT" DESC
                    ) as rn
                FROM {temp_table_name}
            ) source
            ON {' AND '.join([f'target."{pk}" = source."{pk}"' for pk in primary_keys])}
            AND source.rn = 1
            WHEN MATCHED AND ({comparison_clause})
            THEN UPDATE SET
                {', '.join([f'"{col}" = source."{col}"' for col in df.columns])}
            WHEN NOT MATCHED AND source.rn = 1
            THEN INSERT (
                {', '.join([f'"{col}"' for col in df.columns])}
            )
            VALUES (
                {', '.join([f'source."{col}"' for col in df.columns])}
            )
            """

            # Execute merge
            self.snow_cursor.execute(merge_sql)

            # Clean up temp table
            self.snow_cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

            print(f"""Successfully loaded data to {full_table_name}:
            - Rows processed: {num_rows}
            - Using primary keys: {', '.join(primary_keys)}""")

        except Exception as e:
            print(f"Error loading to Snowflake: {str(e)}")
            raise
        
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

    def find_valid_date_column(self, df, table_name):
        """Find the first valid date column from the possible options"""
        possible_columns = self.get_date_column(table_name)
        
        for col in possible_columns:
            if col in df.columns:
                return col
                
        return None
    
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
            
    def run_etl(self):
        """Execute ETL process with historic CSV data"""
        try:
            print("\nStarting ETL process...")
            
            # Create local directory for CSV files if it doesn't exist
            os.makedirs("ingested_data", exist_ok=True)
            
            for table in self.postgres_tables + self.s3_tables:
                print(f"\nProcessing {table}")
                
                try:
                    # For categories and subcategories, just use latest data
                    if table in ['categories', 'subcategories']:
                        if table in self.postgres_tables:
                            latest_df = self.extract_latest_from_postgres(table)
                        else:
                            latest_df = self.extract_latest_from_s3(table)
                            
                        if not latest_df.empty:
                            # Transform data
                            transformed_df = self.transform_data(latest_df, table, 'latest')
                            
                            # Save to local CSV
                            combined_path = f"ingested_data/{table}_combined.csv"
                            transformed_df.to_csv(combined_path, index=False)
                            print(f"Saved data to {combined_path}")
                            
                            # Prepare metadata
                            metadata = {
                                'table_name': table,
                                'batch_id': self.batch_id,
                                'timestamp': str(self.batch_timestamp),
                                'records': len(transformed_df),
                                'columns': transformed_df.columns.tolist(),
                                'data_types': {col: str(dtype) for col, dtype in transformed_df.dtypes.items()}
                            }
                            
                            # Save to S3 and Snowflake
                            self.save_to_s3_historic(transformed_df, table, metadata)
                            self.load_to_snowflake(transformed_df, table)
                        continue

                    # For other tables, process both historic and latest data
                    # Extract historic data from S3 CSV
                    historic_df = self.extract_historic_from_s3(table)
                    
                    # Extract latest data from original sources
                    latest_df = (self.extract_latest_from_s3(table) 
                            if table in self.s3_tables 
                            else self.extract_latest_from_postgres(table))
                    
                    # Skip if both dataframes are empty
                    if historic_df.empty and latest_df.empty:
                        print(f"No data found for {table}, skipping...")
                        continue
                    
                    # Get primary keys
                    primary_keys = self.get_primary_keys(table)
                    primary_keys = [pk.lower() for pk in primary_keys]
                    
                    # Print ID ranges before processing
                    if primary_keys and not historic_df.empty and not latest_df.empty:
                        primary_key = primary_keys[0]
                        if primary_key in historic_df.columns and primary_key in latest_df.columns:
                            print(f"\nBefore processing:")
                            print(f"Historic {primary_key} range: {historic_df[primary_key].min()} to {historic_df[primary_key].max()}")
                            print(f"Latest {primary_key} range: {latest_df[primary_key].min()} to {latest_df[primary_key].max()}")
                    
                    # Get primary date column for this table
                    date_column = self.get_primary_date_column(table)
                    min_latest_date = None
                    
                    if date_column:
                        if not latest_df.empty and date_column in latest_df.columns:
                            latest_df[date_column] = pd.to_datetime(latest_df[date_column], errors='coerce')
                            min_latest_date = latest_df[date_column].min()
                        
                        if not historic_df.empty and date_column in historic_df.columns:
                            historic_df[date_column] = pd.to_datetime(historic_df[date_column], errors='coerce')
                            if min_latest_date is not None:
                                historic_df = historic_df[historic_df[date_column] < min_latest_date]
                    
                    # Transform data
                    historic_transformed_df = self.transform_data(historic_df, table, 'historic') if not historic_df.empty else pd.DataFrame()
                    latest_transformed_df = self.transform_data(latest_df, table, 'latest') if not latest_df.empty else pd.DataFrame()
                    
                    # Combine data
                    combined_df = pd.concat([historic_transformed_df, latest_transformed_df], ignore_index=True)
                    
                    # Skip if no data after combination
                    if combined_df.empty:
                        print(f"No data after combining for {table}, skipping...")
                        continue
                    
                    # Sort by ID and primary date column
                    sort_columns = []
                    if primary_keys:
                        sort_columns.extend([col.upper() for col in primary_keys])
                    if date_column:
                        sort_columns.append(date_column.upper())
                    
                    if sort_columns:
                        combined_df = combined_df.sort_values(by=sort_columns)
                    
                    # Remove duplicates based on primary keys
                    before_dedup = len(combined_df)
                    combined_df = self.remove_duplicate_primary_keys(combined_df, table)
                    after_dedup = len(combined_df)
                    duplicates_removed = before_dedup - after_dedup
                    
                    # Print final ID range and check for gaps
                    if primary_keys:
                        primary_key = primary_keys[0].upper()
                        if primary_key in combined_df.columns:
                            min_id = combined_df[primary_key].min()
                            max_id = combined_df[primary_key].max()
                            print(f"\nFinal {primary_key} range: {min_id} to {max_id}")
                            
                            # Count any gaps in IDs
                            all_ids = set(combined_df[primary_key].unique())
                            expected_range = set(range(int(min_id), int(max_id) + 1))
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
                        'timestamp': str(self.batch_timestamp),
                        'historic_records': len(historic_transformed_df),
                        'latest_records': len(latest_transformed_df),
                        'total_records': len(combined_df),
                        'duplicates_removed': duplicates_removed,
                        'date_column_used': date_column,
                        'min_latest_date': str(min_latest_date) if min_latest_date is not None else None,
                        'primary_keys_used': primary_keys,
                        'columns': combined_df.columns.tolist(),
                        'data_types': {col: str(dtype) for col, dtype in combined_df.dtypes.items()}
                    }
                    
                    # Save to S3 and Snowflake
                    self.save_to_s3_historic(combined_df, table, metadata)
                    self.load_to_snowflake(combined_df, table)
                    
                    print(f"""
                    Completed processing for {table}:
                    - Historic records: {len(historic_transformed_df)}
                    - Latest records: {len(latest_transformed_df)}
                    - Duplicates removed: {duplicates_removed}
                    - Total records processed: {len(combined_df)}
                    - Date column used: {date_column if date_column else 'None'}
                    - Primary keys used: {', '.join(primary_keys)}
                    - Data saved locally to: {combined_path}
                    - Data saved to S3 historic bucket
                    - Data loaded to Snowflake table: {table.upper()}
                    """)

                except Exception as e:
                    print(f"Error processing table {table}: {str(e)}")
                    continue

        except Exception as e:
            print(f"ETL process error: {str(e)}")
            raise
        finally:
            self.snow_cursor.close()
            self.snow_conn.close()
            print("\nETL process completed. Connections closed.")
        
if __name__ == "__main__":
    try:
        etl = IncrementalETL()
        etl.run_etl()
    except Exception as e:
        print(f"Main execution error: {str(e)}")
        raise