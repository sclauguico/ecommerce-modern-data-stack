from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

from src.pipeline.generate_latest_synthetic_data import RecentEcommerceDataGenerator, save_data
from src.pipeline.ingest_latest_synthetic_data import IncrementalETL

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'generate_append_latest_pipeline',
    default_args=default_args,
    description='E-commerce data generation and ETL pipeline',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    catchup=False
)

def generate_latest_data(**context):
    """Generate latest synthetic e-commerce data"""
    try:
        generator = RecentEcommerceDataGenerator()
        data = generator.generate_all_data()
        
        # Save data directly to S3 and PostgreSQL
        save_data(data)
        
        print("Latest data generation completed successfully")
        return True
    except Exception as e:
        print(f"Error in generate_latest_data: {str(e)}")
        raise

def ingest_etl(**context):
    """Process ETL pipeline"""
    try:
        etl = IncrementalETL()
        etl.run_etl()
        print("ETL processing completed successfully")
        return True
    except Exception as e:
        print(f"Error in ingest_etl: {str(e)}")
        raise

# Define tasks
generate_data_task = PythonOperator(
    task_id='generate_latest_data',
    python_callable=generate_latest_data,
    provide_context=True,
    dag=dag,
)

ingest_etl_task = PythonOperator(
    task_id='ingest_etl',
    python_callable=ingest_etl,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
generate_data_task >> ingest_etl_task