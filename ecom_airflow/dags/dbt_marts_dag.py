import os
from datetime import datetime
from cosmos import DbtDag, ProjectConfig, ProfileConfig, ExecutionConfig
from cosmos.profiles import SnowflakeUserPasswordProfileMapping
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

snowflake_base_config = {
    "type": "snowflake",
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "schema": "ECOM_MARTS",
    "threads": 10
}

# Profile config for marts schema
marts_profile_config = ProfileConfig(
    profile_name="ecom_marts",
    target_name="dev",
    profile_mapping=SnowflakeUserPasswordProfileMapping(
        conn_id="snowflake_conn",
        profile_args=snowflake_base_config
    )
)

# Create DAG for marts models
marts_dag = DbtDag(
    project_config=ProjectConfig(
        dbt_project_path="/usr/local/airflow/dags/dbt/ecom_marts"
    ),
    operator_args={
        "install_deps": True,
        "full_refresh": True
    },
    profile_config=marts_profile_config,
    execution_config=ExecutionConfig(
        dbt_executable_path=f"{os.environ['AIRFLOW_HOME']}/dbt_venv/bin/dbt"
    ),
    schedule_interval="@daily",
    start_date=datetime(2024, 12, 1),
    catchup=False,
    dag_id="dbt_marts_dag",
)