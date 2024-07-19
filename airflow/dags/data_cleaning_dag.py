from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
import os

sys.path.insert(0, '/opt/airflow/dags')

from data_cleaning import main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'data_cleaning_dag',
    default_args=default_args,
    description='A simple data cleaning DAG',
    schedule_interval='@daily',
    start_date=days_ago(1),
    tags=['example'],
)

data_cleaning_task = PythonOperator(
    task_id='clean_data',
    python_callable=main,
    dag=dag,
)

data_cleaning_task
