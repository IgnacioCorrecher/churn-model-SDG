from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_hello():
    print("Hello, World!")
    with open('../data/output.txt', 'w') as f:
        f.write("Hello, World!")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 7, 19),
    'retries': 1,
}

dag = DAG(
    'example_dag',
    default_args=default_args,
    description='An example DAG',
    schedule_interval='@daily',
)

t1 = PythonOperator(
    task_id='print_hello',
    python_callable=print_hello,
    dag=dag,
)
