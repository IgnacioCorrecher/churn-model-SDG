from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from data_cleaning import DataProcessor
from model_train import ModelTrainer

# Definimos los parámetros por defecto
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Inicializamos el DAG
dag = DAG(
    'churn_model_DAG',
    default_args=default_args,
    description='A pipeline to process and train the churn model',
    schedule_interval='@daily',
    catchup=False,
)

# Definimos las diferentes tareas que van a existir en el DAG

def run_data_processor():
    data_processor = DataProcessor()
    data_processor.clean_data()

def run_model_training():
    trainer = ModelTrainer()
    trainer.train_and_evaluate()

task1 = PythonOperator(
    task_id='data_processing',
    python_callable=run_data_processor,
    dag=dag,
)

task2 = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag,
)

# Establecemos las dependencias entre las tareas
task1 >> task2
