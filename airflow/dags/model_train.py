import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
import mlflow

class ModelTrainer:
    
    """ Clase para estructurar los métodos encargados de el entrenamiento del modelo"""
    
    def __init__(self, data_path='/opt/airflow/dags/data/data_clean.csv', test_size=0.2, random_state=77, cv_folds=5):
        
        """Constructor de la clase"""
        
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.pipeline = None
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'roc_auc': 'roc_auc',
            'recall': make_scorer(recall_score),
            'precision': make_scorer(precision_score),
            'f1': make_scorer(f1_score)
        }

    def load_data(self):
        
        """ Cargamos los datos y hacemos un train/test split"""
        
        data = pd.read_csv(self.data_path, delimiter=',', decimal='.')
        X = data.drop(['churn'], axis=1)
        y = data['churn']
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

    def create_pipeline(self, X_train):
        
        """ Creamos la pipeline por donde pasaran nuestros datos y se entrenará el modelo."""
        
        cat_features = X_train.select_dtypes(include=['object']).columns
        num_features = X_train.select_dtypes(include=['int64', 'float64']).columns

        # Pasamos los datos por un scaler y creamos nuevas variables con relación polinómica de grado 2
        num_transf = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
        ])

        # Realizamos un OHE sobre las columnas categóricas y montamos el transformer
        
        transformer = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', num_transf, num_features)
        ])


        # Inicializamos el RFC con los mejores parámetros para el modelo
        
        RFC = RandomForestClassifier(
            random_state=self.random_state,
            criterion='gini',
            max_depth=10,
            max_features='sqrt',
            n_estimators=512
        )

        self.pipeline = Pipeline(steps=[
            ('transformer', transformer),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', RFC)
        ])

    def train_and_evaluate(self):
        
        """ Crearemos la pipeline con los métodos anteriores y entrenamos el modelo. Finalmente lo evaluamos """
        
        X_train, X_test, y_train, y_test = self.load_data()
        self.create_pipeline(X_train)

        tracking_uri ="file:/opt/airflow/dags/data/mlruns"
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = "RandomForestClassifier"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            cv_results = cross_validate(self.pipeline, X_train, y_train, cv=self.cv_folds, scoring=self.scoring, return_train_score=True)

            mlflow.sklearn.log_model(self.pipeline, "model")

            mlflow.log_params({
                "criterion": 'gini',
                "max_depth": 10,
                "max_features": 'sqrt',
                "n_estimators": 512,
                "cv_folds": self.cv_folds
            })

            for metric in self.scoring.keys():
                mlflow.log_metric(f"mean_train_{metric}", cv_results[f"train_{metric}"].mean())
                mlflow.log_metric(f"std_train_{metric}", cv_results[f"train_{metric}"].std())

            self.pipeline.fit(X_train, y_train)

            test_metrics = {
                "accuracy": accuracy_score(y_test, self.pipeline.predict(X_test)),
                "roc_auc": roc_auc_score(y_test, self.pipeline.predict_proba(X_test)[:, 1]),
                "recall": recall_score(y_test, self.pipeline.predict(X_test)),
                "precision": precision_score(y_test, self.pipeline.predict(X_test)),
                "f1": f1_score(y_test, self.pipeline.predict(X_test))
            }

            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)

        print("¡Entrenamiento terminado!")
