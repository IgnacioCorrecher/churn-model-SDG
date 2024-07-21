from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
import pandas as pd
import mlflow
import datetime


data = pd.read_csv('data/data_clean.csv', delimiter=',', decimal='.')


#X = data[data.columns.difference(['churn'])]
X = data.drop(['churn'], axis=1)
y = data['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77,stratify=y)


OHE = OneHotEncoder(handle_unknown='ignore')
scaler = RobustScaler()
RFC = RandomForestClassifier(random_state=77)

cat_features = X_train.select_dtypes(include=['object']).columns
num_features = X_train.select_dtypes(include = ['int64', 'float64']).columns

num_transf = Pipeline(steps=[
    ('scaler', scaler),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
])

transformer = ColumnTransformer([('cat', OHE, cat_features), ('num', num_transf, num_features)])

RFC = RandomForestClassifier(
    random_state=77,
    criterion='gini',
    max_depth=10,
    max_features='sqrt',
    n_estimators=512
)

pipeline = Pipeline(steps=[
    ('transformer', transformer),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RFC)
])

mlflow.set_experiment(f"RandomForestClassifier")


scoring = {
    'accuracy': make_scorer(accuracy_score),
    'roc_auc': 'roc_auc',
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score),
    'f1': make_scorer(f1_score)
}

with mlflow.start_run():
    cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)
    
    mlflow.sklearn.log_model(pipeline, "model")
    
    mlflow.log_params({
        "criterion": 'gini',
        "max_depth": 10,
        "max_features": 'sqrt',
        "n_estimators": 512,
        "cv_folds": 5
    })
    
    for metric in scoring.keys():
        mlflow.log_metric(f"mean_train_{metric}", cv_results[f"train_{metric}"].mean())
        mlflow.log_metric(f"std_train_{metric}", cv_results[f"train_{metric}"].std())

    pipeline.fit(X_train, y_train)

    test_metrics = {
        "accuracy": accuracy_score(y_test, pipeline.predict(X_test)),
        "roc_auc": roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]),
        "recall": recall_score(y_test, pipeline.predict(X_test)),
        "precision": precision_score(y_test, pipeline.predict(X_test)),
        "f1": f1_score(y_test, pipeline.predict(X_test))
    }

    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)

print("Listo!")