from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

data = pd.read_csv('data/data_clean.csv', delimiter=',', decimal='.')


X = data[data.columns.difference(['churn'])]
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

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=77),
        'params': {
            'classifier__C': [ 1, 10, 100],
            'classifier__max_iter': [256, 512, 1024]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=77),
        'params': {
            'classifier__n_estimators': [128, 256, 512],
            'classifier__max_depth': [None, 10, 20],
            'classifier__criterion':['gini', 'entropy', 'log_loss'],
            'classifier__max_features':['sqrt', 'log2']
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=77),
        'params': {
            'classifier__n_estimators': [128, 256, 512],
            'classifier__learning_rate': [0.01, 0.1, 1]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'classifier__n_neighbors': [5, 7, 9]}
    }
}

best_model = None
best_params = None
best_score = -1
best_model_name = None

for name, model_dict in models.items():
    pipe = Pipeline([
        ('preprocessing', transformer),
        ('feature_selection', SelectKBest(f_classif, k=20)),
        ('classifier', model_dict['model'])
    ])
    
    grid = GridSearchCV(pipe, param_grid=model_dict['params'], cv=2, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], refit='roc_auc', verbose = 3)
    grid.fit(X_train, y_train)
    
    if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            best_model_name = name
    
    print(f"Los mejores parámetros para el modelo {name} son: {grid.best_params_}")
    print(f" La CV score para el modelo {name} es: {grid.best_score_:.2f}")

print(f"\nEl mejor modelo es {best_model_name} con parámetros {best_params} y un score de {best_score:.2f}")