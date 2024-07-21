import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

class DataProcessor:
    """ Clase para estructurar todos los métodos usados en la depuración del dataframe inicial """
    
    def __init__(self, miss_pct_th=33, th_num=0.05, th_chi=0.05, outlier_rate=3, corr_th=0.8):
        
        """Constructor de clase"""
        
        self.miss_pct_th = miss_pct_th
        self.th_num = th_num
        self.th_chi = th_chi
        self.outlier_rate = outlier_rate
        self.corr_th = corr_th

    def correct_missings(self, df):
        
        """ El objetivo de esta función es arreglar los valores missings del dataset. Este proceso se divide en 3 etapas
        
        - Fase 1: Se dropean las columnas que tienen un porcentaje de nulos mayor de lo indicado. De tal forma que el resto de columnas con missings se dividen en 2 grupos, numéricas y categóricas.
        - Fase 2: Se estudia la correlación de las columnas numéricas con la columna churn mediante una matriz de correlación, todas aquellas que tengan menor correlación a lo indicado se dropean, las que tengan buena correlación se imputan con la mediana en este caso.
        - Fase 3: Se estudia la correlación de las columnas categóricas con la columna churn, esto se hace mediante un test chi cuadrado. Se dropearan las columnas con p-valor > 0.05 y se imputaran con la moda aquellas con p-valor significativo
        
        """
        
        missings_pct = (df.isnull().sum() / len(df)) * 100

        cols_to_drop = missings_pct[(missings_pct > self.miss_pct_th)].index.tolist()
        df = df.drop(columns=cols_to_drop)

        columns_missings = missings_pct[(missings_pct < self.miss_pct_th) & (missings_pct > 0)].index.tolist()
        df_missings = df[columns_missings + ['churn']]

        df_num_missings = df_missings.select_dtypes(include=[np.number])
        df_cat_missings = df_missings.select_dtypes(include=[object])

        corr_with_churn = df_num_missings.corrwith(df_num_missings['churn'])
        cols_to_keep = corr_with_churn[abs(corr_with_churn) >= self.th_num].index.tolist()
        cols_to_impute = [col for col in cols_to_keep if col != 'churn']

        for col in cols_to_impute:
            median = df_num_missings[col].median()
            df[col] = df[col].fillna(median)

        cols_to_drop = corr_with_churn[abs(corr_with_churn) < self.th_num].index.tolist()
        df = df.drop(columns=cols_to_drop)

        if 'churn' not in df_cat_missings.columns:
            df_cat_missings['churn'] = df['churn']

        def chi2_test(cols, target):
            cont_table = pd.crosstab(cols, target)
            res = chi2_contingency(cont_table)
            return res.pvalue

        chi2_res = df_cat_missings.apply(lambda x: chi2_test(x, df['churn'])).sort_values()
        cols_to_keep = chi2_res[chi2_res <= self.th_chi].index.tolist()
        cols_to_impute = [col for col in cols_to_keep if col != 'churn']

        for col in cols_to_impute:
            mode = df_cat_missings[col].mode()[0]
            df[col] = df[col].fillna(mode)

        cols_to_drop = chi2_res[chi2_res > self.th_chi].index.tolist()
        df = df.drop(columns=cols_to_drop)

        return df

    def correct_outliers(self, df):
        
        """El objetivo de esta función es corregir los valores extremos o outliers del dataset. Se calculan los cuartiles 1 y 3, y con ello el rango intercuartílico.
        Todos los valores de cada columna que caigan fuera de este rango multiplicado por un factor designado como parámetro serán imputados con la mediana.
        """
        
        df = df.select_dtypes(include=[np.number])
        features = df.columns.to_list()

        for f in features:
            Q1 = np.percentile(df[f], 25)
            Q3 = np.percentile(df[f], 75)
            IQR = Q3 - Q1

            low_bound = Q1 - (IQR * self.outlier_rate)
            up_bound = Q3 + (IQR * self.outlier_rate)

            median = df[f].median()
            df[f] = np.where((df[f] < low_bound) | (df[f] > up_bound), median, df[f])

        df = df.loc[:, (df != df.iloc[0]).any()]
        return df

    def feature_engineering(self, df):
        
        """ El objetivo de esta función es eliminar las columnas que tengan mucha correlación entre sí. Para ello seguimos los siguientes pasos:
        
            - Fase 1: Se calcula la matriz de correlación, pero nos vale con la triangular superior únicamente, ya que se trata de una matriz simétrica.
            - Fase 2: Iteramos por las columnas de la matriz y asignamos, en una lista, un integrante de cada pareja de aquellas que tengan correlación superior a un valor designado.
            - Fase 3: Dropeamos las columnas que hemos recogido.
        """
        
        df = df.select_dtypes(include=[np.number])
        corr_matrix = df.corr().abs()
        corr_triu = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        cols_to_drop = []

        for column in corr_triu.columns:
            for row in corr_triu.index:
                if corr_triu.loc[row, column] > self.corr_th:
                    print(f"Par: ({row}, {column}) - Correlación: {corr_triu.loc[row, column]:.2f}")
                    cols_to_drop.append(column)
                    break

        df = df.drop(columns=cols_to_drop)
        return df

    def clean_data(self):
        
        """Designamos los directorios de entrada y salida y ejecutamos las funciones, antes de ejecutar ninguna dropeamos la columna Customer_ID, ya que es un valor único y no aporta nada de cara a la predicción, de hecho puede resultar hasta negativa."""
        
        input_path = os.path.abspath('/opt/airflow/dags/data/dataset.csv')
        output_path = os.path.abspath('/opt/airflow/dags/data/data_clean.csv')
        data = pd.read_csv(input_path, delimiter=';', decimal=',')

        data.drop(["Customer_ID"], axis=1, inplace=True)

        data = self.correct_missings(data)
        data = self.correct_outliers(data)
        data = self.feature_engineering(data)

        data.to_csv(output_path, index=False)
