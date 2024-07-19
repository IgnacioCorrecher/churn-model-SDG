import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sqlalchemy import create_engine

class DataProcessor:
    def __init__(self, miss_pct_th=33, th_num=0.05, th_chi=0.05, outlier_rate=3, corr_th=0.8):
        self.miss_pct_th = miss_pct_th
        self.th_num = th_num
        self.th_chi = th_chi
        self.outlier_rate = outlier_rate
        self.corr_th = corr_th

    def correct_missings(self, df):
        missings_pct = (df.isnull().sum() / len(df)) * 100

        cols_to_drop = missings_pct[(missings_pct > self.miss_pct_th)].index.tolist()
        print("Se van a dropear las siguientes por excesivos nulos:\n", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

        columns_missings = missings_pct[(missings_pct < self.miss_pct_th) & (missings_pct > 0)].index.tolist()
        df_missings = df[columns_missings + ['churn']]

        df_num_missings = df_missings.select_dtypes(include=[np.number])
        df_cat_missings = df_missings.select_dtypes(include=[object])

        corr_with_churn = df_num_missings.corrwith(df_num_missings['churn'])
        cols_to_keep = corr_with_churn[abs(corr_with_churn) >= self.th_num].index.tolist()
        cols_to_impute = [col for col in cols_to_keep if col != 'churn']

        print("Se van a imputar con la mediana las siguientes columnas:\n", cols_to_impute)
        for col in cols_to_impute:
            median = df_num_missings[col].median()
            df[col] = df[col].fillna(median)

        cols_to_drop = corr_with_churn[abs(corr_with_churn) < self.th_num].index.tolist()
        print("Se van a dropear las siguientes por baja correlación con la columna churn:\n", cols_to_drop)
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

        print("Se van a imputar con la moda las siguientes columnas:\n", cols_to_impute)
        for col in cols_to_impute:
            mode = df_cat_missings[col].mode()[0]
            df[col] = df[col].fillna(mode)

        cols_to_drop = chi2_res[chi2_res > self.th_chi].index.tolist()
        print("Se van a dropear las siguientes por baja correlación con la columna churn:\n", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

        return df

    def correct_outliers(self, df):
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

        df_red = df.drop(columns=cols_to_drop)
        return df_red

    def save_data_to_postgres(self, data, table_name):
        engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres/airflow')
        data.to_sql(table_name, engine, if_exists='replace', index=False)


def main():
    data_processor = DataProcessor()

    data = pd.read_csv('../../data/dataset.csv', delimiter=';', decimal=',')
    data_processor.save_data_to_postgres(data, 'data_raw')

    data.drop(["Customer_ID"], axis=1, inplace=True)

    data = data_processor.correct_missings(data)
    data = data_processor.correct_outliers(data)
    data = data_processor.feature_engineering(data)

    data_processor.save_data_to_postgres(data, 'data_clean')


if __name__ == "__main__":
    main()
