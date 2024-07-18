import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def correct_missings(df, miss_pct_th = 33, th_num = 0.05, th_chi=0.05):
    
    missings_pct = (df.isnull().sum()/len(df)) * 100
    
    # Eliminamos directamente las columnas con un % de missing superior al 33%

    cols_to_drop = missings_pct[(missings_pct > miss_pct_th)].index.tolist()
    
    print("Se van a dropear las siguientes por excesivos nulos:\n", cols_to_drop)

    df = df.drop(columns = cols_to_drop)
    
    # Las columnas con missings entre 0 y 33 las dividiremos en 2 grupos, numéricas y categóricas:
    
    columns_missings = missings_pct[(missings_pct < miss_pct_th) & (missings_pct > 0)].index.tolist()
    df_missings = df[columns_missings + ['churn']]
    
    df_num_missings = df_missings.select_dtypes(include=[np.number])
    df_cat_missings = df_missings.select_dtypes(include=[object])
    
    # Numéricas
    #print("Las columnas numéricas con nulos son las siguientes:\n", df_num_missings.index )
    
    corr_with_churn = df_num_missings.corrwith(df_num_missings['churn'])
    
    cols_to_keep = corr_with_churn[abs(corr_with_churn) >= th_num].index.tolist()
    
    cols_to_impute = [col for col in cols_to_keep if col != 'churn']

    print("Se van a imputar con la mediana las siguientes columnas:\n", cols_to_impute)

    for col in cols_to_impute:
        median = df_num_missings[col].median()
        df[col] = df[col].fillna(median)
    
    cols_to_drop = corr_with_churn[abs(corr_with_churn) < th_num].index.tolist()
    
    print("Se van a dropear las siguientes por baja correlación con la columna churn:\n", cols_to_drop)
    
    df = df.drop(columns = cols_to_drop)
    
    # Categóricas
    
    if 'churn' not in df_cat_missings.columns:
        df_cat_missings['churn'] = df['churn']
        
    def chi2_test(cols, target):
        cont_table = pd.crosstab(cols, target)
        res = chi2_contingency(cont_table)
        return res.pvalue
    
    chi2_res = df_cat_missings.apply(lambda x: chi2_test(x, df['churn'])).sort_values()
    
    #print(chi2_res)
    
    cols_to_keep = chi2_res[chi2_res <= th_chi].index.tolist()

    cols_to_impute = [col for col in cols_to_keep if col != 'churn']
    
    print("Se van a imputar con la moda las siguientes columnas:\n", cols_to_impute)
    
    for col in cols_to_impute:
        mode = df_cat_missings[col].mode()[0]
        df[col] = df[col].fillna(mode)
            
    cols_to_drop = chi2_res[chi2_res > th_chi].index.tolist()
    
    print("Se van a dropear las siguientes por baja correlación con la columna churn:\n", cols_to_drop)
    
    df = df.drop(columns = cols_to_drop)
    
    return df
    
def correct_outliers(df, outlier_rate = 3):
    df = df.select_dtypes(include=[np.number])
    features = df.columns.to_list()

    for f in features:
        Q1 = np.percentile(df[f],25)
        Q3 = np.percentile(df[f],75)
        IQR = Q3 - Q1

        low_bound = Q1 - (IQR * outlier_rate)
        up_bound = Q3 + (IQR * outlier_rate)

        median = df[f].median()

        df[f] = np.where((df[f] < low_bound) | (df[f] > up_bound), median, df[f])

        df = df.loc[:, (df != df.iloc[0]).any()]

    return df


def feature_engineering(df, th = 0.8):
    df = df.select_dtypes(include=[np.number])
    corr_matrix = df.corr().abs()
    corr_triu = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = []
    for column in corr_triu.columns:
        for row in corr_triu.index:
            if corr_triu.loc[row, column] > th:
                print(f"Par: ({row}, {column}) - Correlación: {corr_triu.loc[row, column]:.2f}")
                cols_to_drop.append(column)
                break
    df_red = df.drop(columns = cols_to_drop)

    return df_red


data = pd.read_csv('../data/dataset.csv', delimiter=';', decimal=',')

data.drop(["Customer_ID"], axis = 1, inplace=True)

data = correct_missings(data, miss_pct_th = 0.33)

data = correct_outliers(data)

data = feature_engineering(data)

data.to_csv('../data/data_clean.csv', index=False)

