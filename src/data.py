import neurodatasets as nd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    df = nd.load_dataset('sleep_study_college')
    drop_cols = ['Gender', 'ClassYear', 'LarkOwl', 'NumEarlyClass', 'EarlyClass',
                 'DepressionStatus', 'AnxietyStatus', 'Stress', 'AlcoholUse']
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=feature_names), scaler, feature_names
