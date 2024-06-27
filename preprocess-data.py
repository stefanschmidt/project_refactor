import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import RandomOverSampler
import plotly.express as px

df = pd.read_pickle('data/telecom_users.pkl')
df_original = df.copy()

if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

df = df.drop_duplicates()

if 'TotalCharges' in df.columns:
    df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True, inplace=True)

imputer = SimpleImputer(strategy='median')
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = imputer.fit_transform(df['TotalCharges'].values.reshape(-1, 1))

numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

categorical_columns = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_columns)

ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(df.drop('Churn', axis=1), df['Churn'])
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

#X = df_resampled.drop('Churn', axis=1)
df_resampled = df_resampled.drop('customerID', axis=1)

df_resampled.to_pickle('data/telecom_users_preprocessed.pkl')
df_resampled.to_csv('data/telecom_users_preprocessed.csv')


