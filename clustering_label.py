import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import RandomOverSampler
import plotly.express as px
import plotly.io as pio
import os

pio.renderers.default = 'png'

def clustering():
    df = pd.read_pickle('data/telecom_users.pkl')
    ds_original = df.copy()

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

    X = df_resampled.drop('Churn', axis=1)
    X = X.drop('customerID', axis=1)

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = cluster_labels

    pca_plot = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', title='PCA Result with Cluster Coloring (k=2)')
    pca_plot_png_path = 'static/pca_plot.png'
    pca_plot.write_image(pca_plot_png_path)

    return pca_plot_png_path
