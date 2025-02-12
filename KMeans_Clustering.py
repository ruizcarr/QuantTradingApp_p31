import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def KMeans_Clustering(returns, window_size):
    # Get Rolling Correlation
    rolling_correlations = returns.rolling(window=window_size, min_periods=1).corr().fillna(0).clip(lower=-10, upper=10)  # Calculate rolling correlations

    # Initialize a DataFrame to store cluster assignments over time
    cluster_evolution = pd.DataFrame(index=returns.index[window_size - 1:], columns=returns.columns)

    # Perform rolling window clustering
    for i in range(window_size, len(returns)):
        # Extract the rolling window of returns
        window_date = returns.iloc[i - window_size:i].index[-1]

        # Compute the correlation matrix
        corr_matrix = rolling_correlations.loc[window_date]

        # Perform clustering (e.g., KMeans with 4 clusters)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(corr_matrix)

        # Store the cluster assignments for the current window
        cluster_evolution.iloc[i - window_size] = clusters

    return cluster_evolution

