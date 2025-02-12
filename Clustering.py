import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import the imputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Wider print limits
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)

import Market_Data_Feed as mdf

#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py


def run(settings):

    #Update settings
    settings['start']='1996-01-01'

    # DATA & INDICATORS

    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns = data.tickers_returns

    # Example Usage:
    window_size = 250
    step_size = 22*6
    n_clusters = 6
    n_jobs = -1
    returns=tickers_returns.copy()

    """
    results_df = rolling_corr_and_cluster(returns, window_size, step_size, n_clusters, linkage="average", n_jobs=n_jobs, use_euclidean=False)
    print("\nResults :")
    print(results_df)
    clusters=pd.DataFrame(results_df['clusters'].values.tolist(),columns=returns.columns,index=results_df.index)
    """


    window_size_months=12*3
    clusters=KMeans_Clustering_Monthly(returns=returns,window_size_months=window_size_months)



    print('clusters',clusters)

    #[['ES=F','NQ=F']]
    clusters.plot(title='Clusters')

    #best_assets_df = select_best_assets_as_df(results_df, returns)  # Pass returns to the function
    #print("\nBest Assets Selected :")
    #print(best_assets_df)
    #best_assets_df.plot(title='best_assets_df')

    plt.show()

def KMeans_Clustering_Monthly(returns, window_size_months):  # Changed window size to months

    returns_res=returns.resample('M').mean()

    # Get Rolling Correlation (monthly)
    rolling_correlations = returns_res.rolling(window=window_size_months, min_periods=1).corr().fillna(0).clip(lower=-1, upper=1)

    # Initialize a DataFrame to store cluster assignments over time (monthly)
    monthly_index = returns_res.index[window_size_months - 1:]
    cluster_evolution = pd.DataFrame(index=monthly_index, columns=returns.columns)

    # Perform rolling window clustering (monthly)
    for i in range(window_size_months, len(returns_res)):
        # Extract the rolling window of returns (monthly)
        window_date = returns_res.index[i - window_size_months]

        # Compute the correlation matrix (using the pre-calculated rolling_correlations)
        corr_matrix = rolling_correlations.loc[window_date]

        # Perform clustering (e.g., KMeans with 4 clusters)
        kmeans = KMeans(n_clusters=len(returns.columns)-1, random_state=42)
        clusters = kmeans.fit_predict(corr_matrix)

        # Store the cluster assignments for the current window
        cluster_evolution.iloc[i - window_size_months] = clusters


    cluster_evolution=cluster_evolution.rolling(3).mean()



    return cluster_evolution


def cluster_one_window_euclidean(data, n_clusters, linkage):
    """Clusters a single data window using Euclidean distance."""
    data_scaled = StandardScaler().fit_transform(data)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return clustering.fit_predict(data_scaled)  # or .labels_


def cluster_one_window_precomputed(corr_matrix, n_clusters, linkage):
    """Clusters a single correlation matrix (precomputed affinity)."""
    distance_matrix = 1 - corr_matrix
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    return clustering.fit_predict(distance_matrix)

def rolling_corr_and_cluster(returns, window_size, step_size, n_clusters=3, linkage="ward", n_jobs=-1, use_euclidean=True):
    """Calculates rolling correlations and clusters with parallelization."""

    num_assets = returns.shape[1]
    num_windows = max(0, (len(returns) - window_size) // step_size + 1)
    results = []

    original_index = returns.index
    returns_int_indexed = returns.set_index(np.arange(len(returns)))  # Integer indexing for parallelization

    rolling_correlations = returns.rolling(window=window_size, min_periods=1).corr().fillna(0).clip(lower=-10,upper=10)  # Calculate rolling correlations
    rolling_means = returns.rolling(window=window_size, min_periods=1).mean()  # Calculate rolling means
    rolling_std = returns.rolling(window=window_size, min_periods=1).std()
    rolling_sharpe=rolling_means/(rolling_std+0.000001)
    rolling_performance=rolling_sharpe.copy()

    if num_windows > 0:
        window_indices = [list(range(i * step_size, i * step_size + window_size)) for i in range(num_windows)]
        window_indices = [window for window in window_indices if all(j < len(returns_int_indexed) for j in window)]

        all_data_for_clustering = []
        all_performances = []
        all_dates = []
        all_rolling_correlations = []  # Store correlation matrices

        for i, window in enumerate(window_indices):
            window_data = returns_int_indexed.iloc[window]
            # *** KEY FIX: Transpose window_data before clustering ***
            window_data_transposed = window_data.T  # Transpose for asset-based clustering

            # Impute NaN values in window_data BEFORE clustering (for Euclidean)
            imputer_euclidean = SimpleImputer(strategy='mean')  # Use mean imputation
            window_data_imputed_euclidean = imputer_euclidean.fit_transform(window_data_transposed)  # Impute the data
            window_data_imputed_euclidean_transposed = window_data_imputed_euclidean.T  # Transpose back to original shape

            all_data_for_clustering.append(window_data_imputed_euclidean_transposed)  # Append the IMPUTED and TRANSPOSED data

            performance_date = original_index[window[-1]]  # Date for performance

            # Get the correlation matrix for the date
            corr_matrix = rolling_correlations.loc[performance_date].values

            if corr_matrix is not None:
                # Impute NaNs in correlation matrix (for precomputed)
                imputer_precomputed = SimpleImputer(strategy='mean')
                corr_matrix_imputed = imputer_precomputed.fit_transform(corr_matrix)
                all_rolling_correlations.append(corr_matrix_imputed)
            else:
                all_rolling_correlations.append(np.full((num_assets, num_assets), np.nan))  # Handle if no correlation available yet

            asset_performance = rolling_performance.loc[performance_date]  # Get mean returns for the date
            if isinstance(asset_performance, pd.Series):  # Check if it is a series. If not, then it is a DataFrame, so we need to get the first row
                asset_performance = asset_performance.values
            elif isinstance(asset_performance, pd.DataFrame):
                asset_performance = asset_performance.iloc[0].values
            else:
                asset_performance = np.full(returns.shape[1], np.nan)  # Handle case if the date is not present in rolling_means

            all_performances.append(asset_performance)
            all_dates.append(original_index[window[-1]])  # use original index

        if len(all_data_for_clustering) > 0:
            if use_euclidean:
                results_clustering = Parallel(n_jobs=n_jobs)(
                    delayed(cluster_one_window_euclidean)(data, n_clusters, linkage) for data in all_data_for_clustering
                )
            else:  # Use precomputed affinity matrix
                results_clustering = Parallel(n_jobs=n_jobs)(
                    delayed(cluster_one_window_precomputed)(corr, n_clusters, linkage) for corr in all_rolling_correlations  # Use rolling_correlations
                )

            final_results = []
            for i, cluster_labels in enumerate(results_clustering):
                cluster_labels = np.array(cluster_labels)
                if cluster_labels.ndim > 1:
                    cluster_labels = cluster_labels.flatten()

                final_results.append({
                    "date": all_dates[i],
                    "clusters": cluster_labels,
                    "performance": all_performances[i]
                })

            results = pd.DataFrame(final_results)
            results = results.set_index('date')  # Set 'date' as index HERE

    return results  # Return the DataFrame with 'date' as index

def select_best_assets_as_df_nok(results_df, returns):
    """Selects best assets and returns as DataFrame with 1s and 0s."""

    all_best_assets = []
    for date in results_df.index: # Iterate through the INDEX (dates)
        row = results_df.loc[date] # Get the row using .loc and the date
        clusters = row['clusters']
        performance = row['performance']

        num_assets = len(returns.columns)
        best_assets_for_window = {}

        if not np.isnan(performance).all():
            for cluster_id in np.unique(clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_performance = performance[cluster_indices]
                valid_indices = ~np.isnan(cluster_performance)
                if np.any(valid_indices):
                    best_asset_index = cluster_indices[np.nanargmax(cluster_performance[valid_indices])]
                    best_assets_for_window[cluster_id] = best_asset_index
                else:
                    best_assets_for_window[cluster_id] = None

            # Efficiently create DataFrame with correct index:
            window_data = np.zeros((1, len(returns.columns)), dtype=int)  # Initialize with zeros
            window_df = pd.DataFrame(window_data, columns=returns.columns, index=[date]) # Set index in creation

            for cluster_id, best_asset_index in best_assets_for_window.items():
                if best_asset_index is not None:
                    asset_name = returns.columns[best_asset_index]
                    window_df.loc[date, asset_name] = 1 # Set 1 for best assets

            all_best_assets.append(window_df)
        else:
          window_data = np.zeros((1, len(returns.columns)), dtype=int)  # Initialize with zeros
          window_df = pd.DataFrame(window_data, columns=returns.columns, index=[date]) # Set index in creation
          all_best_assets.append(window_df)


    final_df = pd.concat(all_best_assets)
    return final_df

def select_best_assets_as_df(results_df, returns):
    """Selects best assets and returns as DataFrame with 1s and 0s."""

    all_best_assets = []
    for date, row in results_df.iterrows():
        clusters = row['clusters']
        performance = row['performance']

        num_assets = len(returns.columns)
        best_assets_for_window = {}

        if not np.isnan(performance).all():  # Check if ALL values are NaN
            for cluster_id in np.unique(clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]

                # *** KEY FIX: Ensure indices are within bounds ***
                valid_cluster_indices = cluster_indices[cluster_indices < len(performance)] # Added this line
                cluster_performance = performance[valid_cluster_indices] # Use valid indices

                valid_indices = ~np.isnan(cluster_performance)  # Check if there are ANY valid values
                if np.any(valid_indices):
                    best_asset_index = valid_cluster_indices[np.nanargmax(cluster_performance[valid_indices])] # Use valid indices
                    best_assets_for_window[cluster_id] = best_asset_index
                else:
                    best_assets_for_window[cluster_id] = None  # No best asset if all are NaN

            window_data = np.zeros((1, len(returns.columns)), dtype=int)
            window_df = pd.DataFrame(window_data, columns=returns.columns, index=[date])

            for cluster_id, best_asset_index in best_assets_for_window.items():
                if best_asset_index is not None:
                    asset_name = returns.columns[best_asset_index]
                    window_df.loc[date, asset_name] = 1

            all_best_assets.append(window_df)
        else:
            window_data = np.zeros((1, len(returns.columns)), dtype=int)
            window_df = pd.DataFrame(window_data, columns=returns.columns, index=[date])
            all_best_assets.append(window_df)

    final_df = pd.concat(all_best_assets)
    return final_df

def random_returns(num_rows = 100,columns=['A', 'B', 'C', 'D', 'E'],start='2023-01-01'):
    np.random.seed(0)
    num_cols = len(columns)
    data = np.random.randn(num_rows, num_cols)
    returns = pd.DataFrame(data, columns=columns)
    returns.index = pd.to_datetime(pd.date_range(start, periods=num_rows))

    return returns

if __name__ == '__main__':
    run(settings)




