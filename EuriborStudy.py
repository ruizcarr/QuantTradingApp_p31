# import libraries and functions
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import time

from sklearn.linear_model import LinearRegression

import os

import quantstats
import webbrowser


# Wider print limits
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')

from WalkForwardTraining import WalkForwardTraining,get_params_from_csv
import Market_Data_Feed as mdf
from Backtest_Vectorized import compute_backtest_vectorized
from Training_Markowitz import process_log_data,apply_pos_constrain

#Get SETTINGS
from config import settings
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

def compute(settings):

    start_time = time.time()

    #Update settings
    settings['start']='1996-01-01'

    # Get Data & Indicators
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns = data.tickers_returns
    data_dict=data.data_dict

    print('tickers_returns',tickers_returns)


    if True:
        #Training & Saving Euribor_weigths
        get_trained_Euribor_weigths(tickers_returns)

    #Retrieve Training model and get Euribor Ind
    Euribor_series = tickers_returns['cash'] * 255 * 100
    Euribor_ind = get_Euribor_ind(Euribor_series)

    strategy_returns=tickers_returns*Euribor_ind.shift(1)
    strategy_cum_ret = (1 + strategy_returns).cumprod()
    metrics=pd.DataFrame()
    metrics['ticker_CAGR']=tickers_returns.mean()*255
    metrics['ticker_volat']=tickers_returns.std()*16
    metrics['ticker_sharpe'] =metrics['ticker_CAGR']/metrics['ticker_volat']
    metrics['strat_CAGR'] = strategy_returns.mean() * 255
    metrics['strat_volat'] = strategy_returns.std() * 16
    metrics['strat_sharpe'] = metrics['strat_CAGR'] / metrics['strat_volat']
    metrics['strat_sharpe_improve'] =metrics['strat_sharpe']-metrics['ticker_sharpe']
    metrics['drawdown']=(strategy_cum_ret/strategy_cum_ret.rolling(250).max()-1).min()

    print('metrics',metrics.T)


    cum_ret=(1+tickers_returns).cumprod()


    for ticker in settings['tickers']:
        plot_df=pd.DataFrame()
        plot_df[ticker]=cum_ret[ticker]
        plot_df['Euribor'] = Euribor_series
        plot_df['strategy_cum_returns']=strategy_cum_ret[ticker]
        plot_df['Euribor_ind']=Euribor_ind[ticker]

        plot_df.plot(title=ticker)

    plt.show()


    return

def get_trained_Euribor_weigths(tickers_returns,folder_path="trained_models", filename="Euribor_weights.csv"):

    Euribor_series = tickers_returns['cash'] * 255 * 100

    #Get Euribor Metrics
    Euribor_metrics=get_Euribor_metrics(Euribor_series)

    #Compute Regresion
    #regr_df=Euribor_metrics[['Euribor_is_high','Euribor_is_mid','Euribor_is_low','Euribor_down','Euribor_up']] #'Euribor_is_not_low','Euribor_is_not_high','Euribor_not_fast_up'

    # Compute Correlation matrix
    corr_matrix = calculate_all_correlations_matrix(Euribor_metrics,tickers_returns)
    print('corr_matrix',corr_matrix)

    #Create Euribor_weights
    Euribor_weights = corr_matrix.clip(0)
    Euribor_weights = Euribor_weights/Euribor_weights.sum()

    # Construct the full file path
    file_path = os.path.join(folder_path, filename)

    # Save the DataFrame to a CSV file
    Euribor_weights.to_csv(file_path)

    print('Euribor_weights',Euribor_weights)

def retrieve_Euribor_weights(folder_path="trained_models", filename="Euribor_weights.csv"):
    """
    Retrieves the Euribor weights DataFrame from a CSV file in the specified folder.

    Args:
        folder_path (str, optional): Path to the folder where the file is located. Defaults to "trained_models".
        filename (str, optional): Name of the CSV file. Defaults to "Euribor_weights.csv".

    Returns:
        pd.DataFrame: The retrieved Euribor weights DataFrame, or None if an error occurs.
    """
    try:
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file into a DataFrame
        euribor_weights = pd.read_csv(file_path, index_col=0)  # Set the first column as the index

        return euribor_weights

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found in folder '{folder_path}'.")
        return None
    except Exception as e:
        print(f"Error retrieving Euribor weights: {e}")
        return None


def get_Euribor_ind(Euribor_series):

    # Retrieve Trained Euribor Weights
    Euribor_weights = retrieve_Euribor_weights()

    # Get Euribor Metrics
    Euribor_metrics = get_Euribor_metrics(Euribor_series)

    #Get Euribor Indicator
    Euribor_ind = pd.DataFrame()
    for col in Euribor_weights.columns:
        Euribor_ind[col] = (Euribor_metrics * Euribor_weights[col]).sum(axis=1)

    Euribor_ind = Euribor_ind + 0.25 #0.5 & 1.4 with cash 0.05
    Euribor_ind = Euribor_ind*1.2
    #Euribor_ind=Euribor_ind/Euribor_ind.mean()
    Euribor_ind['cash'] = 1

    print('Euribor_ind',Euribor_ind)

    return Euribor_ind

def get_Euribor_metrics(Euribor):
    """
    :param Euribor: Euribor series
    :return: df: Euribor Metrics
    """

    df=pd.DataFrame()

    slow_p=255*3
    df['Euribor_mean']= Euribor.rolling(slow_p,min_periods=2*255).mean().fillna(Euribor.iloc[0])
    df['Euribor_std'] = Euribor.rolling(slow_p, min_periods=2*255).std().fillna(Euribor.std())
    df['Euribor_low_band'] = df['Euribor_mean'] - df['Euribor_std']
    df['Euribor_upper_band'] = df['Euribor_mean'] + df['Euribor_std']

    df['Euribor_is_low']=np.where(Euribor<df['Euribor_low_band'],1,0)
    df['Euribor_is_high'] = np.where(Euribor > df['Euribor_upper_band'], 1, 0)
    df['Euribor_is_mid'] = np.where((Euribor>df['Euribor_low_band']) &  ( Euribor < df['Euribor_upper_band']), 1, 0)

    fast_p=22*3*3
    df['Euribor_fast_mean'] = Euribor.rolling(fast_p).mean()
    #df['Euribor_fast_std'] = Euribor.rolling(fast_p).std()
    #df['Euribor_fast_low_band'] =df['Euribor_fast_mean']-df['Euribor_fast_std']
    #df['Euribor_fast_upper_band'] = df['Euribor_fast_mean'] + df['Euribor_fast_std']

    df['Euribor_down'] = np.where(Euribor < df['Euribor_fast_mean'], 1, 0)
    #df['Euribor_down'] = np.where(Euribor < df['Euribor_fast_low_band'], 1, 0)
    df['Euribor_up'] = np.where(Euribor > df['Euribor_fast_mean'], 1, 0)
    #df['Euribor_up'] = np.where(Euribor > df['Euribor_fast_upper_band'], 1, 0)

    df['Euribor_is_low_up']=np.where(df['Euribor_is_low']&df['Euribor_up'],1,0)
    df['Euribor_is_high_dn'] = np.where(df['Euribor_is_high'] & df['Euribor_down'], 1, 0)

    #Keep only data to be used at regresion
    regr_df = df[['Euribor_is_high', 'Euribor_is_mid', 'Euribor_is_low', 'Euribor_down', 'Euribor_up','Euribor_is_low_up','Euribor_is_high_dn']]

    return regr_df

def get_regression_coefficients(df, target_series):
    """
    Calculates the regression coefficients for each column in a DataFrame against a target Series.

    Args:
        df (pd.DataFrame): DataFrame containing predictor variables.
        target_series (pd.Series): Target variable Series.

    Returns:
        dict: A dictionary where keys are column names and values are the regression coefficients.
        Returns None if there is a type error or the series or dataframe is empty.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(target_series, pd.Series):
        return None

    if df.empty or target_series.empty:
        return None

    coefficients = {}
    for col in df.columns:
        try:
            model = LinearRegression()
            model.fit(df[[col]], target_series)
            coefficients[col] = model.coef_[0]  # Store the coefficient
        except ValueError as e:
            print(f"ValueError during linear regression with column '{col}': {e}")
            return None
        except TypeError as e:
            print(f"TypeError during linear regression with column '{col}': {e}")
            return None

    coef_series = pd.Series(coefficients)[df.columns]

    return coef_series

def get_regression_coefficients_and_correlations(df, target_series):
    """
    Calculates the regression coefficients and correlation coefficients for each
    column in a DataFrame against a target Series.

    Args:
        df (pd.DataFrame): DataFrame containing predictor variables.
        target_series (pd.Series): Target variable Series.

    Returns:
        tuple: A tuple containing two pandas Series:
               - Regression coefficients (indexed by column names).
               - Correlation coefficients (indexed by column names).
        Returns None if there is a type error or the series or dataframe is empty.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(target_series, pd.Series):
        return None

    if df.empty or target_series.empty:
        return None

    coefficients = {}
    correlations = {}
    for col in df.columns:
        try:
            model = LinearRegression()
            model.fit(df[[col]], target_series)
            coefficients[col] = model.coef_[0]
            correlations[col] = df[col].corr(target_series)
        except (ValueError, TypeError) as e:
            print(f"Error during regression/correlation with column '{col}': {e}")
            return None

    coef_series = pd.Series(coefficients)[df.columns]
    corr_series = pd.Series(correlations)[df.columns]

    return coef_series, corr_series

def get_correlations(df, target_series):
    """
    Calculates the regression coefficients and correlation coefficients for each
    column in a DataFrame against a target Series.

    Args:
        df (pd.DataFrame): DataFrame containing predictor variables.
        target_series (pd.Series): Target variable Series.

    Returns:
        tuple: A tuple containing two pandas Series:
               - Regression coefficients (indexed by column names).
               - Correlation coefficients (indexed by column names).
        Returns None if there is a type error or the series or dataframe is empty.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(target_series, pd.Series):
        return None

    if df.empty or target_series.empty:
        return None

    coefficients = {}
    correlations = {}
    for col in df.columns:
        try:
            model = LinearRegression()
            model.fit(df[[col]], target_series)
            coefficients[col] = model.coef_[0]
            correlations[col] = df[col].corr(target_series)
        except (ValueError, TypeError) as e:
            print(f"Error during regression/correlation with column '{col}': {e}")
            return None

    coef_series = pd.Series(coefficients)[df.columns]
    corr_series = pd.Series(correlations)[df.columns]

    return coef_series, corr_series

def calculate_all_correlations_matrix(df1, df2):
    """
    Calculates the correlation between each column of df1 with all columns of df2
    using matrix operations.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the correlation coefficients.
        Returns None if there is a type error or the dataframe is empty.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        return None

    if df1.empty or df2.empty:
        return None

    # Convert DataFrames to numpy arrays for efficient computation
    array1 = df1.to_numpy()
    array2 = df2.to_numpy()

    # Center the arrays by subtracting the mean of each column
    array1_centered = array1 - array1.mean(axis=0)
    array2_centered = array2 - array2.mean(axis=0)

    # Calculate the covariance matrix
    covariance_matrix = np.dot(array1_centered.T, array2_centered) / (len(df1) - 1)

    # Calculate the standard deviations of each column
    std1 = df1.std().to_numpy()
    std2 = df2.std().to_numpy()

    # Calculate the correlation matrix
    correlation_matrix = covariance_matrix / np.outer(std1, std2)

    return pd.DataFrame(correlation_matrix, index=df1.columns, columns=df2.columns)


if __name__ == '__main__':
    compute(settings)