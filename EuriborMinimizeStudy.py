# import libraries and functions
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import time

from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize

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

    filename="Euribor_minimize_weights.csv"

    if True:
        #Training & Saving Euribor_weigths
        get_trained_Euribor_weigths(tickers_returns,filename=filename)

    if True:
        #Retrieve Training model and get Euribor Ind
        Euribor_series = tickers_returns['cash'] * 255 * 100
        Euribor_ind = get_Euribor_ind_minimize(Euribor_series, filename=filename)

        strategy_returns=tickers_returns*Euribor_ind
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


        for ticker in settings['tickers']+['cash']:
            plot_df=pd.DataFrame()
            plot_df[ticker]=cum_ret[ticker]
            plot_df['Euribor'] = Euribor_series
            plot_df['strategy_cum_returns']=strategy_cum_ret[ticker]
            plot_df['Euribor_ind']=Euribor_ind[ticker]

            plot_df.plot(title=ticker)

    plt.show()


    return

def calculate_cagr(returns):
    """Calculates the Compound Annual Growth Rate."""
    return returns.mean()*252

def calculate_volatility(returns):
    """Calculates the annualized volatility."""
    return returns.std() * np.sqrt(252)  # Annualized volatility

def objective_function_OK(weights, returns, euribor_metrics):
    """Calculates the objective function (CAGR - Volatility) using combined returns."""

    # Reshape weights to a matrix
    num_euribor = euribor_metrics.shape[1]
    num_returns = returns.shape[1]
    weights_matrix = weights.reshape((num_euribor, num_returns))

    # Create weight matrix DataFrame
    weights_df = pd.DataFrame(weights_matrix, index=euribor_metrics.columns, columns=returns.columns)

    # Calculate weighted returns
    weighted_returns = euribor_metrics.dot(weights_df)

    # Calculate portfolio returns
    portfolio_returns = (returns * weighted_returns).sum(axis=1)

    cagr = calculate_cagr(portfolio_returns)
    volatility = calculate_volatility(portfolio_returns)
    opt_fun= -(cagr - volatility)
    return opt_fun  # Minimize the negative to maximize

def objective_function(weights, returns, euribor_metrics):
    """Calculates the objective function with a penalty for low CAGR."""

    # Reshape weights to a matrix
    num_euribor = euribor_metrics.shape[1]
    num_returns = returns.shape[1]
    weights_matrix = weights.reshape((num_euribor, num_returns))

    # Create weight matrix DataFrame
    weights_df = pd.DataFrame(weights_matrix, index=euribor_metrics.columns, columns=returns.columns)

    # Calculate weighted returns
    weighted_returns = euribor_metrics.dot(weights_df)

    # Calculate portfolio returns
    portfolio_returns = (returns * weighted_returns).sum(axis=1)

    cagr = calculate_cagr(portfolio_returns)
    volatility = calculate_volatility(portfolio_returns)

    # Penalty for low CAGR
    min_cagr = 0.03
    low_cagr_penalty= np.where(cagr<0.001,1,np.where(cagr<min_cagr,min_cagr/cagr-1,0))

    # Penalty for low Volatility
    min_volat=0.03
    low_volat_penalty = np.where(volatility < 0.001, 1, np.where(volatility < min_volat, min_volat / volatility - 1, 0))

    penalty= low_cagr_penalty + low_volat_penalty

    return -(cagr - volatility - penalty)  # Minimize the negative to maximize

def optimize_weights(returns, euribor_metrics):
    """Optimizes weights to maximize CAGR - Volatility."""
    num_euribor = euribor_metrics.shape[1]
    num_returns = returns.shape[1]
    num_weights = num_euribor * num_returns  # Total number of weights

    initial_weights = np.array([1 / num_weights] * num_weights)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 2})
    bounds = tuple((0, 1) for _ in range(num_weights))

    result = minimize(objective_function, initial_weights, args=(returns, euribor_metrics),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    # Reshape optimized weights to matrix
    weights_matrix = result.x.reshape((num_euribor, num_returns))
    weights_df = pd.DataFrame(weights_matrix, index=euribor_metrics.columns, columns=returns.columns)

    # Set near-zero weights to zero
    tolerance = 1e-6
    weights_df[abs(weights_df) < tolerance] = 0

    return weights_df

def get_trained_Euribor_weigths(tickers_returns,folder_path="trained_models", filename="Euribor_weights.csv"):

    Euribor_series = tickers_returns['cash'] * 255 * 100

    #Get Euribor Metrics
    Euribor_metrics=get_Euribor_metrics(Euribor_series)

    print('Euribor_metrics',Euribor_metrics)

    #Create Euribor_weights
    Euribor_weights = optimize_weights(tickers_returns,Euribor_metrics)

    #Rescale Up
    Euribor_weights = Euribor_weights*10

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


def get_Euribor_ind_minimize(Euribor_series, filename="Euribor_minimize_weights.csv"):

    # Retrieve Trained Euribor Weights
    Euribor_weights = retrieve_Euribor_weights(filename=filename)

    # Get Euribor Metrics
    Euribor_metrics = get_Euribor_metrics(Euribor_series)

    #Get Euribor Indicator
    Euribor_ind = pd.DataFrame()
    for col in Euribor_weights.columns:
        Euribor_ind[col] = (Euribor_metrics * Euribor_weights[col]).sum(axis=1)

    #Keep yesterday value
    Euribor_ind = Euribor_ind.shift(1)

    #Power factor
    Euribor_ind = Euribor_ind**(1/2)

    #Add factor to all values
    Euribor_ind = Euribor_ind + 0.4  # 0.4 #0.25 & 1.5 with cash 0.5

    #Set Cash Mean to one
    Euribor_ind['cash'] = Euribor_ind['cash']/Euribor_ind['cash'].mean()


    #Keep only positive values
    Euribor_ind = Euribor_ind.clip(lower=0)

    #Avoid peack short signals
    Euribor_ind = Euribor_ind.rolling(3).mean()

    #Set Indicator for Cash
    Euribor_ind['cash'] = np.where(Euribor_series>0.005,1.0,0)


    #print('Euribor_ind',Euribor_ind)

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


if __name__ == '__main__':
    compute(settings)