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

    print('tickers_returns', tickers_returns)


    cum_ret=(1+tickers_returns).cumprod()

    #Apply SMA 1y Strategy to study only bullish
    cum_ret_mean=cum_ret.rolling(255).mean()
    sma_weights=tickers_returns/tickers_returns #Initialize weights df with ones
    sma_weights[cum_ret.shift(1)<cum_ret_mean.shift(1)]=0 #Fill zero value where price belllow mean
    sma_ret=tickers_returns*sma_weights #SMA Strategy returns
    cum_ret_sma = (1 + sma_ret).cumprod()


    #Drawdown
    ddn_df = calculate_drawdown(cum_ret)
    sma_ddn_df = calculate_drawdown(cum_ret_sma)

    print('ddn_df',ddn_df.describe() )
    print('sma_ddn_df', sma_ddn_df.describe())

    sma_ddn_df.plot()

    ticker="ES=F"
    plolt_df = cum_ret[[ticker]]
    plolt_df["SMA"] =cum_ret_mean[ticker]
    plolt_df["SMA_Strategy"] = cum_ret_sma[ticker]
    plolt_df["ddn"] = ddn_df[ticker] * 10
    plolt_df["SMA_ddn"] = sma_ddn_df[ticker]*10

    all_rebounds= analyze_multi_asset_drawdown_rebounds_risk_reward(sma_ddn_df, sma_ret)

    print('all_rebounds',all_rebounds)

    all_rebounds['ES=F'].plot()



    if False:

        final_weights = generate_multi_asset_weights_ddn_ttkeep_daily_optimized(ddn_df, sma_ret)

        print(final_weights)

        final_weights[["ES=F"]].plot()


        strategy_returns=sma_ret*final_weights.shift(1)
        strategy_cumreturns=(1+strategy_returns).cumprod()

        strategy_cumreturns.plot()


        plolt_df["ES_Strategy"]=strategy_cumreturns["ES=F"]


    plolt_df.plot(title=ticker)

    plt.show()


    return

def calculate_drawdown(close_df):
    """Calculates drawdown."""
    cumulative_max = close_df.cummax()
    drawdown = (close_df - cumulative_max) / cumulative_max
    return drawdown.fillna(0)

def analyze_multi_asset_drawdown_rebounds(ddn_df, rebound_window=22):
    """Analyzes drawdown rebounds for multiple assets using rolling windows."""
    all_rebounds = {}
    for ticker in ddn_df.columns:
        future_min_drawdowns = ddn_df[ticker].rolling(window=rebound_window, min_periods=1).min().shift(-rebound_window + 1)
        recovery = future_min_drawdowns - ddn_df[ticker]
        rebounds_df = pd.DataFrame({'drawdown': ddn_df[ticker], 'recovery': recovery})
        rebounds_df['rebound'] = (rebounds_df['recovery'] < 0).astype(int)
        rebounds_df = rebounds_df.iloc[:-rebound_window + 1]
        all_rebounds[ticker] = rebounds_df
    return all_rebounds

def analyze_multi_asset_drawdown_rebounds_risk_reward(ddn_df, returns_df, rebound_window=22):
    """Analyzes drawdown rebounds using risk and reward within the rebound window."""
    all_rebounds = {}
    for ticker in ddn_df.columns:
        # Calculate cumulative returns within the window
        cumulative_returns = returns_df[ticker].rolling(window=rebound_window, min_periods=1).sum().shift(-rebound_window + 1)

        # Calculate additional drop within the window
        future_min_drawdowns = ddn_df[ticker].rolling(window=rebound_window, min_periods=1).min().shift(-rebound_window + 1)
        additional_drop = future_min_drawdowns - ddn_df[ticker]

        # Calculate rebound function (returns - risk)
        rebound_function = cumulative_returns + additional_drop

        # Create DataFrame
        rebounds_df = pd.DataFrame({'drawdown': ddn_df[ticker], 'cumulative_returns': cumulative_returns, 'additional_drop': additional_drop, 'rebound_function': rebound_function})

        # Remove trailing rows
        rebounds_df = rebounds_df.iloc[:-rebound_window + 1]

        all_rebounds[ticker] = rebounds_df

    return all_rebounds

def analyze_multi_asset_time_to_keep(weights_df, weight_threshold=0.8):
    """Analyzes time to keep positions for multiple assets."""
    ttkeep_data = {}
    for ticker in weights_df.columns:
        ttkeep_list = []
        holding = False
        start_date = None
        for date, weight in weights_df[ticker].items():
            if weight >= weight_threshold and not holding:
                holding = True
                start_date = date
            elif weight < weight_threshold and holding:
                holding = False
                ttkeep = (date - start_date).days
                ttkeep_list.append(ttkeep)
        ttkeep_data[ticker] = pd.Series(ttkeep_list)
    return ttkeep_data

def generate_multi_asset_weights_ddn_ttkeep_daily_optimized_OK(ddn_df, portfolio_returns, weight_threshold=0.7):
    """Generates multi-asset weights with DDN and TTKEEP analysis for daily rebalancing (optimized)."""
    daily_returns = portfolio_returns
    daily_weights = pd.DataFrame(1.0, index=daily_returns.index, columns=daily_returns.columns)

    all_rebounds_analysis = analyze_multi_asset_drawdown_rebounds(ddn_df, 21)
    initial_weights = pd.DataFrame(1.0 / len(daily_returns.columns), index=daily_returns.index, columns=daily_returns.columns)
    ttkeep_analysis = analyze_multi_asset_time_to_keep(initial_weights, weight_threshold)

    for ticker in daily_returns.columns:
        current_drawdown = ddn_df[ticker]
        rebounds_analysis_ticker = all_rebounds_analysis[ticker]
        rebounds_analysis_ticker['drawdown_bin'] = pd.cut(rebounds_analysis_ticker['drawdown'], bins=10)
        rebound_probabilities = rebounds_analysis_ticker.groupby('drawdown_bin')['rebound'].mean()

        # Vectorized drawdown adjustment
        adjustment = pd.Series(1.0, index=daily_returns.index)
        for bin_range, prob in rebound_probabilities.items():
            # Extract interval bounds
            lower_bound = bin_range.left
            upper_bound = bin_range.right
            mask = (current_drawdown >= lower_bound) & (current_drawdown <= upper_bound) & (prob > 0.6)
            adjustment[mask] = 1.2

        daily_weights[ticker] *= adjustment

        # Vectorized TTKEEP adjustment
        if len(ttkeep_analysis[ticker]) > 0:
            holding_time = (daily_weights[ticker].cumsum() >= weight_threshold).astype(int).cumsum()
            holding_time_diff = holding_time.diff().fillna(0)
            holding_time_reset = holding_time_diff[holding_time_diff < 0].index
            holding_time = holding_time - holding_time.iloc[0]

            for reset_index in holding_time_reset:
                holding_time.loc[reset_index:] = holding_time.loc[reset_index:] - holding_time.loc[reset_index]

            mask = (holding_time.values < ttkeep_analysis[ticker].mean()*0.8) & (daily_weights[ticker] < weight_threshold)
            daily_weights.loc[mask, ticker] = weight_threshold

        # Normalize weights
        daily_weights = daily_weights.div(daily_weights.mean(axis=1), axis=0)

    return daily_weights

def generate_multi_asset_weights_ddn_ttkeep_daily_optimized_OK2(ddn_df, portfolio_returns, weight_threshold=1.01):
    """Generates multi-asset weights with DDN and TTKEEP analysis for daily rebalancing (optimized)."""
    daily_returns = portfolio_returns
    daily_weights = pd.DataFrame(1.0, index=daily_returns.index, columns=daily_returns.columns)
    all_rebounds_analysis = analyze_multi_asset_drawdown_rebounds(ddn_df, 5)

    print('all_rebounds_analysis',all_rebounds_analysis)

    for ticker in daily_returns.columns:
        current_drawdown = ddn_df[ticker]
        rebounds_analysis_ticker = all_rebounds_analysis[ticker]
        rebounds_analysis_ticker['drawdown_bin'] = pd.cut(rebounds_analysis_ticker['drawdown'], bins=10)
        rebound_probabilities = rebounds_analysis_ticker.groupby('drawdown_bin')['rebound'].mean()
        print(ticker,'rebound_probabilities', rebound_probabilities)

        # Vectorized drawdown adjustment
        adjustment = pd.Series(0, index=daily_returns.index)
        for bin_range, prob in rebound_probabilities.items():
            lower_bound = bin_range.left
            upper_bound = bin_range.right
            mask = (current_drawdown >= lower_bound) & (current_drawdown <= upper_bound) #& (prob > 0.5)
            adjustment[mask] = prob #1.0

        daily_weights[ticker] *= adjustment

    daily_weights.plot()

    # Normalize weights
    daily_weights = daily_weights/daily_weights.mean()

    daily_weights.plot()

    # Update TTKEEP analysis after each day's weight calculation
    ttkeep_analysis = analyze_multi_asset_time_to_keep(daily_weights, weight_threshold)

    for ticker in daily_returns.columns:
        # Vectorized TTKEEP adjustment
        if len(ttkeep_analysis[ticker]) > 0:
            holding_time = (daily_weights[ticker].cumsum() >= weight_threshold).astype(int).cumsum()
            holding_time_diff = holding_time.diff().fillna(0)
            holding_time_reset = holding_time_diff[holding_time_diff < 0].index
            holding_time = holding_time - holding_time.iloc[0]

            for reset_index in holding_time_reset:
                holding_time.loc[reset_index:] = holding_time.loc[reset_index:] - holding_time.loc[reset_index]

            mask = (holding_time.values < ttkeep_analysis[ticker].mean()) & (daily_weights[ticker] < weight_threshold)
            daily_weights.loc[mask, ticker] = weight_threshold

    # Normalize weights
    daily_weights = daily_weights.div(daily_weights.mean(axis=1), axis=0)
    daily_weights = (daily_weights-daily_weights.min())/(daily_weights.max()-daily_weights.min())*1.5 #+0.5


    return daily_weights

def apply_ttkeep_adjustment_fixed_window_vectorized(daily_weights, holding_window=5):
    """Applies TTKEEP adjustment to enforce a fixed holding window (vectorized)."""
    shifted_weights = [daily_weights.shift(i) for i in range(holding_window)]
    shifted_weights = pd.concat(shifted_weights, axis=1)
    shifted_weights = shifted_weights.iloc[holding_window - 1:]

    is_holding = (shifted_weights.iloc[:, :holding_window].apply(lambda row: len(set(row)) == 1, axis=1))

    adjustment_mask = ~is_holding
    adjustment_mask = adjustment_mask.reindex(daily_weights.index, fill_value=False)
    adjustment_mask.iloc[:holding_window] = False

    adjusted_weights = daily_weights.copy() # Create a copy to avoid modifying the original during the loop

    # Create a shifted DataFrame for previous day's weights
    previous_day_weights = daily_weights.shift(1)

    # Exclude the first row from the adjustment mask
    adjustment_mask.iloc[0] = False

    # Use .where() to apply the adjustment conditionally
    adjusted_weights = adjusted_weights.where(~adjustment_mask, previous_day_weights.reindex(daily_weights.index))

    return adjusted_weights

def generate_multi_asset_weights_ddn_ttkeep_daily_optimized(ddn_df, portfolio_returns, weight_threshold=1.01):
    """Generates multi-asset weights with DDN and TTKEEP analysis for daily rebalancing (optimized)."""
    daily_returns = portfolio_returns
    daily_weights = pd.DataFrame(1.0, index=daily_returns.index, columns=daily_returns.columns)
    rebound_window=5
    all_rebounds_analysis = analyze_multi_asset_drawdown_rebounds(ddn_df, rebound_window)

    print('all_rebounds_analysis',all_rebounds_analysis)

    for ticker in daily_returns.columns:
        current_drawdown = ddn_df[ticker]
        rebounds_analysis_ticker = all_rebounds_analysis[ticker]
        rebounds_analysis_ticker['drawdown_bin'] = pd.cut(rebounds_analysis_ticker['drawdown'], bins=10)
        rebound_probabilities = rebounds_analysis_ticker.groupby('drawdown_bin')['rebound'].mean()
        print(ticker,'rebound_probabilities', rebound_probabilities)

        # Vectorized drawdown adjustment
        adjustment = pd.Series(0, index=daily_returns.index)
        for bin_range, prob in rebound_probabilities.items():
            lower_bound = bin_range.left
            upper_bound = bin_range.right
            mask = (current_drawdown >= lower_bound) & (current_drawdown <= upper_bound)
            adjustment[mask] = prob

        daily_weights[ticker] *= adjustment

    daily_weights.plot()

    # Normalize weights
    daily_weights = daily_weights/daily_weights.mean()

    daily_weights.plot()

    #daily_weights = apply_ttkeep_adjustment_fixed_window_vectorized(daily_weights, rebound_window)

    #daily_weights = daily_weights / daily_weights.mean()

    #Final Normalization
    daily_weights = daily_weights*1.25

    return daily_weights

if __name__ == '__main__':
    compute(settings)