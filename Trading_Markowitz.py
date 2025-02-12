# import libraries and functions
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import time

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
from Training_Markowitz import process_log_data

# Import Trading Settings
from config.trading_settings import settings

def run(settings):

    start_time = time.time()

    #Main Code
    verbose=settings['verbose']

    # Get Data & Indicators
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns = data.tickers_returns
    data_dict=data.data_dict


    # Get Trained Optimized Parameters from csv File
    wft = WalkForwardTraining(data_ind, settings)
    params_train = get_params_from_csv(settings['path_train']+'params_train.csv',
        wft.tt_windows, settings)

    #Trading/Test: Apply Params Train to Test
    wft.Test(indicators_dict,params_train,do_annalytics=False)
    positions=wft.test_positions

    #Apply Exposition Constraints
    #Exponential factor,Mult factor & Limit maximum/minimum individual position
    if settings['apply_pos_consraints']:
        from Training_Markowitz import apply_pos_constrain
        positions = apply_pos_constrain(positions,settings )

    #Cash BackTest with Backtrader
    if settings['do_BT'] :
        if verbose: print('\nCash BackTest with Backtrader ')
        _, log_history = compute_backtest_vectorized(positions, settings, data_dict)

        #Get End Of Day Values
        settings['tickers']=list(positions.columns)
        eod_log_history, trading_history= process_log_data(log_history,settings)


        if verbose:
            print("tickers_closes\n", data.tickers_closes[:-5].tail(10))

            print("tickers_returns\n", tickers_returns[:-5].tail(10))

            print("positions\n", positions.tail(15))

            print("log_history\n", log_history.tail(30))

            print("eod_log_history\n", eod_log_history.tail(15))

            #print("trading_history\n", trading_history.tail(20))

    end_time = time.time()
    if verbose:
        print('\nTrading timeTaken:', end_time - start_time)

        plot_len=250
        positions.tail(plot_len).plot(title='Positions')
        eod_log_history[settings['tickers']].tail(plot_len).plot(title='Contracts')

        plot_df = positions.copy()
        plot_df['sum'] = plot_df.sum(axis=1)
        plot_df['cum_ret'] = (1 + eod_log_history['portfolio_value_eur'].pct_change()).cumprod() - 1
        plot_df.plot(title='positions')
        if settings['do_BT']:
            eod_log_history[positions.columns].plot(title='n_contracts')

    if settings['verbose']:
        plt.show()

    return log_history,positions, data

def get_orders_log(log_history):
    def print_orders_log(df, title):
        if len(df) > 0:
            print(f"{title} {df['date'].iloc[0]} 00:00(CET)")
            for i, row in df.iterrows():
                order_log = f"{row['ticker']} {row['exectype']} {row['B_S']}  {row['size']}"
                if row['exectype'] == "Stop":
                    order_log = order_log + f" @ {row['price']}"
                print(order_log)
        else:
            print(f"No {title}")
    # Get Today SELL Stops Log
    orders_history = log_history[log_history['event'].str.contains('Order Created')]  # [['date','event','ticker','size','price']]
    today = date.today()
    today_orders = orders_history.loc[orders_history['date'] == today]

    print_orders_log(today_orders, 'Today Orders')

    # Get Next days SELL Stops Log
    orders_ahead = orders_history.loc[orders_history['date'] > today]
    if len(orders_ahead) > 0:
        next_day = orders_ahead['date'].iloc[0]
        next_orders = orders_history.loc[orders_history['date'] == next_day]

        print_orders_log(next_orders, 'Next Orders Forecast')

    else:
        print("No Orders Forecast  in the next days")

def process_log_data_duplicated(log_history,settings):

    # Overwrite Drawdown YTD EUR
    max_portfolio_value_eur = log_history['portfolio_value_eur'].rolling(252, min_periods=5).max()
    log_history['ddn_eur'] = round(1 - max_portfolio_value_eur / log_history['portfolio_value_eur'], 3)

    #End of day Portfolio data
    eod_log_history=log_history.drop_duplicates(subset='date', keep='last').set_index('date')[settings["tickers"]+["portfolio_value","portfolio_value_eur","pos_value","ddn","exchange_rate" , "dayly_profit","dayly_profit_eur"]]

    # Add Drawdown YTD EUR
    max_portfolio_value_eur = eod_log_history['portfolio_value_eur'].rolling(252, min_periods=5).max()
    eod_log_history['ddn_eur'] = round(1 - max_portfolio_value_eur / eod_log_history['portfolio_value_eur'], 3)


    #Filter days where trading
    eod_log_history['keep_day']=(eod_log_history[settings["tickers"]] == eod_log_history[settings["tickers"]].shift(1)).all(axis=1).astype(int)
    trading_history = eod_log_history[eod_log_history['keep_day']!=1]

    #Add some annalytics
    eod_log_history["portfolio_return"]=eod_log_history["portfolio_value_eur"].pct_change()
    eod_log_history["cagr"]=eod_log_history["portfolio_return"].rolling(252).sum()
    eod_log_history["weekly_return"]=eod_log_history["portfolio_return"].rolling(5).sum()
    eod_log_history["monthly_return"]=eod_log_history["portfolio_return"].rolling(22).sum()

    return eod_log_history,trading_history


if __name__ == '__main__':
    run(settings)