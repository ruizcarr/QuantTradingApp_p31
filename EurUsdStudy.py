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
    tickers_closes=data.tickers_closes
    data_dict=data.data_dict

    print('tickers_returns',tickers_returns)

    fed_df=indicators_dict["FED_rates"]

    print("fed_df",fed_df)

    df=tickers_closes[['EURUSD=X']].copy()
    df["EurUsd_mean"] = df["EURUSD=X"].rolling(255).mean()
    df["EurUsd_fast_mean"] = df["EURUSD=X"].rolling(22).mean()

    #df["EurUsd_chg"]=df["EurUsd_mean"].pct_change().rolling(255).mean()*255

    df["EurUsd_mean_diff"]=df["EurUsd_fast_mean"]-df["EurUsd_mean"]


    df["EurUsd_mean_diff_std"] = df["EurUsd_mean_diff"].rolling(255).std()*0.5
    treshold = 0.02
    treshold = df["EurUsd_mean_diff_std"]
    #df["EurUsd_up"]=df["EurUsd_chg"]> treshold
    #df["EurUsd_dn"] = df["EurUsd_chg"] < -treshold
    #df["EurUsd_ind"] = np.where(df["EurUsd_up"],2,np.where(df["EurUsd_dn"],0,1))
    #df["EurUsd_ind"] = np.where(df["EurUsd_fast_mean"] >df["EurUsd_mean"],2,0)
    df["EurUsd_up"] = df["EurUsd_mean_diff"] > treshold
    df["EurUsd_dn"] = df["EurUsd_mean_diff"] < -treshold
    df["EurUsd_ind"] = np.where(df["EurUsd_up"], 1, np.where(df["EurUsd_dn"], 0, 0.0))
    df["EurUsd_ret"] = tickers_closes['EURUSD=X'].iloc[0]*(1+tickers_returns['EURUSD=X']*df["EurUsd_ind"]).cumprod()

    #df["Fed"]=fed_df["FED"]*10
    #df["Euribor"] = tickers_returns['cash'] * 255*10
    #df["Fed_Euribor"]=df["Fed"]-df["Euribor"]


    df.plot()




    plt.show()


    return




if __name__ == '__main__':
    compute(settings)