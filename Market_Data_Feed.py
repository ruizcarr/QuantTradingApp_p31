# import libraries and functions
import numpy as np
import pandas as pd
import pandas_ta as ta
import os.path

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import date
from datetime import timedelta
import yfinance as yf
import itertools
import quantstats as qs
# extend pandas functionality with mettickers, etc.
qs.extend_pandas()
from arch import arch_model
from sklearn.metrics import r2_score

from Euribor_Download import get_euribor_1y_daily

from utils import sigmoid

import webbrowser


# Wider print limits
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')

class Data_Ind_Feed:
    def __init__(self,settings):


        # Get Data Instance
        data=Data(settings,settings['tickers'],settings['start'],settings['end'],settings['add_days'],settings['offline'])

        #Get Indicators Instance
        ind=Indicators(data.tickers_returns,settings)

        #Get Data , Indicators Dict tuple
        self.data_ind=(data,ind.indicators_dict)


class Data:
    def __init__(self,settings, tickers=['ES=F'], start='2003-12-01', end=(date.today() + timedelta(days=1)).isoformat(),add_days=0,offline=False):

        #self.yf_data(tickers, start, end,add_days)

        #print(self.data_bundle)

        self.path = "datasets/" #"datasets\\"
        self.db_file = self.path + 'data_bundle.csv'

        if not offline:

            #Get Data Bundle from yahoo finance
            self.yf_data_bundle(tickers, start, end, add_days)

        elif offline & os.path.isfile(self.db_file):

            # Read data_bundle from csv
            data = pd.read_csv(self.db_file, header=[0,1], index_col=0)
            data.index = pd.to_datetime(data.index)
            #data.sort_index(inplace=True)
            #Keep only data from start on settings
            self.data_bundle=data[start:]

        else:
            print('Off-Line and not File with Saved Data available')



        #Add Next Days for Trading
        if add_days>0:
            self.add_next_days(add_days)

        # Get Close Prices from data_bundle in the order of tickers
        #self.tickers_closes = pd.DataFrame(data=np.asarray([self.data_bundle[tic, 'Close'] for tic in tickers]).T, columns=tickers, index=self.data_bundle.index)
        #self.tickers_closes = pd.concat([self.data_bundle[tic, 'Close'] for tic in tickers], axis=1)

        # Save data to dictionary for further use
        self.data_dict = {
            tick: self.data_bundle[tick]
            for tick in settings['tickers']
            if tick in self.data_bundle.columns  # Check if ticker exists!
        }


        if start < self.data_bundle.index[0].isoformat():  # if start requested is older than available data
            self.extended_data(self.data_dict, start)

        #Get tickers_closes from dict
        closes={}
        for ticker, df in self.data_dict.items():
            closes[ticker] = df['Close']
        self.tickers_closes =pd.DataFrame(closes)

        #Add Cash
        if settings['add_cash']:
            #self.tickers_closes['cash']=get_cash_values(self.tickers_closes.index, settings['cash_rate'], cash_init=1000)
            euribor_df=get_euribor_1y_daily().reindex(self.tickers_closes.index, method="ffill")
            self.tickers_closes['cash'] =1000*(1+euribor_df['Euribor']/255).cumprod()

            df=list(self.data_dict.values())[0].copy()
            for col in df.columns:
                 df[col] = self.tickers_closes['cash']
            self.data_dict['cash']= df

        self.tickers_returns = self.tickers_closes.pct_change().fillna(0)

        # Replace returns values of futures at expiration_dates by cash returns where available
        # Create Quarterly Trading Calendar and save for further use
        #self.q_calendar = get_es_trading_calendar(self.tickers_returns, expiration_freq='Q')
        #fut_cash_tickers_dict={'ES=F': '^GSPC', 'NQ=F': '^NDX'}
        #self.tickers_returns = replace_fut_by_cash_returns_at_q_exp_or_after_dates(self.tickers_returns, fut_cash_tickers_dict,self.q_calendar,offline)

        # Apply Contango at ticker returns
        #contango = [1 - contangos[ticker] / 100 / 252 if ticker in contangos else 1 for ticker in tickers]
        #self.tickers_returns = self.tickers_returns.multiply(contango, axis=1)

        # Sanitize Open, High & Low
        self.data_dict_sanitize_OHL(self.data_dict)

        # Get Tickers Returns in EUR

        # Get historical of Exchange Rate EUR/USD (day after)
        if "EURUSD=X" in self.tickers_closes.columns:
            exchange_rate = 1 / self.tickers_closes["EURUSD=X"].shift(1).fillna(method='bfill')
        else:
            # Get EURUSD=X
            print("No EURUSD=X available")

        self.exchange_rate = exchange_rate

        # Get Tickers Returns in EUR
        tickers_closes_eur = self.tickers_closes.multiply(self.exchange_rate, axis='index')
        self.tickers_returns_eur = tickers_closes_eur.pct_change().fillna(0)



    def yf_data_bundle(self, tickers, start, end,add_days=0):
        """"Get Closes & Returns from yahoo finance
                Save OHLC to tickers dict"""

        # Get Data Bundle from yf
        tickers_space_sep = " ".join(tickers)
        data_bundle = yf.download(tickers_space_sep, start, end, group_by='ticker', progress=False).dropna()


        # Convert the index to naive timestamps (no timestamps)
        data_bundle.index = data_bundle.index.tz_localize(None)

        # Drop duplicated in calse
        data_bundle.drop_duplicates()

        # Datetime Index
        data_bundle.index = pd.to_datetime(data_bundle.index)

        # Retrive data_bundle from csv and update if exist
        if os.path.isfile(self.db_file):
            data = pd.read_csv(self.db_file, header=[0,1], index_col=0)
            data.index = pd.to_datetime(data.index)

        #Update csv file data
        updated_data_bundle= pd.concat([data, data_bundle])
        updated_data_bundle = updated_data_bundle.loc[~updated_data_bundle.index.duplicated(keep='first')]
        updated_data_bundle = updated_data_bundle.sort_index()

        #Save to csv
        updated_data_bundle.to_csv(self.db_file)

        #Save for further use
        self.data_bundle = data_bundle


    def extended_data(self, data_dict, start):
        """Extend with  Historical Data if requested """
        aka_dict = {'ES=F': '^GSPC', 'NQ=F': '^NDX', 'GC=F': 'GOLD', 'EURUSD=X': 'EURUSD', 'CL=F': 'OIL'}
        tickers = list(data_dict.keys())

        # Check if all Historical data files are available
        h_tickers=[]
        for tick in tickers:
            if tick in aka_dict.keys():
                h_tickers.append(aka_dict[tick])
            else:
                if tick in aka_dict.values():
                    h_tickers.append(tick)
                else:
                    print(tick, 'has not Historical data file available !!!')

        #h_tickers = [aka_dict[tick] for tick in tickers]

        # Get all tickers historical csv data in a dict
        self.data_from_csv(h_tickers)
        h_data_dict = self.data_dict_csv

        # Start from where data is available
        date_0 = max([min(df.index) for df in data_dict.values()])

        # Start date from where data is available for all historical data h_data or Start
        h_date_0 = max([max([min(df.index) for df in h_data_dict.values()]).isoformat(), start])

        e_data_dict = {}
        e_closes = pd.DataFrame()

        for i, tick in enumerate(tickers):
            h_tick = h_tickers[i]
            h_data_tick = h_data_dict[h_tick]
            data_tick = data_dict[tick]

            # Repair data when new csv available
            # self.repair_data(h_data_tick,h_tick)

            # Slice historical data
            h_data_tick = h_data_tick.loc[(h_data_tick.index >= h_date_0) & (h_data_tick.index < date_0)]

            # Reindex as Bechmark
            if i == 0:
                idx_0 = h_data_tick.index
            else:
                h_data_tick = h_data_tick.reindex(idx_0).fillna(method='ffill')

            # Get Extended Data: Concatenate Historical data with current data
            e_data_tick = pd.concat([h_data_tick, data_tick])

            # Save data in dict
            e_data_dict[tick] = e_data_tick

            # Get Closes
            e_closes = pd.concat([e_closes, e_data_tick.Close], axis=1)

        e_closes.columns = tickers

        e_closes.index = pd.to_datetime(e_closes.index)

        self.data_dict = e_data_dict
        self.tickers_closes = e_closes

    def repair_data(self, data, tick):

        #Add missing columns
        col=data.columns
        print(tick, col)
        yf_col=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_col=[c for c in yf_col if c not in col]
        print('missing_col',missing_col)
        data[missing_col]=np.nan


        # Check for duplicates
        duplicated = data[data.index.duplicated(keep=False)]
        print('duplicated', duplicated)
        if len(duplicated) == 0:
            data = data[~data.index.duplicated(keep='first')]

        # Check for NaN at Close
        nans_close = data[data.Close.isnull().values]
        print('nans_close', len(nans_close))
        if len(nans_close) > 1:
            # Close nan Fill Close nan with Open day after
            data.Close = data.Close.fillna(data.Open.shift(-1)).fillna(method='ffill')

        # Check for NaN at Open
        nans_open = data[data.Open.isnull().values]
        print('nans_open', len(nans_open))
        if len(nans_open) > 1:
            # Fill Open nan with Close day before
            data.Open = data.Open.fillna(data.Close.shift(1)).fillna(method='ffill')

        #Fill nan at [ 'High', 'Low', 'Adj Close'] with close
        data.High=data.High.fillna(data.Close)
        data.Low = data.Low.fillna(data.Close)
        data['Adj Close']= data['Adj Close'].fillna(data.Close)

        data.Volume=data.Volume.fillna(0)

        #Reorder columns as yf style
        data=data[yf_col]

        # Check for NaN
        nans_close = data[data.Close.isnull().values]
        nans_open = data[data.Open.isnull().values]
        print('nans_close', len(nans_close), 'nans_open', len(nans_open))

        # Save repaired data to csv
        data.to_csv(tick + '.csv')

    def data_bundle_to_tick_csv(self, data_bundle):
        """Tickers must be at  level 0
            Save individual OHLC to tick.csv file"""
        tickers = list(data_bundle.columns.levels[0])
        for tick in tickers: data_bundle[tick].to_csv(self.path+tick + '.csv')

    def data_from_csv(self, tickers):
        closes_csv = pd.DataFrame()
        data_dict_csv = {}
        for tick in tickers:
            if os.path.isfile(self.path+tick + '.csv'):
                data = pd.read_csv(self.path+tick + '.csv', index_col=0)
                # Convert the index to datetime with naive timestamps (no timestamps)
                data.index = pd.to_datetime(data.index).tz_localize(None)
                data.sort_index(inplace=True)
                data_dict_csv[tick] = data
                closes_csv[tick] = data['Close']
            else:
                print(self.path+tick + '.csv', 'do not exist !')
        closes_csv.index = pd.DatetimeIndex(closes_csv.index)
        self.returns_csv = closes_csv.pct_change().fillna(0)
        self.closes_csv = closes_csv
        self.data_dict_csv = data_dict_csv

    def yf_data_to_csv(self, tickers, start, end=(date.today() + timedelta(days=1)).isoformat()):
        for tick in tickers:
            yf_data = yf.download(tick, start, end, progress=False)
            yf_data.to_csv(self.path+tick + '.csv')

    def add_next_days(self,add_days):

        # Create and concat df copy of last days of data_bundle with index next business days range
        last_day=self.data_bundle.index[-1]
        next_days_range = pd.bdate_range(start=last_day, periods=add_days+1 , inclusive='right')

        next_days_data=self.data_bundle.tail(add_days).copy()
        next_days_data.index=next_days_range
        next_days_data.loc[:,] = [np.asarray(self.data_bundle.loc[last_day,])]
        self.data_bundle=pd.concat([self.data_bundle,next_days_data],axis=0)
        # Replace returns values of futures at expiration_dates by cash returns where available

    def data_dict_sanitize_OHL(self, data_dict):

        for ticker in data_dict.keys():
            # Get Ticker Data from Dict with OHLC for each ticker
            data = data_dict[ticker]

            # Locate where Open= Close , so Open to Close Return is Zero
            oc_diff_is_zero = (data['Close'] - data['Open']).abs() < 0.00000001

            # Replace Open by previous Close Where oc_diff_is_zero
            data.loc[oc_diff_is_zero, 'Open'] = data['Close'].shift(1)

            # Replace High by max(Open,Close) where High is equal to Low
            # Replace Low by min(Open,Close)
            hl_diff_is_zero = (data['Low'] - data['High']).abs() < 0.00000001
            data.loc[hl_diff_is_zero, 'High'] = data[['Open', 'Close']].max(axis=1)
            data.loc[hl_diff_is_zero, 'Low'] = data[['Open', 'Close']].min(axis=1)

            # Update data_dict
            data_dict[ticker] = data

        self.data_dict = data_dict

        return data_dict

def replace_fut_by_cash_returns_at_q_exp_or_after_dates(tickers_returns, fut_cash_tickers_dict,calendar,offline=False):

    path= "datasets/" # "datasets\\"
    # Get Expiration Dates
    exp_or_dayafter_dates = calendar.loc[calendar['is_expire'] | (calendar['days_to_exp'] == '1')].dropna().index
    end_is_before_expire=tickers_returns.index[-1]<=exp_or_dayafter_dates[-2]

    # Replace returns values of futures at expiration_dates by cash returns where available
    # Get Cash Returns
    cash_tickers = list(fut_cash_tickers_dict.values())

    if not offline & end_is_before_expire:
        cash_data_bundle=yf.download(cash_tickers, progress=False)
        cash_data_bundle.index=cash_data_bundle.index.tz_localize(None)
        cash_returns = cash_data_bundle['Close'].pct_change().dropna()

        #Save data to csv for further use
        for ticker in cash_tickers:
            cash_data_ticker=cash_data_bundle.xs(ticker, axis=1, level=1)
            cash_data_ticker.to_csv(path+ticker+'.csv')

    else:
        cash_returns=pd.DataFrame()
        for ticker in cash_tickers:
            if os.path.isfile(path+ticker+'.csv'):
                data = pd.read_csv(path+ticker+'.csv', index_col=0)
                data.index = pd.to_datetime(data.index)
            else:
                print(path+ticker + '.csv file not available')

            cash_returns[ticker]=data['Close'].pct_change()

        cash_returns=cash_returns.dropna()

        if end_is_before_expire:
            print('Offline Warning. Data at expiration not updated')


    # Replace returns values of Future at expiration_dates by Cash return
    start=max(cash_returns.index[0],tickers_returns.index[0])
    end=min(cash_returns.index[-1],tickers_returns.index[-1])

    for fut, cash in fut_cash_tickers_dict.items():
        tickers_returns[fut][start:end].loc[tickers_returns[start:end].index.isin(exp_or_dayafter_dates)] = cash_returns[cash].loc[cash_returns.index.isin(exp_or_dayafter_dates)].copy()

    return tickers_returns


class Indicators:

    def __init__(self,tickers_returns,settings):

        #Get cumulated returns
        self.cum_ret = (1 + tickers_returns).cumprod()

        # Rolling CAGR
        #cagr = tickers_returns.rolling(252).mean() * 252

        # Rolling VolatilityAnnualized
        #volat = tickers_returns.rolling(20).std() * (252) ** 0.5

        #Get RSI  for further use
        self.rsi = self.get_rsi(closes=tickers_returns,len=settings['rsi_w'],returns=True)

        self.rsi_weekly=self.get_rsi(closes=tickers_returns,len=14*5,returns=True).rolling(5).mean()

        self.rsi_weekly_weights = ((self.rsi_weekly > 50).astype(int)).shift(1).fillna(0)

        # Get Weights from selected indicators for further use

        #Get RSI Weights
        self.rsi_reverse_keep_weights=self.rsi_reverse_keep(self.rsi, upp=settings['rsi_upp'],window=settings['rsi_window'])
        #rsi_weights =self.rsi_weight(rsi,tickers_returns,settings, upp=70, low=30,window=22)

        #Plot
        if False:
            plt_df=pd.DataFrame()
            ticker='ES=F'
            plt_df['cum_ret'] = self.cum_ret[ticker]*100
            plt_df['rsi']=self.rsi[ticker]
            plt_df['rsi_high'] = self.rsi_high[ticker] * 100
            plt_df['rsi_high_keep'] = self.rsi_high_keep[ticker] * 100
            plt_df['rsi_weights'] = self.rsi_weights[ticker] * 100
            plt_df.plot(title='rsi_reverse_keep')



        # Get Data Normalized Weights

        data_norm=self.get_data_norm(self.cum_ret,252)
        r_opt_fun = self.get_rolling_opt_fun(tickers_returns, 22)
        self.norm_weights = self.get_corr_idx(data_norm, r_opt_fun, settings,fw=22, center=1.2, width=2.0) #center=1.2, width=2.0)

        #Get Bollinger Bands Indicators
        #lower, mid, upper, bandwidth, boll_pct=self.bbands(closes=cum_ret, len=250, std=2)

        # Get Bollinger pct weight
        #boll_pct_weights = self.get_boll_pct_weights(boll_pct)

        #Get Covariance Matrices
        #cov_matrices = tickers_returns.rolling(window=22).cov()*252

        # Get Rolling Covariance Matrices Dict for each Lookback  Period Window
        #rolling_cov_matrices_by_w_dict={str(w): tickers_returns.rolling(w).cov()*252 for w in settings['mkwtz_ws'] }

        #Get Rolling CAGR Dict for each Lookback  Period Window
        #rolling_cagr_by_w_dict = {str(w): tickers_returns.rolling(w).mean() * 252 for w in settings['mkwtz_ws']}

        #To retrieve Covariance Matrix for a specific date
        #cov_matrix_on_specific_date = cov_matrices.loc['your_specific_date']
        #For last date of tickers_returns
        #cov_matrix=cov_matrices.loc[tickers_returns.index[-1]]

        # ChoppinessIndex factor
        #ch_w = 180
        #chopp_factor = self.get_chopp_factor(cum_ret, ch_w)

        # Get Pair Correlation df
        corr_df = self.get_pair_correlation(tickers_returns)

        # Get Bounds df
        upper_corr = 0.7
        high_corr = (corr_df > upper_corr) * 1

        # High Correlation ES-NQ means strong trend
        if False:
        #if 'NQ=F' in settings['tickers']:
            overweighted_tickers = ['ES=F', 'NQ=F'] #
            bnds_max_df = pd.DataFrame()
            for tick in tickers_returns.columns:
                if tick in overweighted_tickers:
                    a = 1
                    b = 1
                else:
                    a = 1
                    b = -0
                esnq_high_corr = a + high_corr[['ES_NQ']].shift(1).rolling(5).mean() * b
                bnds_max_df[tick] = settings['tickers_bounds'][tick][1] * esnq_high_corr

        #Strong Trend Indicator
        trend_corr = self.get_trend_corr(tickers_returns)
        trend_corr_slow = self.get_trend_corr(tickers_returns, window=22 * 2)
        trend_corr_high = ((trend_corr > 0.5) & (trend_corr_slow > 0.5)) * 1
        trend_corr_high = 1 + trend_corr_high.shift(1) * 0.5

        # Get All Expiration Weights
        self.exp_weights = get_expiration_weights(tickers_returns)

        #RSI Sigmoid Weight
        self.rsi_sigmoid_weight=self.rsi_sigmoid(self.rsi)

        #Euribor Indicator Weights
        from EuriborStudy import get_Euribor_ind
        # Retrieve Training model and get Euribor Ind
        Euribor_series = tickers_returns['cash'] * 255 * 100
        self.Euribor_ind = get_Euribor_ind(Euribor_series)

        # Combined Weights
        self.comb_weights = self.rsi_reverse_keep_weights * self.norm_weights * self.exp_weights*self.Euribor_ind #* self.rsi_sigmoid_weight * m_trend_weights #* trend_corr_high#  * rsi_weights * chopp_factor #* boll_pct_weights
        self.comb_weights['cash']=1
        self.comb_weights = self.comb_weights/2.75
        self.comb_weights =self.comb_weights.clip(upper=2.5,lower=0)

        #Store indicators in a dict
        self.indicators_dict={
            'cum_ret': self.cum_ret,
            'rsi': self.rsi,
            #'rsi_weights': rsi_weights,
            'norm_weights': self.norm_weights,
            #'boll_pct': boll_pct,
            #'boll_pct_weights': boll_pct_weights,
            #'cov_matrices': cov_matrices,
            #'rolling_cagr_by_w_dict':rolling_cagr_by_w_dict,
            #'rolling_cov_matrices_by_w_dict': rolling_cov_matrices_by_w_dict,
            'corr_df': corr_df,
            #'bnds_max_df': bnds_max_df,
            'trend_corr_high': trend_corr_high,
            'rsi_high':self.rsi_high,
            'rsi_high_keep': self.rsi_high_keep,
            'rsi_reverse_keep_weights': self.rsi_reverse_keep_weights,
            'comb_weights': self.comb_weights,
        }

    def kelly(self, closes, len=14,returns=False):
        pass

    def get_trend_corr(self, tickers_returns, window=22):
        returns = tickers_returns.copy()
        # Add Linear Growth
        returns['linear'] = 0.01
        cum_ret = (1 + returns).cumprod()
        trend_corr = pd.DataFrame(index=returns.index)
        for ticker in tickers_returns.columns:
            trend_corr[ticker] = cum_ret[ticker].rolling(window=window).corr(cum_ret['linear'])
        return trend_corr

    def get_rsi(self, closes, len=14,returns=False):
        if returns== True:closes=(1+closes).cumprod()
        rsi = pd.DataFrame(np.asarray([
            ta.rsi(closes[tick], length=len)
            for tick in closes.columns]).T,
                           columns=closes.columns, index=closes.index)
        return rsi

    def bbands(self, closes, len=250, std=2):
        lower = closes * 0
        mid = closes * 0
        upper  = closes * 0
        bandwidth =  closes * 0
        percent = closes * 0

        for tick in closes.columns:
            df = ta.bbands(closes[tick], length=len, std=std)
            lower[tick],mid[tick], upper[tick], bandwidth[tick], percent[tick] = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3], df.iloc[:, 4]

        return lower, mid, upper, bandwidth, percent

    def rsi_countertrend(self,rsi,upp=60, low=40,upp_v=0.5, low_v=1.5):
        rsi_weights = pd.DataFrame(
            np.where(rsi > upp, upp_v,
                     np.where(rsi > low, 1.0, low_v)),
            columns=rsi.columns, index=rsi.index).shift(1).rolling(5, min_periods=1).mean()
        return rsi_weights


    def rsi_reverse_keep(self,rsi, upp=70,  window=22,width=.5): #_sigmoid rsi_not_low# upp_v=0.25, low_v=1.25, k=1.50,

        # Shift for yesterday close data
        rsi = rsi.shift(1)

        # Set rsi position with sigmoid function
        rsi_high =1 - sigmoid(x=rsi, center=upp,width=width)
        self.rsi_high = pd.DataFrame(rsi_high, columns=rsi.columns, index=rsi.index)

        # Keep high / low position for a while=window
        self.rsi_high_keep = self.rsi_high.rolling(window).max().fillna(0)

        # Set value around 1
        #self.rsi_weights = 1.5-self.rsi_high_keep
        self.rsi_weights = (1.324 - 1.136*self.rsi_high_keep)*1.3
        return self.rsi_weights


    def rsi_sigmoid(self,rsi): #_sigmoid rsi_not_low# upp_v=0.25, low_v=1.25, k=1.50,

        plt_df=pd.DataFrame()
        ticker='ES=F'
        plt_df['cum_ret'] = self.cum_ret[ticker]*100
        plt_df['rsi']=rsi[ticker]

        #Sigmoid RSI
        rsi_pct = rsi.pct_change()
        sigmoid_rsi = 1 / (1 + np.exp(-1000 * rsi_pct.rolling(20).mean()))  # returns values 0 to 1.

        plt_df['sigmoid_rsi'] = sigmoid_rsi[ticker]*100

        rsi_weights=0.5 + sigmoid_rsi
        rsi_weights = rsi_weights.clip(upper=1.20, lower=0.8)  # Limit Upper, Lower values

         # Shift for yesterday close data
        rsi_weights = rsi_weights.shift(1)

        #plt_df['rsi_weights'] = rsi_weights[ticker] * 100

        #plt_df.plot(title='rsi_sigmoid')

        return rsi_weights


    def rsi_weight(self,rsi,tickers_returns,settings, upp=70, low=30,window=22): #_sigmoid rsi_not_low# upp_v=0.25, low_v=1.25, k=1.50,

        #Tickers Metrics
        tickers_cumret = (1 + tickers_returns).cumprod()
        tickers_cumret_mean = tickers_cumret.rolling(10).mean()
        tickers_cumret_mean_fast = tickers_cumret.rolling(5).mean()
        tickers_cumret_mean_slow = tickers_cumret.rolling(22).mean()
        tickers_cumret_up = ((tickers_cumret_mean_fast > tickers_cumret_mean) & (tickers_cumret_mean > tickers_cumret_mean_slow)) * 1
        tickers_cumret_dn = ((tickers_cumret_mean < tickers_cumret_mean_slow) & (tickers_cumret_mean_fast < tickers_cumret_mean)) * 1

        # RSI metrics
        upp = 70
        upp = rsi / rsi * upp
        low = 30
        low = rsi / rsi * low
        window = 20
        rsi_high = (rsi > upp) * 1
        rsi_not_high = 1 - rsi_high
        rsi_low = (rsi < low) * 1
        rsi_not_low = 1 - rsi_low
        rsi_high_keep = rsi_high.rolling(window).max().fillna(0)
        rsi_low_keep = rsi_low.rolling(window).max().fillna(0)
        rsi_over_mid = (rsi > 65) * 1

        #Sell Signal
        rsi_sell_conditions = (rsi_high_keep * rsi_not_high + rsi_low).clip(upper=1)
        rsi_sell_signal = rsi_sell_conditions.diff().clip(lower=0)

        # Buy SIgnal
        rsi_buy_conditions = tickers_cumret_up * (rsi_low_keep * rsi_not_low + rsi_over_mid).clip(upper=1)
        rsi_buy_signal = rsi_buy_conditions.diff().clip(lower=0)
        rsi_buy_signal = (rsi_buy_signal + rsi_high.diff().clip(lower=0)).clip(upper=1)

        # Positions (Long or Closed)
        rsi_pos = (rsi_buy_signal - rsi_sell_signal)
        rsi_pos.replace(0, np.nan, inplace=True)
        rsi_pos = (rsi_pos.fillna(method='ffill').fillna(-1) + 1) / 2


        # Weights
        upper_w = 1.2 #1.5  # settings['upper_w']  # 1.25 #Value 1 to 2  #2 means 2 when True , 0 when false
        rsi_weights = (rsi_pos - 0.5) * 0.5 / (2 - upper_w) + 1

        # Resample with value of friday
        #rsi_weights = rsi_weights.resample('W-FRI').mean().reindex(rsi_weights.index).fillna(method='ffill')

        # Shift for yesterday close data
        rsi_weights = rsi_weights.shift(1)

        return rsi_weights

    def get_data_norm(self,data, window):
        mid = data.rolling(window).mean()
        std = data.rolling(window).std()
        data_norm = (data - mid) / std
        return data_norm

    def get_rolling_opt_fun(self,ret, fw):
        # Rolling CAGR
        r_cagr = ret.rolling(fw).mean() * 252
        # Rolling VolatilityAnnualized
        r_volat = ret.rolling(fw).std() * (252) ** 0.5
        # Sharpe=CAGR/Volat
        r_sharpe = r_cagr / r_volat
        # Optimize Function =  Monthly CAGR - Monthly average volatility
        # opt_fun=r_cagr-r_volat

        r_opt_fun = r_sharpe.shift(1)

        return r_opt_fun

    def get_corr_idx(self,indicator, r_opt_fun, settings,fw=22, center=1.0, width=2.0):
        """Correlation opt_fun vs indicator fw days ago"""
        corr = r_opt_fun.rolling(252).corr(indicator.shift(fw))

        #Keep only Significative Correlation over critical value
        corr_citical = 0.25 #Bellow 0.30 means negligible correlation
        corr_signif = np.where(np.abs(corr) > corr_citical, corr, 0)
        corr_signif = pd.DataFrame(corr_signif, columns=corr.columns, index=corr.index)

        # Raw Correlation Index
        corr_idx_raw = corr_signif * indicator

        # idx cut upper, lower parameters
        nd = 1.5
        std = corr_idx_raw.shift(5).std()
        mean = corr_idx_raw.shift(5).mean()
        u = mean + nd * std
        l = mean - nd * std

        # idx oxcilation params: center, width
        width = min(center * 2, width)  # to avoid negative values

        # corr_idx adjustement
        # Make idx oscilate from zero to width
        corr_idx = (corr_idx_raw.clip(upper=u, lower=l, axis=1) - l) / (u - l) * width
        # Raise mean to the center value
        corr_idx = (corr_idx + (center - corr_idx.shift(5).mean()))

        # Resample with value of friday
        if settings['weekly_trading_only']:
            corr_idx = corr_idx.resample('W-FRI').mean().reindex(corr_idx.index).fillna(method='ffill')

        # Use yesterday close value
        corr_idx = corr_idx.shift(1)

        return corr_idx

    # Get Bollinger pct weight
    def get_boll_pct_weights(self,boll_pct):
        max_val = 1.0
        min_val = 0.0 #0.5
        cut_val = 0.5 #0.5
        boll_pct_abs = (boll_pct.shift(1).rolling(5).mean() > cut_val).astype(int)
        boll_pct_weights = boll_pct_abs * (max_val - min_val) + min_val
        #Upper cut
        #boll_pct_weights[boll_pct > 1] = 0.5

        #Normalize
        boll_pct_weights = boll_pct_weights / boll_pct_weights.rolling(252 * 6).mean().fillna(1)

        # Apply only to selected assets
        #assets_to_apply = ['ES=F']
        #assets_to_keep = [element for element in boll_pct_weights.columns if element not in assets_to_apply]
        #boll_pct_weights[assets_to_keep] = 1

        #Reduce exposition of rest of assets when ES=F is bullish
        #boll_pct_weights.loc[boll_pct_abs['ES=F']==1,assets_to_keep]=0.5

        #Apply K
        #boll_pct_weights=boll_pct_weights*0.85

        return boll_pct_weights

    # ChoppinessIndex
    def get_chopiness(self,cum_ret, w=180):
        # Calculate True Range
        # Replace dayly High & Low by weekly
        high = cum_ret.rolling(5).max()
        low = cum_ret.rolling(5).min()
        TR = np.maximum(high - low,
                        np.maximum(abs(high - cum_ret.shift()),
                                   abs(low - cum_ret.shift())))

        # Calculate Average True Range Medium
        ATR = TR.rolling(window=w).mean()

        # Calculate logaritm of ATR / Price Range
        ln_ATR_PriceRange = np.log(ATR / (high.rolling(window=w).max() -
                                          low.rolling(window=w).min()))

        # Calculate Choppiness Index
        ChoppinessIndex = ln_ATR_PriceRange.rolling(window=w).sum() / np.log10(w)

        # Normalizar el índice de choppiness al rango de 0 a 100
        # ChoppinessIndex = (ChoppinessIndex - ChoppinessIndex.min()) / (ChoppinessIndex.max() - ChoppinessIndex.min()) * 100
        # ChoppinessIndex = (ChoppinessIndex - ChoppinessIndex.min()) / (ChoppinessIndex.rolling(250).max() - ChoppinessIndex.rolling(250).min()) * 100
        ch_mean = ChoppinessIndex.rolling(250 * 3).mean()
        ch_std = ChoppinessIndex.rolling(250 * 3).std()
        ch_max = ch_mean + 2.0 * ch_std
        ch_min = ch_mean - 2.0 * ch_std
        ChoppinessIndex = ((ChoppinessIndex - ch_min) / (ch_max - ch_min)).clip(lower=0, upper=1) * 100
        ChoppinessIndex.fillna(method='ffill', inplace=True)

        return ChoppinessIndex

    # ChoppinessIndex factor
    def get_chopp_factor(self,cum_ret, ch_w=180):
        ChoppinessIndex = self.get_chopiness(cum_ret, ch_w).shift(1)
        chopp_factor = np.where(ChoppinessIndex < 40, 1.25,
                                np.where(ChoppinessIndex > 60, 0.25, 1)) * 1.0 #0.90
        chopp_factor = pd.DataFrame(chopp_factor, index=cum_ret.index, columns=cum_ret.columns).shift(1).fillna(1)
        return chopp_factor

    # Get Pair Correlation
    def get_pair_correlation(self,tickers_returns):
        # Get Pair Correlation df
        corr_df = pd.DataFrame(index=tickers_returns.index)
        i = 0
        for ticker1 in tickers_returns.columns:
            i = i + 1
            for ticker2 in tickers_returns.columns[i:]:
                corr_matrix = tickers_returns[[ticker1, ticker2]].rolling(window=22).corr()
                corr_matrix.drop(columns=ticker1, inplace=True)
                corr_matrix.drop(ticker2, level=1, inplace=True)
                col = ticker1[:2] + '_' + ticker2[:2]
                corr_df[col] = np.array(corr_matrix)
        return corr_df

# Expiration Functions

def get_expiration_weights(tickers_returns):
    #Create Monthly Trading Calendar for Option of Index ES, NQ
    m_calendar=get_es_trading_calendar(tickers_returns,expiration_freq='M')
    #print('m_calendar\n',m_calendar['2023':].tail(20))

    #Get Rolling Bullish Probability Weight around expiration date
    fut_index_tickers=['ES=F', 'NQ=F']
    if 'NQ=F' not in tickers_returns.columns:
        fut_index_tickers = ['ES=F']


    #Get Dayly rolling weight for Futures of Index ES, NQ
    es_rolling_weights=get_dayly_df_rolling_weight(tickers_returns, m_calendar, fut_index_tickers)
    #print('rolling_weights_df\n', es_rolling_weights['2023'].tail(15))

    #Get Quarterly Calendar for Index Futures ES, NQ
    es_q_calendar = get_es_trading_calendar(tickers_returns,expiration_freq='Q')
    #print('es_q_calendar\n', es_q_calendar['2023'].tail(20))

    # Set weight zero Expiration and Day After for Futures Index ES,NQ
    tickers=tickers_returns.columns
    es_q_exp_closed = set_weight_zero_at_expiration_and_day_after(es_q_calendar,fut_index_tickers,tickers)

    #print('es_q_exp_closed\n', es_q_exp_closed['2023'].tail(15))

    #Gold Expiration calendar
    gc_calendar=get_gc_trading_calendar(tickers_returns)
    #print('gc_calendar\n', gc_calendar.tail(20))

    #Get Dayly rolling weight for Futures of Gold GC=F
    gc_tickers=['GC=F']
    gc_rolling_weights=get_dayly_df_rolling_weight(tickers_returns, gc_calendar, gc_tickers)
    #print('gc_rolling_weights_df\n', gc_rolling_weights.tail(15))

    # Set weight zero Expiration and Day After for Futures Gold GC=F
    gc_exp_closed = set_weight_zero_at_expiration_and_day_after(gc_calendar,gc_tickers,tickers)

    #print('gc_exp_closed \n', gc_exp_closed .tail(15))

    #All Rolling Weights
    exp_closed_weights=es_q_exp_closed * gc_exp_closed
    rolling_weights= es_rolling_weights * gc_rolling_weights
    exp_weights = exp_closed_weights  * rolling_weights

    return exp_weights

def get_bull_prob_around_expiration_dates(ret_around_expiration,ret_out_dates_around_exp):

    # Calculate the mean by Weekday & Weeks after Expiration day_exp
    ret_around_expiration_mean_by_day_exp = ret_around_expiration.groupby('days_to_exp').mean()
    #print('ret_around_expiration_mean_by_day_exp\n', ret_around_expiration_mean_by_day_exp)

    # Perform t-test
    res_by_ticker = []
    tickers=ret_around_expiration.drop(columns=[ 'days_to_exp']).columns
    for ticker in tickers: #Loop over each ticker
        res_by_day_exp=[]
        unique_day_exp=list(ret_around_expiration['days_to_exp'].unique())
        for days_to_exp in unique_day_exp: #Loop over each day around expiration dates
            sample_in=ret_around_expiration[ticker][ret_around_expiration['days_to_exp']==days_to_exp]
            day=ret_around_expiration.index.weekday[ret_around_expiration['days_to_exp']==days_to_exp][-1]
            sample_out=ret_out_dates_around_exp[ticker][ret_out_dates_around_exp.index.weekday==day]
            res = stats.ttest_ind(sample_in, sample_out)
            """The t-statistic is a measure of the difference between the two means relative to the variability in the data. A larger absolute value of the t-statistic indicates a larger difference between the means.
                The p-value is the probability of observing a t-statistic as extreme as the one you calculated, assuming the null hypothesis is true (i.e., the population means are equal). A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis."""
            res_by_day_exp.append(res)
        res_by_ticker.append(res_by_day_exp)

    p_value_df = pd.DataFrame([[i[1] for i in row ] for row in res_by_ticker], columns=unique_day_exp, index=tickers).T
    t_stat_df = pd.DataFrame([[i[0] for i in row ] for row in res_by_ticker], columns=unique_day_exp, index=tickers).T

    #print('p_value_ret_around_expiration_by_day_exp\n',p_value_ret_around_expiration_by_day_exp)
    #print('t_stat_ret_around_expiration_by_day_exp\n',t_stat_ret_around_expiration_by_day_exp)

    #Lets get an indicator of Bullish Probability with sign
    bull_prob_weight=(1-p_value_df) *np.sign(t_stat_df)

    return bull_prob_weight

def get_es_trading_calendar(returns,expiration_freq = 'Q'):
    """
    https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html
    MICRO E-MINI S&P 500 INDEX FUTURES, Also Valid fo Nasdaq NQ=F
    Quarterly contracts (Mar, Jun, Sep, Dec): q_months = [3, 6, 9, 12]
    TERMINATION OF TRADING: Trading terminates at 9:30 a.m. ET on the 3rd Friday of the contract month.

    :param returns:
    :return: calendar
    """

    #Create Calendar Dates to update
    start=returns.index[0]
    end=returns.index[-1]

    #Create a Calendar with all calendar data from Start to end of available tickers_returns
    calendar=pd.DataFrame(index=pd.date_range(start,end),columns=['is_expire']).fillna(False)

    #Add date of last data available
    calendar['date_of_last_data']=returns.index.to_series().reindex(calendar.index).fillna(method='ffill')

    #Add Month Expiration third friday of month
    third_friday_index=pd.date_range(start=start, end=end, freq='WOM-3FRI')
    #calendar['is_third_friday'].loc[third_friday_index]=True
    calendar['is_third_friday']=calendar.index.isin(third_friday_index)

    #Add quarterly Expiration
    # Define quarterly months
    q_months = [3, 6, 9, 12]
    is_contract_month = calendar.index.month.isin(q_months)
    calendar['is_q_third_friday']= calendar['is_third_friday'] & is_contract_month

    #Bollean for Expiration as Calendar dates as per selected expiration frequencey
    # Monhtly: 'M' or Quarterly: 'Q'
    if expiration_freq=='Q':
        is_calendar_exp=calendar['is_q_third_friday']
    elif expiration_freq=='M':
        is_calendar_exp = calendar['is_third_friday']

    #Add Quarterly Expiration Dates with last date of available data
    calendar['exp_date']=calendar['date_of_last_data'].loc[is_calendar_exp]

    #Add is effective/real expiration date (previous date in case of market close)
    calendar['is_expire'].loc[calendar['exp_date'].dropna()] = True

    #Drop Rows not in returns df
    calendar=calendar.reindex(returns.index)

    # Keep only 'is_expire'
    calendar = calendar[['is_expire']]  # 'day_exp_q',

    # Add days to Expiration
    calendar['days_to_exp'] = np.nan
    i_range = [-4, -3, -2, -1, 0, 1, 2]
    #i_range = [ -3, -2  ]
    for i in i_range:
        is_quarter_expire_i = calendar['is_expire'].shift(i).fillna(False)
        index_quarter_expire_i = is_quarter_expire_i.loc[is_quarter_expire_i].index
        calendar['days_to_exp'].loc[index_quarter_expire_i] = str(i)

    return calendar

def get_gc_trading_calendar(returns):
    """
    https://www.cmegroup.com/markets/metals/precious/e-micro-gold.contractSpecs.html
    Micro Gold Futures and Options
    TERMINATION OF TRADING:Trading terminates on the third last business day of the contract month.
    contract_months = [2, 4, 6, 8, 10, 12]
    :param tickers_returns:
    :return:
    df with expiration dates and more
    """

    #Create Calendar Dates
    start=returns.index[0]
    end=returns.index[-1]

    #Create a Calendar with all calendar data from Start to end of available returns
    calendar=pd.DataFrame(index=pd.date_range(start,end))

    #Get Calendar End of Month
    calendar['is_cal_end_of_month']=calendar.index.isin(calendar.resample('M').last().index)

    #Add contract months
    contract_months = [2, 4, 6, 8, 10, 12]
    is_contract_month = calendar.index.month.isin(contract_months)
    calendar['is_cal_end_of_contract_month'] = (calendar['is_cal_end_of_month'] & is_contract_month)

    # Add date of last data available in returns df
    calendar['date_of_last_data'] = returns.index.to_series().reindex(calendar.index).fillna(method='ffill')

    #Add End of Month Contract in returns index
    ret_end_of_contract_month =calendar['date_of_last_data'][calendar['is_cal_end_of_contract_month']]

    #Get calendar index End of Month Contract in returns index
    calendar['is_ret_end_of_contract_month']=calendar.index.isin(ret_end_of_contract_month)

    #Drop Rows not in returns df
    calendar=calendar.reindex(returns.index)

    #Get Third previous available day
    calendar['is_expire'] = calendar['is_ret_end_of_contract_month'].shift(-2)

    #Drop auxiliar columns
    calendar=calendar[['is_expire']]

    # Add days to Expiration
    calendar['days_to_exp'] = np.nan
    i_range = [-4, -3, -2, -1, 0, 1, 2]
    for i in i_range:
        is_quarter_expire_i = calendar['is_expire'].shift(i).fillna(False)
        index_quarter_expire_i = is_quarter_expire_i.loc[is_quarter_expire_i].index
        calendar['days_to_exp'].loc[index_quarter_expire_i] = str(i)


    return calendar

def get_yearly_dict_rolling_bull_prob_around_expiration_dates_OK(tickers_returns,calendar,tickers):

    # Returns around expiration dates
    dates_around_expiration_bool = ~calendar['days_to_exp'].isna()
    ret_around_expiration = tickers_returns.loc[dates_around_expiration_bool]
    #Add days to expiration for further use
    ret_around_expiration['days_to_exp']=calendar['days_to_exp']
    # Returns out of around expiration dates
    ret_out_dates_around_exp = tickers_returns[~dates_around_expiration_bool]

    #Create a dict with bull_prob_dict for all tickers and year end as key

    #Ceate Years intervals for rolling
    years = list(ret_around_expiration.index.year.unique())
    y=10 #10
    ends=[str(year) for year in years[y:]]
    starts=[str(year) for year in years[:-y]]
    ends_0=[str(year) for year in years[2:y]]
    starts_0=[str(years[0]) for i in range(len(ends_0))]
    starts=starts_0+starts
    ends=ends_0+ends

    #Rolling loop
    bull_prob_dict={}
    for start, end in zip(starts,ends):
        slice_in=ret_around_expiration.loc[start:end]
        slice_out = ret_out_dates_around_exp.loc[start:end]
        slice_bull_prob_ret_around_expiration_by_day_exp = get_bull_prob_around_expiration_dates(slice_in, slice_out)
        bull_prob_dict[end]=slice_bull_prob_ret_around_expiration_by_day_exp

    #Extract rolling_bull_prob_weight_dict for each selected ticker trough the years from bull_prob_dict for all tickers with years as keys
    rolling_bull_prob_weight_dict={}


    for ticker in tickers:

        ticker_rolling = pd.DataFrame()

        for year,bull_prob_df in bull_prob_dict.items():
            ticker_rolling[year] = bull_prob_df[ticker]

        ticker_rolling_bull_prob_ret_around_expiration_by_day_exp = ticker_rolling.T
        ticker_rolling_bull_prob_weight = ticker_rolling_bull_prob_ret_around_expiration_by_day_exp.shift(1)
        ticker_rolling_bull_prob_weight.dropna(inplace=True)
        ticker_rolling_bull_prob_weight.plot(title=ticker + '  p_value')

        if False:
            ##Discrete cut of p_value (weight is 1 when p_value > p_value_lim else 0)
            p_value_lim = 0.80 # 0.85
            bull_prob_weight = np.where(ticker_rolling_bull_prob_weight > p_value_lim, 1.5,
                                        np.where(ticker_rolling_bull_prob_weight < -p_value_lim , 0.0001, 1 )) #Avoid zero for further calc
            ticker_rolling_bull_prob_weight = pd.DataFrame(bull_prob_weight, index=ticker_rolling_bull_prob_weight.index, columns=ticker_rolling_bull_prob_weight.columns)

        else:
            #Continous weight
            ticker_rolling_bull_prob_weight = (ticker_rolling_bull_prob_weight+1)/2+0.5

        ticker_rolling_bull_prob_weight.plot(title=ticker + '  weight')

        rolling_bull_prob_weight_dict[ticker]=ticker_rolling_bull_prob_weight

    return rolling_bull_prob_weight_dict

def get_yearly_dict_rolling_bull_prob_around_expiration_dates(tickers_returns,calendar,tickers):

    # Returns around expiration dates
    dates_around_expiration_bool = ~calendar['days_to_exp'].isna()
    ret_around_expiration = tickers_returns.loc[dates_around_expiration_bool]
    #Add days to expiration for further use
    ret_around_expiration['days_to_exp']=calendar['days_to_exp']
    # Returns out of around expiration dates
    ret_out_dates_around_exp = tickers_returns[~dates_around_expiration_bool]

    #Create a dict with bull_prob_dict for all tickers and year end as key

    #Ceate Years intervals for rolling
    years = list(ret_around_expiration.index.year.unique())
    y=10 #10
    ends=[str(year) for year in years[y:]]
    starts=[str(year) for year in years[:-y]]
    ends_0=[str(year) for year in years[2:y]]
    starts_0=[str(years[0]) for i in range(len(ends_0))]
    starts=starts_0+starts
    ends=ends_0+ends

    #Rolling loop
    bull_prob_dict={}
    for start, end in zip(starts,ends):
        slice_in=ret_around_expiration.loc[start:end]
        slice_out = ret_out_dates_around_exp.loc[start:end]
        slice_bull_prob_ret_around_expiration_by_day_exp = get_bull_prob_around_expiration_dates(slice_in, slice_out)
        bull_prob_dict[end]=slice_bull_prob_ret_around_expiration_by_day_exp

    #Extract rolling_bull_prob_weight_dict for each selected ticker trough the years from bull_prob_dict for all tickers with years as keys
    rolling_bull_prob_weight_dict={}


    for ticker in tickers:

        ticker_rolling = pd.DataFrame()

        for year,bull_prob_df in bull_prob_dict.items():
            ticker_rolling[year] = bull_prob_df[ticker]

        #p_value df for the ticker
        ticker_p_value= ticker_rolling.T

        # Compute Times Series Consistency-> easy to predict

        # Create df for weight
        ticker_rolling_bull_prob_weight = pd.DataFrame(index=ticker_p_value.index, columns=ticker_p_value.columns)

        for m in ticker_p_value.columns:
            ts=ticker_p_value[m]#.shift(1).dropna() #Avoid last year to keep out of sample
            #Get Time Series Consistency
            ts_is_consistent, r2_value, std, model = time_series_consistency(ts, std_lim=0.10, r2_lim=0.65)

            if not ts_is_consistent:
                ticker_rolling_bull_prob_weight[m]=1

            else:

                # Create Weight from predicted p_value

                #Predicted p_value
                predicted_p_value=model(range(len(ticker_p_value)))

                if True:

                    # Continous weight
                    ticker_rolling_bull_prob_weight[m] = (predicted_p_value + 1) / 2 + 0.5

                else:

                    ##Discrete cut of p_value (weight is 1 when p_value > p_value_lim else 0)
                    p_value_lim = 0.80  # 0.85
                    ticker_rolling_bull_prob_weight[m]  = np.where(predicted_p_value > p_value_lim, 1.5,
                                                np.where(predicted_p_value < -p_value_lim, 0.0001, 1))  # 0.0001 Avoid zero for further calc

        #Clip Weights beetween 0.5-1.5
        ticker_rolling_bull_prob_weight=ticker_rolling_bull_prob_weight.clip(upper=1.5,lower=0.5)

        #print(ticker, ' ', ' ticker_rolling_bull_prob_weight\n', ticker_rolling_bull_prob_weight)

        #Plot p-value & Weight
        #ticker_p_value.plot(title=ticker + '  p_value')
        #ticker_rolling_bull_prob_weight.plot(title=ticker + '  weight')

        #Save weight to dict
        rolling_bull_prob_weight_dict[ticker]=ticker_rolling_bull_prob_weight

    return rolling_bull_prob_weight_dict

def time_series_consistency(ts, std_lim=0.07, r2_lim=0.85): #
    # R2 from Regresion
    n_dimension = 3  # 1 for linear regresion, 2 for quadratic regresions,...
    y_observed = ts
    x_observed = range(len(y_observed))
    model = np.poly1d(np.polyfit(x_observed, y_observed, n_dimension))
    y_model = model(x_observed)
    r2_value = r2_score(y_observed, y_model)

    # Standard Deviation
    std = ts.pct_change().fillna(0).std().item()

    # Values with Low std bellow std_lim
    low_std = std < std_lim

    # Values with high R2 over r2_lim
    high_r2 = r2_value > r2_lim

    # Time-series is consistent when Std is Low or R2 is High
    ts_is_consistent = (high_r2 | low_std)

    #Plot
    if False & ts_is_consistent:
        plot_df = pd.DataFrame(index=ts.index)
        plot_df['y_observed'] = ts
        plot_df['y_model'] = model(range(len(ts)))
        plot_df.plot(title=' R2=' + str(r2_value)[:4] + ' std=' + str(std)[:4] +
                           'ts_is_consistent = ' + str(ts_is_consistent))

    return ts_is_consistent, r2_value, std, model

def get_dayly_df_rolling_weight(returns,calendar,tickers):

        # Get Rolling Bullish Probability Weight around expiration date
        weights_dict = get_yearly_dict_rolling_bull_prob_around_expiration_dates(returns, calendar, tickers)

        rolling_weights_df = pd.DataFrame(columns=returns.columns)
        for ticker in tickers:
            weights_df = weights_dict[ticker]

            # Convert the index to datetime
            weights_df.index = pd.to_datetime(weights_df.index)

            #Shift to use the results of last year to current year
            weights_df=weights_df.shift(1)

            # Upsample to calendar data
            idx = calendar.index
            weights_df = weights_df.reindex(pd.date_range(idx[0], idx[-1])).fillna(method='ffill')

            # Reindex as returns index
            weights_df = weights_df.reindex(returns.index).fillna(method='ffill')

            #Fill back with first vailable data
            #weights_df = weights_df.fillna(method='bfill')

            # Add day_exp from calendar
            weights_df['days_to_exp'] = calendar.reindex(weights_df.index)['days_to_exp']

            # Create a new column 'weight' with values from the specified columns
            weights_df['weight']=0
            unique_day_exp = list(weights_df['days_to_exp'].dropna().unique())
            for col in unique_day_exp:
                weights_df['weight'] = weights_df['weight']+weights_df[col] * (weights_df['days_to_exp'] == col)

            # Replace zeros with np.nan
            weights_df['weight'] .replace(0, np.nan, inplace=True)

            # Keep only weight column and save
            rolling_weights_df[ticker]=weights_df[['weight']]

        #Set np.nan values to 1
        rolling_weights_df = rolling_weights_df.fillna(1)

        return rolling_weights_df

def set_weight_zero_at_expiration_and_day_after(calendar,fut_index_tickers,all_tickers):
    """
    Set weight zero Expiration and Day After
    Returns Data is not reliable at Expiration Dates and Day After, so Keep aout of market
    :param calendar:
    :return: exp_closed
    """
    exp_or_dayafter_dates = (calendar['days_to_exp'] == '0') | (calendar['days_to_exp'] == '1')
    exp_closed=pd.DataFrame(index=calendar.index,columns=all_tickers).fillna(1)
    exp_closed.loc[exp_or_dayafter_dates,fut_index_tickers] = 0

    return exp_closed


def get_fun_corr_factor(fun_df, len):
    # Get Optimize Function factor when Prediction is good
    # AutoCorrelation beetween current Optimize Function and future values
    fun_autocorrel = fun_df.rolling(window=len).corr(fun_df.shift(1))
    fun_autocorrel = fun_autocorrel.fillna(0).clip(upper=1, lower=-1)
    fun_autocorrel = fun_autocorrel.where(fun_df < -0.01, 1)
    #Factor making the mean and normalizing
    fun_corr_factor = fun_autocorrel.rolling(len).mean()
    fun_corr_factor = fun_corr_factor / fun_corr_factor.rolling(250 * 4).mean().fillna(0.8).mean().mean()
    return fun_corr_factor


def get_week_friday_resample(tickers_returns):
    # create a date range with daily frequency
    calendar_range = pd.date_range(start=tickers_returns.index.min(), end=tickers_returns.index.max(), freq='D')
    # reindex the original dataframe with the date range and fill missing values backward
    # To avoid Friday Holidays
    tickers_returns_all_calendar = tickers_returns.reindex(calendar_range).bfill()
    tickers_returns_res = tickers_returns_all_calendar.resample('W-FRI').last()
    return tickers_returns_res

def get_np_cov_matrices(tickers_returns,len):

    # Calculate the rolling covariance matrix
    cov_matrices_df = tickers_returns.rolling(window=len).cov()

    #Get numpy array
    return get_np_cov_matrices_from_df(cov_matrices_df)

def get_np_cov_matrices_from_df(cov_matrices_df):
    #Get numpy array
    np_cov_matrices=np.array(cov_matrices_df)
    l,n=np.shape(np_cov_matrices)

    #Reshape as Matrix (n_days,n_assets,n_assets)
    np_cov_matrices = np_cov_matrices.reshape(int(l / n), n , n)

    return np_cov_matrices



def get_garch_var(returns):
    returns=returns.fillna(0.0001)
    garch_var=pd.DataFrame(index=returns.index)
    np.random.seed(10)
    for ticker in returns.columns:
        ret=returns[ticker]
        model = arch_model(ret, vol='ARCH', p=1)
        #model = arch_model(ret)  # GARCH (with a Constant Mean)
        model_fit = model.fit(disp=False)
        forecast= model_fit.forecast(start=0)
        garch_var[ticker]=forecast.variance*252

    return garch_var

def get_np_cov_matrices_replaced_diagonal_with_garch_var(tickers_returns, np_cov_matrices):
    garch_var = get_garch_var(tickers_returns)
    m, n, _ = np_cov_matrices.shape
    np_cov_matrices[np.arange(m)[:, None], np.arange(n), np.arange(n)] = np.array(garch_var)

    return np_cov_matrices

def get_cash_values(df_index,rate,cash_init=1000):
    cash_rate_series = pd.Series(index=df_index, dtype='float64')
    cash_rate_series.iloc[:] = rate
    cash_values = cash_init * (1 + cash_rate_series / 255).cumprod()

    return cash_values

def add_cash_to_data_bundle(data_bundle, cash_rate):

    cash_values = get_cash_values(data_bundle.index,cash_rate,cash_init=1000)

    # Get the existing sub-column names
    sub_columns = data_bundle.columns.get_level_values(1).unique()

    # Assign the cash values to the new columns
    for sub_column in sub_columns:
        data_bundle[('cash', sub_column)] = cash_values.values  # Use .values to avoid alignment issues

    data_bundle_with_cash = data_bundle.copy()

    return data_bundle_with_cash








