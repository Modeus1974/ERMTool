
import numpy as np
import pandas as pd
from scipy.optimize import brute
import scipy as sp
import matplotlib.pyplot as plt
import pandas_datareader.data as getData
import numpy as np
from scipy import stats
import yfinance as yf


class TABacktester(object):
    ''' Class for the vectorized backtesting of SMA-based trading strategies.
    Attributes
    ==========
    symbol: str
        RIC symbol with which to work with
    SMA1: int
        time window in days for shorter SMA
    SMA2: int
        time window in days for longer SMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    set_parameters:
        sets one or two new SMA parameters
    run_strategy:
        runs the backtest for the SMA-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    update_and_run:
        updates SMA parameters and returns the (negative) absolute performance
    optimize_parameters:
        implements a brute force optimizeation for the two SMA parameters
    '''

    def __init__(self, symbol, SMA1, SMA2, RSI, MR, start, end, enable_short):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.enable_short = enable_short
        self.results_sma = None
        self.results_rsi = None
        self.results_mr = None
        self.signal_sma = 'Buy'
        self.signal_rsi = 'Buy'
        self.signal_mr = 'Buy'
        self.RSI = RSI
        self.MR = MR
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        stock_data = getData.DataReader(self.symbol,data_source='yahoo',start=self.start,end=self.end)
        raw = stock_data.drop(["Open","High","Low","Adj Close","Volume"],axis=1)
        #print(raw)
        raw.rename(columns={"Close": 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()

        ''' Deriving the RSI
        '''
        dUp, dDown = raw['return'].copy(), raw['return'].copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        RolUp = dUp.rolling(self.RSI).mean()
        RolDown = dDown.rolling(self.RSI).mean().abs()
        RS = RolUp / RolDown
        raw['RSI'] = 100 - (100 / (1+RS))

        self.data = raw
        #print(raw)


    def set_parameters(self, SMA1=None, SMA2=None, RSI=None, MR=None):
        ''' Updates SMA parameters and resp. time series.
        '''
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(
                self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()
        if (RSI or MR) is not None:
            if (RSI is not None ):
              self.RSI = RSI
            if (MR is not None ):
              self.MR = MR
            dUp, dDown = self.data['return'].copy(),self.data['return'].copy()
            dUp[dUp < 0] = 0
            dDown[dDown > 0] = 0

            RolUp = dUp.rolling(RSI).mean()
            RolDown = dDown.rolling(RSI).mean().abs()

            RS = RolUp / RolDown
            self.data['RSI'] = 100 - (100 / (1+RS))

    def run_SMA_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data['position_sma'] = np.where(data['SMA1'] > data['SMA2'], 1, self.enable_short)
        data['strategy_sma'] = data['position_sma'].shift(1) * data['return']
        data.dropna(inplace=True)
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy_sma'] = data['strategy_sma'].cumsum().apply(np.exp)
        final_position = data['position_sma'].tail(1)
        if final_position[0] == 1:
          self.signal_sma = "Buy"
        elif final_position[0] == 0:
          self.signal_sma = "Don't Buy"
        else:
          self.signal_sma = "Go Short"
        self.results_sma = data
        # gross performance of the strategy
        aperf = data['cstrategy_sma'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def run_RSI_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data['position_rsi'] = np.where(data['RSI'] > 50, 1, self.enable_short)
        data['strategy_rsi'] = data['position_rsi'].shift(1) * data['return']
        data.dropna(inplace=True)
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy_rsi'] = data['strategy_rsi'].cumsum().apply(np.exp)

        final_position = data['position_rsi'].tail(1)
        if final_position[0] == 1:
          self.signal_rsi = "Buy"
        elif final_position[0] == 0:
          self.signal_rsi = "Don't Buy"
        else:
          self.signal_rsi = "Go Short"

        self.results_rsi = data
        # gross performance of the strategy
        aperf = data['cstrategy_rsi'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def run_MR_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()

        data['position_mr'] = np.where(data['RSI'] > 70, -1, np.nan)
        data['position_mr'] = np.where(data['RSI'] < 30, 1, np.nan)
        data['position_mr'] = np.where((data['RSI']-50)*(data['RSI'].shift(1)-50)<0,0,data['position_mr'])
        data['position_mr'] = data['position_mr'].ffill().fillna(0)

        data['strategy_mr'] = data['position_mr'].shift(1) * data['return']
        data.dropna(inplace=True)
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy_mr'] = data['strategy_mr'].cumsum().apply(np.exp)

        final_position = data['position_mr'].tail(1)
        if final_position[0] == 1:
          self.signal_mr = "Buy"
        elif final_position[0] == 0:
          self.signal_mr = "Don't Buy"
        else:
          self.signal_mr = "Go Short"

        self.results_mr = data
        # gross performance of the strategy
        aperf = data['cstrategy_mr'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_SMA_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results_sma is None:
            print('No results to plot yet. Run a strategy.')
        signal = self.signal_sma
        title = '%s | MA crossover SMA1=%d, SMA2=%d | Current Signal is %s' % (self.symbol,
                                               self.SMA1, self.SMA2,signal)
        plt = self.results_sma[['creturns', 'cstrategy_sma']].plot(title=title,
                                                     figsize=(10, 6))
        return plt

    def plot_RSI_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results_rsi is None:
            print('No results to plot yet. Run a strategy.')
        signal = self.signal_rsi
        title = '%s | Momentum RSI=%d | Current Signal is %s' % (self.symbol,self.RSI,signal)
        plt = self.results_rsi[['creturns', 'cstrategy_rsi']].plot(title=title,
                                                     figsize=(10, 6))
        return plt

    def plot_MR_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results_mr is None:
            print('No results to plot yet. Run a strategy.')
        signal = self.signal_mr
        title = '%s | Mean Reverting RSI=%d | Current Signal is %s ' % (self.symbol,self.MR,signal)
        plt = self.results_mr[['creturns', 'cstrategy_mr']].plot(title=title,
                                                     figsize=(10, 6))
        return plt

    def update_and_run_SMA(self, SMA):
        ''' Updates SMA parameters and returns negative absolute performance
        (for minimazation algorithm).
        Parameters
        ==========
        SMA: tuple
            SMA parameter tuple
        '''
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_SMA_strategy()[0]

    def update_and_run_RSI(self, RSI):
        ''' Updates RSI parameters and returns negative absolute performance
        (for minimazation algorithm).
        Parameters
        ==========
        RSI
        '''
        self.set_parameters(RSI=RSI)
        return -self.run_RSI_strategy()[0]

    def update_and_run_MR(self, RSI):
        ''' Updates RSI parameters and returns negative absolute performance
        (for minimazation algorithm).
        Parameters
        ==========
        RSI
        '''
        self.set_parameters(RSI=RSI)
        return -self.run_MR_strategy()[0]

    def optimize_parameters_SMA(self, SMA1_range, SMA2_range):
        ''' Finds global maximum given the SMA parameter ranges.
        Parameters
        ==========
        SMA1_range, SMA2_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run_SMA, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run_SMA(opt)

    def optimize_parameters_RSI(self, max):
        optimal = 0
        best_return = 9999
        for i in range(1,180):
          current = self.update_and_run_RSI(i+1)
          if (current<best_return):
            best_return = current
            optimal = i+1
        return optimal, -self.update_and_run_RSI(optimal)

    def optimize_parameters_MR(self, max):
        optimal = 0
        best_return = 9999
        for i in range(max):
          current = self.update_and_run_MR(i+1)
          if (current<best_return):
            best_return = current
            optimal = i+1
        return optimal, -self.update_and_run_MR(optimal)

    def optimize(self):
      self.optimize_parameters_SMA((5, 105, 10), (100, 300, 20))
      self.optimize_parameters_RSI(180)
      self.optimize_parameters_MR(180)
      return

    def plot_all(self):
      self.plot_SMA_results()
      self.plot_RSI_results()
      self.plot_MR_results()
      return
