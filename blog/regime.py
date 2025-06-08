import numpy as np
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date,timedelta

import scipy as sp
import matplotlib.pyplot as plt
import pandas_datareader.data as getData
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import yfinance as yf
from datetime import date,timedelta
from hurst import compute_Hc, random_walk

import cvxpy as cp
import seaborn as sns
sns.set()

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def build_stock_prices_dataframe(stocklist,lookback_in_years,monthly=False):
      today = date.today()
      start_date = today-timedelta(lookback_in_years*252)

      stocks_dataframe = pd.DataFrame()

      for stock in stocklist:
        stock_data = yf.download(stock,start_date,today)
        returns_data = stock_data["Close"]
        stocks_dataframe[stock] = returns_data

      stocks_dataframe = stocks_dataframe.ffill(axis=0)

      if monthly:
        stocks_dataframe = stocks_dataframe.resample("M").mean()

      return stocks_dataframe

def build_portfolio_prices_dataframe(stocklist,lookback_in_years):
      today = date.today()
      start_date = today-timedelta(lookback_in_years*252)
      stocks_dataframe = pd.DataFrame()

      for stock in stocklist:
        stock_data = yf.download(stock,start_date,today)
        returns_data = stock_data["Close"]
        stocks_dataframe[stock] = returns_data.ffill(axis=0)
        stocks_dataframe[stock] = stocks_dataframe[stock].pct_change()

      stocks_dataframe['Ret'] = stocks_dataframe.mean(axis=1)
      stocks_dataframe['Return'] =  stocks_dataframe.mean(axis=1) + 1
      stocks_dataframe['Portfolio'] = stocks_dataframe['Return'].cumprod()
      stocks_dataframe['Portfolio'][0] = 1

      return stocks_dataframe


def get_stock_returns_array(stocklist, lookback_in_years, monthly=False):

      returns_list=[]
      size = len(stocklist)

      if monthly:
        period=12
      else:
        period=252

      stocks_dataframe = build_stock_prices_dataframe(stocklist, lookback_in_years,monthly)
      #print(stocks_dataframe)

      for stock in stocklist:
        stock_daily_returns = stocks_dataframe[stock].pct_change().dropna()
        #print(stock_daily_returns)
        number_days = stock_daily_returns.shape[0]
        daily_return = ((stock_daily_returns+1).prod()**(1/number_days)-1)
        annual_return = round(((daily_return+1)**period - 1)*100,3)
        returns_list.append(annual_return)

      input = stocks_dataframe.pct_change().dropna().transpose()
      covariance_matrix = np.cov(input)

      return np.array(returns_list),covariance_matrix

      # Functions for our analysis - feel free to edit
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

def regime_return(asset_pd, column_number, regime_col):
    """Computes returns of a list of regime columns identified by number"""
    asset_name = asset_pd.columns[column_number]
    regime = asset_pd[regime_col].values[:-1]
    asset_return = np.diff(asset_pd[asset_name],axis=0) / asset_pd[asset_name].values[:-1,:]
    ret_g, ret_c = asset_return[regime==1,:], asset_return[regime==-1,:]
    return asset_return, ret_g, ret_c

def regime_hist(asset_pd, column_number, regime_col):
    """Plots the asset regimes with regime column identified by number"""
    asset_return, ret_g, ret_c = regime_return(asset_pd, column_number, regime_col)
    plt.hist(ret_g, bins=20, color='green', label='Growth Regime',alpha=0.3)
    plt.hist(ret_c, bins=15, color='red', label='Contraction Regime',alpha=0.3)
    plt.xlabel('Monthly Return')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title('Regime Histogram of Asset: ' + asset_name)
    return ret_g, ret_c

def Q_Q_plot(asset_data, column_num):
    plt.figure(figsize=(12,9))
    res = scipy.stats.probplot(ret[:,column_num], plot=plt)
    plt.title('Q-Q Plot of Asset: ' + asset_data.columns[column_num], fontsize=24)
    plt.ylabel('Returns')
    plt.show()

def regime_plot(asset_data, column_num):
    ret_g1 = ret_g[:,column_num]
    ret_c1 = ret_c[:,column_num]
    plt.figure(figsize=(12,9))
    plt.plot(ecdf(ret_g1)[0], ecdf(ret_g1)[1], color='green',label='Normal Regime')
    plt.plot(ecdf(ret_c1)[0], ecdf(ret_c1)[1], color='red',label='Crash Regime')
    plt.xlabel('Monthly Return')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='upper left')
    plt.title('Cumulative Density of Asset: ' + asset_data.columns[column_num], fontsize=24)
    plt.show()

def trend_filtering(data,lambda_value):
    '''Runs trend-filtering algorithm to separate regimes.
        data: numpy array of total returns.'''

    n = np.size(data)
    x_ret = data.reshape(n)

    Dfull = np.diag([1]*n) - np.diag([1]*(n-1),1)
    D = Dfull[0:(n-1),]

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)

    def tf_obj(x,beta,lambd):
        return cp.norm(x-beta,2)**2 + lambd*cp.norm(cp.matmul(D, beta),1)

    problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))

    lambd.value = lambda_value
    problem.solve()

    return beta.value

def filter_plot(data,lambda_value,regime_num=0, TR_num=1):
    ret_sp = data.iloc[:,regime_num]
    sp500TR = data.values[:,TR_num]

    beta_value = trend_filtering(ret_sp.values,lambda_value)
    betas = pd.Series(beta_value,index=data.index)

    plt.figure(figsize=(12,9))
    plt.plot(ret_sp, alpha=0.4,label='Original Series')
    plt.plot(betas,label='Fitted Series')
    plt.xlabel('Year')
    plt.ylabel('Monthly Return (%)')
    plt.legend(loc='upper right')
    plt.show()

def regime_switch(betas,threshold=1e-5):
    '''returns list of starting points of each regime'''
    n = len(betas)
    init_points = [0]
    curr_reg = (betas[0]>threshold)
    for i in range(n):
        if (betas[i]>threshold) == (not curr_reg):
            curr_reg = not curr_reg
            init_points.append(i)
    init_points.append(n)
    return init_points

def plot_regime_color(dataset, regime_num=0, TR_num=1, lambda_value=16, log_TR = True,label = ""):
    '''Plot of return series versus regime'''
    returns = dataset.iloc[:,regime_num]
    TR = dataset.iloc[:,TR_num]
    betas = trend_filtering(returns.values,lambda_value)
    regimelist = regime_switch(betas)
    curr_reg = np.sign(betas[0]-1e-5)
    y_max = np.max(TR) + 500

    if log_TR:
        fig, ax = plt.subplots()
        for i in range(len(regimelist)-1):
            if curr_reg == 1:
                ax.axhspan(0, y_max+500, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1],
                       facecolor='green', alpha=0.3)
            else:
                ax.axhspan(0, y_max+500, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1],
                       facecolor='red', alpha=0.5)
            curr_reg = -1 * curr_reg

        fig.set_size_inches(12,9)
        plt.plot(TR, label='Total Return')
        plt.ylabel(f'{label} Log-scale')
        plt.xlabel('Year')
        plt.yscale('log')
        plt.xlim([dataset.index[0], dataset.index[-1]])
        plt.ylim([80, 3000])
        plt.yticks([100, 500, 1000, 2000, 3000],[100, 500, 1000, 2000, 3000])
        plt.title(f'Regime Plot of {label} values', fontsize=24)
        plt.show()

def geo_return(X, input_type='Return'):
    """Computes geometric return for each asset"""
    if input_type == 'Return':
        X_geo = 1+X
        y = np.cumprod(X_geo,axis=0)
        return (y[-1,:]) ** (1/X.shape[0]) - 1
    else:
        return (X[-1,:] / X[0,:]) ** (1/(X.shape[0]-1)) - 1

def portfolio_opt(mu, Q, r_bar):
    w = cp.Variable(mu.size)
    ret = mu.T*w
    risk = cp.quad_form(w, Q)
    prob = cp.Problem(cp.Minimize(risk),
                   [cp.sum(w) == 1, w >= 0, ret >= r_bar])
    prob.solve()
    return np.round_(w.value,decimals=3)

def efficient_frontier_traditional(r_annual, Q_all, r_bar):

    n_asset = r_annual.size
    weight_vec = np.zeros((len(r_bar),n_asset))
    risk_port = np.zeros(len(r_bar))
    ret_port = np.zeros(len(r_bar))

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        ret_opt = r_annual.T*w_opt
        risk_opt = cp.quad_form(w_opt, Q_all)
        prob = cp.Problem(cp.Minimize(risk_opt),
                       [cp.sum(w_opt) == 1, w_opt >= 0, ret_opt >= r_curr])
        prob.solve()

        weight_vec[i,:] = w_opt.value
        ret_port[i] = ret_opt.value
        risk_port[i] = np.sqrt(risk_opt.value)

    plt.figure(figsize=(12,9))
    plt.plot(risk_port*100, ret_port*100, 'xb-')
    plt.xlabel("Risk (%)")
    plt.ylabel("Nominal Return (%)")
    plt.title("Efficient Frontier: Single-Period", fontsize=24);

def efficient_frontier_scenario(r_all_1, r_bar):
    Q_1 = np.cov(r_all_1.T)
    mu_1 = np.mean(r_all_1, axis=0)

    efficient_frontier_traditional(r_annual, Q_all, r_bar)
    plt.title("Efficient Frontier: Single-Period, Scenario-Equivalent Version", fontsize=24);

def efficient_frontier_comparison(r_annual, Q_all, r_bar, n_scenarios=10000):
    n_asset = r_annual.size
    weight_vec = np.zeros((len(r_bar),n_asset))
    risk_port = np.zeros(len(r_bar))
    ret_port = np.zeros(len(r_bar))

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        ret_opt = r_annual.T*w_opt
        risk_opt = cp.quad_form(w_opt, Q_all)
        prob = cp.Problem(cp.Minimize(risk_opt),
                       [cp.sum(w_opt) == 1, w_opt >= 0, ret_opt >= r_curr])
        prob.solve()

        weight_vec[i,:] = w_opt.value
        ret_port[i] = ret_opt.value
        risk_port[i] = np.sqrt(risk_opt.value)

    plt.subplot(121)
    plt.plot(risk_port*100, ret_port*100, 'xb-')
    plt.xlabel("Risk (%)")
    plt.ylabel("Nominal Return (%)")
    plt.title("Traditional", fontsize=16);

    r_all_1 = np.random.multivariate_normal(r_annual.reshape(n_asset), Q_all, n_scenarios)

    Q_1 = np.cov(r_all_1.T)
    mu_1 = np.mean(r_all_1, axis=0)
    weight_vec1 = np.zeros((len(r_bar),n_asset))
    risk_port1 = np.zeros(len(r_bar))
    ret_port1 = np.zeros(len(r_bar))

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        ret_opt = mu_1.T*w_opt
        risk_opt = cp.quad_form(w_opt, Q_1)
        prob = cp.Problem(cp.Minimize(risk_opt),
                       [cp.sum(w_opt) == 1, w_opt >= 0, ret_opt >= r_curr])
        prob.solve()

        weight_vec1[i,:] = w_opt.value
        ret_port1[i] = ret_opt.value
        risk_port1[i] = np.sqrt(risk_opt.value)

    plt.subplot(122)
    plt.plot(risk_port1*100, ret_port1*100, 'xb-')
    plt.xlabel("Risk (%)")
    plt.title("Scenario-Equivalent", fontsize=16);

def efficient_frontier_twoRegime(ret, ret_g, ret_c, r_bar, n_scenarios = 10000):
    Q_all = np.cov(ret.T) * 12
    r_annual = (1+geo_return(ret)) ** 12 - 1
    r_annual = r_annual.reshape(-1,1)
    r_g = (1+geo_return(ret_g)) ** 12 - 1
    r_c = (1+geo_return(ret_c)) ** 12 - 1
    n_g = int(n_scenarios*ret_g.shape[0] / ret.shape[0])
    Q_g = np.cov(ret_g.T) * 12
    Q_c = np.cov(ret_c.T) * 12
    n_asset = r_annual.size

    s_1 = np.random.multivariate_normal(r_g, Q_g, n_g)
    s_2 = np.random.multivariate_normal(r_c, Q_c, n_scenarios-n_g)
    r_all_2 = np.vstack((s_1,s_2))

    Q_2 = np.cov(r_all_2.T)
    mu_2 = np.mean(r_all_2, axis=0)

    weight_vec2 = np.zeros((len(r_bar),n_asset))
    risk_port2 = np.zeros(len(r_bar))
    ret_port2 = np.zeros(len(r_bar))

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        ret_opt = mu_2.T*w_opt
        risk_opt = cp.quad_form(w_opt, Q_2)
        prob = cp.Problem(cp.Minimize(risk_opt),
                       [cp.sum(w_opt) == 1, w_opt >= 0, ret_opt >= r_curr])
        prob.solve()

        weight_vec2[i,:] = w_opt.value
        ret_port2[i] = ret_opt.value
        risk_port2[i] = np.sqrt(risk_opt.value)

    efficient_frontier_traditional(r_annual, Q_all, r_bar)
    plt.plot(risk_port2*100, ret_port2*100, 'xr-',label='Two-Regime')
    plt.legend(loc='best', ncol=2, shadow=True, fancybox=True,fontsize=16)
    plt.title("Efficient Frontier: Single-Period, Traditional vs Two-Regime Version")
    plt.show()
    return r_all_2

# Endowment Simulation: still in progress
def regime_asset(n,mu1,mu2,Q1,Q2,p1,p2):
    s_1 = np.random.multivariate_normal(mu1, Q1, n).T
    s_2 = np.random.multivariate_normal(mu2, Q2, n).T
    regime = np.ones(n)
    for i in range(n-1):
        if regime[i] == 1:
            if np.random.rand() > p1:
                regime[i+1] = 0
        else:
            if np.random.rand() > p2:
                regime[i+1] = 0
    return (regime*s_1 + (1-regime)*s_2).T

def transition_matrix(regime):
    """Computes the transition matrix given the regime vector
    """
    n1,n2,n3,n4 = 0,0,0,0
    for i in range(len(regime)-1):
        if regime[i] == 1:
            if regime[i+1] == 1:
                n1 += 1
            else:
                n2 += 1
        else:
            if regime[i+1] == 1:
                n3 += 1
            else:
                n4 += 1
    return n1/(n1+n2), n2/(n1+n2), n3/(n3+n4), n4/(n3+n4)

def asset_simulation(assets_info, asset_num, regime_name, random_seed=777, n_scenarios=10000, n_years=50):
    """Simulates regime-based monthly returns.
    assets_info is a pandas Dataframe containing asset total return indices; please refer to the dataset for format.
    asset_num is the number of assets we would like to use. By default, this should be the first few columns in dataset.
    regime_name is the column name of regime in the dataset.

    Returns a (n_year*12) * n_asset * n_scenario tensor for all asset information.
    """
    ret_all, ret_g, ret_c = regime_return(assets_info, np.arange(asset_num), 'Regime-5')
    regime = assets_info[regime_name].values[:-1] # lose 1 value from computing returns
    p1, _, p2, _ = transition_matrix(regime)
    mu1 = 1+geo_return(ret_g)
    mu2 = 1+geo_return(ret_c)
    Q1 = np.cov(ret_g.T)
    Q2 = np.cov(ret_c.T)
    r_all = np.zeros((n_years*12, asset_num, n_scenarios))

    np.random.seed(random_seed)
    for i in range(n_scenarios):
        r_all[:,:,i] = regime_asset(n_years*12,mu1,mu2,Q1,Q2,p1,p2)
    return r_all

def fund_simulation(holdings, asset_return, hold_type='fixed',spending_rate=0.03):
    """Simulates monthly data of a fund for a certain number of years.
    asset_return should be total return, i.e. 1 plus the percentage return.
    if hold_type is "fixed" (by default), holdings is fixed mix
    if hold_type is a number, this is rebalance frequency (in months)
    if hold_type is "dynamic", dynamic portfolio optimization will be conducted (to be implemented...)

    The simulation returns a full path of wealth at the end of each year, so it is a n_scenarios*n_years matrix.
    """
    n_months, n_assets, n_scenarios = asset_return.shape
    wealth_path = np.zeros((n_scenarios, int(n_months/12)))

    if hold_type == 'fixed':
        for i in range(n_scenarios):
            holdings_each = holdings
            for j in range(n_months):
                holdings_each = holdings_each * asset_return[j,:,i]
                if j%12==0:
                    holdings_each = holdings_each * (1-spending_rate)
                    wealth_path[i,int(j/12)] = np.sum(holdings_each)
        return wealth_path

    elif type(hold_type)==int:
        for i in range(n_scenarios):
            holdings_each = holdings
            for j in range(n_months):
                holdings_each = holdings_each * asset_return[j,:,i]
                if j%hold_type == 0: # Rebalance
                    asset_temp = np.sum(holdings_each)
                    holdings_each = asset_temp * holdings
                if j%12==0:
                    holdings_each = holdings_each * (1-spending_rate)
                    wealth_path[i,int(j/12)] = np.sum(holdings_each)
        return wealth_path

    else: # "Dynamic" -- to be implemented
        return 0

def plot_regime_color_new(dataset, regime_num=0, TR_num=1, lambda_value=16, log_TR = True,label = ""):
    '''Plot of return series versus regime'''
    returns = dataset.iloc[:,regime_num]
    TR = dataset.iloc[:,TR_num]
    betas = trend_filtering(returns.values,lambda_value)
    betas_df = pd.Series(betas,index=dataset.index)
    #print(TR)
    #print(betas)
    regimelist = regime_switch(betas)
    #print(regimelist)
    curr_reg = np.sign(betas[0]-1e-5)
    y_max = np.max(TR) #+ 500
    state = []

    if log_TR:
        fig, ax = plt.subplots()
        for i in range(len(regimelist)-1):
            if curr_reg == 1:
                ax.axhspan(0, y_max, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1],
                       facecolor='green', alpha=0.3)
                state.extend(["Normal"]*(regimelist[i+1]-regimelist[i]))
            else:
                ax.axhspan(0, y_max, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1],
                       facecolor='red', alpha=0.5)
                state.extend(["Crash"]*(regimelist[i+1]-regimelist[i]))
            curr_reg = -1 * curr_reg

        #fig.set_size_inches(7,2)
        #plt.plot(TR, label='Total Return)
        #plt.plot(betas_df,color="black",label='Fitted Series')
        plt.plot(TR)
        plt.ylabel(f'{label}',fontsize=12)
        plt.xticks(rotation=30,fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Year')
        #plt.yscale('log')
        plt.xlim([dataset.index[0], dataset.index[-1]])
        plt.title(f'Regime Plot of {label} values', fontsize=12)
        plt.legend(loc='upper right')
    dataset["State"] = state
    return plt

def analyse_regime(stocks_ticker):

  normal_data = stocks_data[stocks_data["State"]=="Normal"]
  crash_data = stocks_data[stocks_data["State"]=="Crash"]

  stocks_return = (1+stocks_data["Ret"]/100).prod()**(252/len(stocks_data))-1
  normal_return = (1+normal_data["Ret"]/100).prod()**(252/len(normal_data))-1
  crash_return = (1+crash_data["Ret"]/100).prod()**(252/len(crash_data))-1

  stocks_std = (stocks_data["Ret"]/100).std()*np.sqrt(252)*100
  normal_std = (normal_data["Ret"]/100).std()*np.sqrt(252)*100
  crash_std = (crash_data["Ret"]/100).std()*np.sqrt(252)*100

  #print(f"Normal Shape {len(normal_data)} Crash Shape {len(crash_data)} ")

  print(f"          Overall return = {round(stocks_return*100,2)}%, SD = {round(stocks_std,2)}%")
  print(f"          Normal return = {round(normal_return*100,2)}%, SD = {round(normal_std,2)}%")
  print(f"          Crash return = {round(crash_return*100,2)}, SD = {round(crash_std,2)}%")

  df = pd.DataFrame({'Normal':[normal_return,normal_std],'Crash':[crash_return,crash_std],'Overall':[stock_return,stock_std]})
  print(df)


def return_regime_graph(stock_ticker):
  stock = stock_ticker

  stock_data = build_stock_prices_dataframe([stock],5,monthly=False)
  stock_data["Ret"] = stock_data.pct_change()*100
  stock_data = stock_data[['Ret', stock]].dropna()
  plt = plot_regime_color_new(stock_data, lambda_value=10,log_TR = True,label=stock)

  normal_data = stock_data[stock_data["State"]=="Normal"]
  crash_data = stock_data[stock_data["State"]=="Crash"]

  stock_return = (1+stock_data["Ret"]/100).prod()**(252/len(stock_data))-1
  normal_return = (1+normal_data["Ret"]/100).prod()**(252/len(normal_data))-1
  crash_return = (1+crash_data["Ret"]/100).prod()**(252/len(crash_data))-1

  stock_return = round(stock_return*100,0)
  normal_return = round(normal_return*100,0)
  crash_return = round(crash_return*100,0)

  stock_std = (stock_data["Ret"]/100).std()*np.sqrt(252)*100
  normal_std = (normal_data["Ret"]/100).std()*np.sqrt(252)*100
  crash_std = (crash_data["Ret"]/100).std()*np.sqrt(252)*100

  df = pd.DataFrame({'Normal Regime':[normal_return,normal_std],'Crash Regime':[crash_return,crash_std],'Overall Performance':[stock_return,stock_std]},index=['Annualised Return (%)','Annualised Standard Deviation (%)'])
  #print(df)

  return plt,df

def return_portfolio_regime_graph(stock_list):

   stocks = []

   for i in stock_list:
     stocks.append(i)

   #print(stocks)

   stock_data = build_portfolio_prices_dataframe(stocks,5)
   stock_data["Ret"][0] = 0

   regime_data =  pd.DataFrame()
   regime_data["Ret"] = stock_data["Ret"]*100
   regime_data["Portfolio"] = stock_data["Portfolio"]

   #print(regime_data)

   plt = plot_regime_color_new(regime_data, lambda_value=10,log_TR = True,label="Portfolio")

   #print(regime_data)

   normal_data = regime_data[regime_data["State"]=="Normal"]
   crash_data = regime_data[regime_data["State"]=="Crash"]

   stock_return = (1+regime_data["Ret"]/100).prod()**(252/len(regime_data))-1
   normal_return = (1+normal_data["Ret"]/100).prod()**(252/len(normal_data))-1
   crash_return = (1+crash_data["Ret"]/100).prod()**(252/len(crash_data))-1

   stock_return = round(stock_return*100,0)
   normal_return = round(normal_return*100,0)
   crash_return = round(crash_return*100,0)

   stock_std = (regime_data["Ret"]/100).std()*np.sqrt(252)*100
   normal_std = (normal_data["Ret"]/100).std()*np.sqrt(252)*100
   crash_std = (crash_data["Ret"]/100).std()*np.sqrt(252)*100

   df = pd.DataFrame({'Normal Regime':[normal_return,normal_std],'Crash Regime':[crash_return,crash_std],'Overall Performance':[stock_return,stock_std]},index=['Annualised Return (%)','Annualised Standard Deviation (%)'])
   print(df)

   return plt,df

#plt,df = return_regime_graph('CJLU.SI')
#plt.show()
#print(df)
