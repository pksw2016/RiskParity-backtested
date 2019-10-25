import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import Strategy,Portfolio
from Risk_Parity_v1 import _get_risk_parity_weights




class RiskParity(Strategy):

    def __init__(self,names,df,date_list):
        self.names = names
        self.df = df
        self.date_list = date_list

    def generate_signals(self):
        self.df = self.df.resample('M').last()
        self.sig = pd.DataFrame(index = self.date_list)

        for index,x in enumerate(self.names):
            self.sig[x] = 1

        return self.sig


class MarketOnClosePortfolio(Portfolio):

    def __init__(self,df,names,sig,date_list):
        self.df = df
        self.names = names
        self.sig = sig
        self.date_list = date_list
        self.positions = self.generate_positions()


    def generate_positions(self):
        pos = pd.DataFrame(index=self.sig.index)
        weights = self.generate_risk_parity_positions()
        pos = weights.mul(self.sig.values)
        return pos

    def generate_risk_parity_positions(self):
        df_m = self.df.resample('BM').last()

        # Get the start date
        #lookbackperiod = 6
        #start_date = pd.date_range(df_m.index[0], periods=lookbackperiod, freq='M')[lookbackperiod - 1]
        #date_list = df_m[start_date:].index.tolist()

        import datetime
        from dateutil.relativedelta import relativedelta

        weights_all = []
        # cov
        for dts in self.date_list:
            lb_startdate = dts + relativedelta(months=-6)
            covariances = 252 * self.df[lb_startdate:dts].pct_change(1).dropna().cov().values
            assets_risk_budget = [1.0 / self.df.shape[1]] * self.df.shape[1]
            init_weights = [1.0 / self.df.shape[1]] * self.df.shape[1]
            weights = _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)
            weights_all.append(weights)

        df = pd.DataFrame(weights_all, index=date_list)
        return df

    def backtest_portfolio(self):
        self.df = self.df.resample('BM').last()
        self.df = self.df.reindex(self.sig.index)
        pct_chg = self.df.pct_change()

        rtns = pd.DataFrame(index=self.sig.index)
        rtns = pct_chg.mul(self.positions.values)
        rtns_comb = rtns.sum(axis=1)
        return rtns_comb

if __name__ == '__main__':


    data = pd.read_csv('spx_comp_longhist.csv', index_col='date', parse_dates=True)
    df = data['2005':]
    mask = df.isnull().sum() < 10
    stock_list = mask[mask == 1].index
    df = df[stock_list]

    stocks = ['MSFT', 'AMZN', 'AAPL']  # 'AAPL', 'AMZN', 'BA', 'BAC', 'C', 'CSCO', 'CVX', 'DIS', 'HD', 'INTC',
                                        # 'JNJ', 'JPM', 'KO', 'MSFT', 'ORCL', 'PEP', 'PFE', 'PG', 'T', 'UNH',
                                         # 'VZ', 'WFC', 'WMT', 'XOM'
    df = df[stocks]
    all_list = []
    for i,x in enumerate(stocks):
        al = str(stocks[i])
        all_list.append(al)


    df_m = df.resample('BM').last()

    # Get the start date
    lookbackperiod = 6
    start_date = pd.date_range(df_m.index[0], periods=lookbackperiod, freq='BM')[lookbackperiod - 1]
    date_list = df_m[start_date:].index.tolist()


    rp = RiskParity(stocks,df,date_list)
    sig = rp.generate_signals()

    port = MarketOnClosePortfolio(df,stocks,sig,date_list)
    returns = port.backtest_portfolio()

    cum_returns = np.cumproduct(returns+1)-1
    cum_returns = pd.DataFrame(cum_returns)
    cum_returns.columns = [str(all_list)+str('Risk Parity')]

    dfri = df.reindex(cum_returns.index)
    df_chg = dfri.resample('BM').last().pct_change()

    df_chg.replace(np.nan, 0, inplace=True)
    df_chg_cum = np.cumproduct(df_chg + 1) - 1

    cum_returns = pd.concat([cum_returns, df_chg_cum], axis=1)

    print(cum_returns)
    cum_returns.plot()
    plt.legend(cum_returns.columns)
    plt.title('Allocation returns')

    plt.show()