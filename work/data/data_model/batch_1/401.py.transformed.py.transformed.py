import os
import numpy as np
import numpy.random as rd
import pandas as pd
import yfinance as yf
class FinanceStockEnv:
    def __init__(self, tickers, initial_stocks, initial_capital=1e6, max_stock=1e2,
                 transaction_fee_percent=1e-3,
                 if_train=True,
                 train_beg=0, train_len=1024):
        tickers = ['AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS',
                   'AXP', 'HD', 'INTC', 'WMT', 'IBM', 'MRK', 'UNH', 'KO', 'CAT',
                   'TRV', 'JNJ', 'CVX', 'MCD', 'VZ', 'CSCO', 'XOM', 'BA', 'MMM',
                   'PFE', 'WBA', 'DD'] if ticker_list is None else ticker_list
        self.num_stocks = len(tickers)
        assert self.num_stocks == len(initial_stocks)
        self.initial_capital = initial_capital
        self.initial_stocks = initial_stocks
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock
        self.preprocess_data(tickers)
        ary = self.load_training_data_for_multi_stock(data_path='./FinanceStock.npy')
        assert ary.shape == (
        1699, 5 * self.num_stocks)
        assert train_beg < train_len
        assert train_len < ary.shape[0]
        self.ary_train = ary[train_beg:train_len]
        self.ary_valid = ary[train_len:]
        self.ary = self.ary_train if if_train else self.ary_valid
        self.day = 0
        self.initial_account__reset = self.initial_capital
        self.account = self.initial_account__reset
        self.day_npy = self.ary[self.day]
        self.stocks = self.initial_stocks
        self.total_asset = self.account + (self.day_npy[:self.num_stocks] * self.stocks).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0
        '''env information'''
        self.env_name = 'FinanceStock-v2'
        self.state_dim = 1 + (5 + 1) * self.num_stocks
        self.action_dim = self.num_stocks
        self.if_discrete = False
        self.target_reward = 1.25
        self.max_step = self.ary.shape[0]
    def reset(self) -> np.ndarray:
        self.initial_account__reset = self.initial_capital * rd.uniform(0.9, 1.1)
        self.account = self.initial_account__reset
        self.stocks = self.initial_stocks
        self.total_asset = self.account + (self.day_npy[:self.num_stocks] * self.stocks).sum()
        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1
        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)
        return state
    def step(self, action) -> (np.ndarray, float, bool, None):
        action = action * self.max_stock
        for index in range(self.num_stocks):
            stock_action = action[index]
            adj = self.day_npy[index]
            if stock_action > 0:
                available_amount = self.account // adj
                delta_stock = min(available_amount, stock_action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:
                delta_stock = min(-stock_action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step
        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)
        next_total_asset = self.account + (self.day_npy[:self.num_stocks] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * 0.99 + reward
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0
            self.episode_return = next_total_asset / self.initial_capital
        return state, reward, done, None
    @staticmethod
    def load_training_data_for_multi_stock(data_path='./FinanceStock.npy'):
        if os.path.exists(data_path):
            data_ary = np.load(data_path).astype(np.float32)
            assert data_ary.shape[1] == 5 * 30
            return data_ary
        else:
            raise RuntimeError(
                f'| Download and put it into: {data_path}\n for FinanceStockEnv()'
                f'| https://github.com/Yonv1943/ElegantRL/blob/master/FinanceMultiStock.npy'
                f'| Or you can use the following code to generate it from a csv file.')
    def preprocess_data(self, tickers):
        df = self.fecth_data(start_date='2009-01-01',
                             end_date='2021-01-01',
                             ticker_list=tickers).fetch_data()
        data = preprocess_data()
        data = add_turbulence(data)
        df = data
        rebalance_window = 63
        validation_window = 63
        i = rebalance_window + validation_window
        unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        train__df = self.data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        train_ary = train__df.to_numpy().reshape((-1, 30, 12))
        data_ary = np.empty((train_ary.shape[0], 5, 30), dtype=np.float32)
        data_ary[:, 0] = train_ary[:, :, 2]
        data_ary[:, 1] = train_ary[:, :, 7]
        data_ary[:, 2] = train_ary[:, :, 8]
        data_ary[:, 3] = train_ary[:, :, 9]
        data_ary[:, 4] = train_ary[:, :, 10]
        data_ary = data_ary.reshape((-1, 5 * 30))
        data_path = './FinanceStock.npy'
        os.makedirs(data_path[:data_path.rfind('/')])
        np.save(data_path, data_ary.astype(np.float16))
        print('| FinanceStockEnv(): save in:', data_path)
        return data_ary
    @staticmethod
    def data_split(df, start, end):
        data = df[(df.date >= start) & (df.date < end)]
        data = data.sort_values(["date", "tic"], ignore_index=True)
        data.index = data.date.factorize()[0]
        return data
    @staticmethod
    def fetch_data(start_date, end_date, ticker_list) -> pd.DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(tic, start=start_date, end=end_date)
            temp_df["tic"] = tic
            data_df = data_df.append(temp_df)
        data_df = data_df.reset_index()
        try:
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            data_df["close"] = data_df["adjcp"]
            data_df = data_df.drop("adjcp", 1)
        except NotImplementedError:
            print("the features are not supported currently")
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        return data_df
