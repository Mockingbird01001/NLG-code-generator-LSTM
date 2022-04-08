import os
import pandas as pd
import numpy as np
import numpy.random as rd
import torch
class StockTradingEnv:
    def __init__(self, cwd='./envs/FinRL', gamma=0.995,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, sell_cost_pct=1e-3,
                 start_date='2008-03-19', end_date='2016-01-01', data_gap=4,
                 ticker_list=None, tech_indicator_list=None, initial_stocks=None, if_eval=False):
        self.min_stock_rate = 0.1
        price_ary, tech_ary = self.load_data(cwd, ticker_list, tech_indicator_list,
                                             start_date, end_date, )
        beg_i, mid_i, end_i = 0, int(32e4), int(528026)
        if if_eval:
            self.price_ary = price_ary[beg_i:mid_i:data_gap]
            self.tech_ary = tech_ary[beg_i:mid_i:data_gap]
        else:
            self.price_ary = price_ary[mid_i:end_i:data_gap]
            self.tech_ary = tech_ary[mid_i:end_i:data_gap]
        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_capital = initial_capital
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.initial_total_asset = None
        self.gamma_reward = 0.0
        self.stock_cd = None
        self.env_name = 'StockTradingEnv-v2'
        self.state_dim = 1 + 3 * stock_dim + self.tech_ary.shape[1]
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_discrete = False
        self.target_return = 2.2
        self.episode_return = 0.0
    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]
        self.stocks = self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
        self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()
        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.stock_cd = np.zeros_like(price)
        state = np.hstack((max(self.amount, 1e4) * (2 ** -12),
                           price,
                           self.stock_cd,
                           self.stocks,
                           self.tech_ary[self.day],
                           )).astype(np.float32) * (2 ** -6)
        return state
    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)
        self.day += 1
        price = self.price_ary[self.day]
        self.stock_cd += 64
        min_action = int(self.max_stock * self.min_stock_rate)
        for index in np.where(actions < -min_action)[0]:
            if price[index] > 0:
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                self.stock_cd[index] = 0
        for index in np.where(actions > min_action)[0]:
            if price[index] > 0:
                buy_num_shares = min(self.amount // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                self.stock_cd[index] = 0
        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * 2 ** -10
        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset
        return state, reward, done, dict()
    def get_state(self, price):
        state = np.hstack((max(self.amount, 1e4) * (2 ** -12),
                           price,
                           self.stock_cd,
                           self.stocks,
                           self.tech_ary[self.day],
                           )).astype(np.float32) * (2 ** -5)
        return state
    def load_data(self, cwd='./envs/FinRL', ticker_list=None, tech_indicator_list=None,
                  start_date='2016-01-03', end_date='2021-05-27'):
        raw_data1_path = f'{cwd}/StockTradingEnv_raw_data1.df'
        raw_data2_path = f'{cwd}/final1.df'
        data_path_array = f'{cwd}/StockTradingEnv_arrays_float16.npz'
        tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'
        ] if tech_indicator_list is None else tech_indicator_list
        ticker_list = [
            'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN',
            'AMZN', 'ASML', 'ATVI', 'BIIB', 'BKNG', 'BMRN', 'CDNS', 'CERN', 'CHKP', 'CMCSA',
            'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'FAST',
            'FISV', 'GILD', 'HAS', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG',
            'JBHT', 'KLAC', 'LRCX', 'MAR', 'MCHP', 'MDLZ', 'MNST', 'MSFT', 'MU', 'MXIM',
            'NLOK', 'NTAP', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'QCOM', 'REGN',
            'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS', 'TTWO', 'TXN', 'VRSN', 'VRTX', 'WBA',
            'WDC', 'WLTW', 'XEL', 'XLNX'
        ] if ticker_list is None else ticker_list
        '''get: train_price_ary, train_tech_ary, eval_price_ary, eval_tech_ary'''
        if os.path.exists(data_path_array):
            load_dict = np.load(data_path_array)
            price_ary = load_dict['price_ary'].astype(np.float32)
            tech_ary = load_dict['tech_ary'].astype(np.float32)
        else:
            processed_df = self.processed_raw_data(raw_data1_path, raw_data2_path,
                                                   ticker_list, tech_indicator_list)
            def data_split(df, start, end):
                data = df[(df.date >= start) & (df.date < end)]
                data = data.sort_values(["date", "tic"], ignore_index=True)
                data.index = data.date.factorize()[0]
                return data
            train_df = data_split(processed_df, start_date, end_date)
            price_ary, tech_ary = self.convert_df_to_ary(train_df, tech_indicator_list)
            price_ary, tech_ary = self.deal_with_split_or_merge_shares(price_ary, tech_ary)
            np.savez_compressed(data_path_array,
                                price_ary=price_ary.astype(np.float16),
                                tech_ary=tech_ary.astype(np.float16), )
        return price_ary, tech_ary
    def processed_raw_data(self, raw_data_path, processed_data_path,
                           ticker_list, tech_indicator_list):
        if os.path.exists(processed_data_path):
            processed_df = pd.read_pickle(processed_data_path)
            print(f"| load data: {processed_data_path}")
        else:
            print("| FeatureEngineer: start processing data (2 minutes)")
            fe = FeatureEngineer(use_turbulence=True,
                                 user_defined_feature=False,
                                 use_technical_indicator=True,
                                 tech_indicator_list=tech_indicator_list, )
            raw_df = self.get_raw_data(raw_data_path, ticker_list)
            processed_df = fe.preprocess_data(raw_df)
            processed_df.to_pickle(processed_data_path)
            print("| FeatureEngineer: finish processing data")
        '''you can also load from csv'''
        return processed_df
    def draw_cumulative_return(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim
        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd
        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device
        state = self.reset()
        episode_returns = list()
        with _torch.no_grad():
            for i in range(self.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
                state, reward, done, _ = self.step(action)
                total_asset = self.amount + (self.price_ary[self.day] * self.stocks).sum()
                episode_return = total_asset / self.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
        import matplotlib.pyplot as plt
        plt.plot(episode_returns)
        plt.grid()
        plt.title('cumulative return')
        plt.xlabel('day')
        plt.xlabel('multiple of initial_account')
        plt.savefig(f'{cwd}/cumulative_return.jpg')
        print(f"| draw_cumulative_return: save in {cwd}/cumulative_return.jpg")
        return episode_returns
    def deal_with_split_or_merge_shares(self, price_ary, tech_ary):
        tech_ary = tech_ary.reshape((tech_ary.shape[0], -1, 7))
        data_idx = list(range(price_ary.shape[1]))
        price_ary = price_ary[:, data_idx]
        tech_ary = tech_ary[:, data_idx, :]
        for j in range(price_ary.shape[1]):
            x = price_ary[:, j]
            x = self.fill_nan_with_next_value(x)
            x_offset0 = x[1:]
            x_offset1 = x[:-1]
            x_delta = np.abs(x_offset0 - x_offset1) / ((x_offset0 * x_offset1) ** 0.5)
            x_where = np.where(x_delta > 0.25)[0]
            for i in x_where:
                x[i + 1:] *= x[i] / x[i + 1]
            price_ary[:, j] = x
        for tech_i1 in range(tech_ary.shape[1]):
            for tech_i2 in range(tech_ary.shape[2]):
                tech_item = tech_ary[:, tech_i1, tech_i2]
                tech_item = self.fill_nan_with_next_value(tech_item)
                tech_ary[:, tech_i1, tech_i2] = tech_item
        tech_ary = tech_ary.reshape((528026, -1))
        return price_ary, tech_ary
    @staticmethod
    def fill_nan_with_next_value(ary):
        x_isnan = np.isnan(ary)
        value = ary[np.where(~x_isnan)[0][0]]
        for k in range(ary.shape[0]):
            if x_isnan[k]:
                ary[k] = value
            value = ary[k]
        return ary
    @staticmethod
    def get_raw_data(raw_data_path, ticker_list):
        if os.path.exists(raw_data_path):
            raw_df = pd.read_pickle(raw_data_path)
            print(f"| load data: {raw_data_path}")
        else:
            print("| YahooDownloader: start downloading data (1 minute)")
            raw_df = YahooDownloader(start_date="2000-01-01",
                                     end_date="2021-01-01",
                                     ticker_list=ticker_list, ).fetch_data()
            raw_df.to_pickle(raw_data_path)
            print("| YahooDownloader: finish downloading data")
        return raw_df
    @staticmethod
    def convert_df_to_ary(df, tech_indicator_list):
        tech_ary = list()
        price_ary = list()
        for day in range(len(df.index.unique())):
            item = df.loc[day]
            tech_items = [item[tech].values.tolist() for tech in tech_indicator_list]
            tech_items_flatten = sum(tech_items, [])
            tech_ary.append(tech_items_flatten)
            price_ary.append(item.close)
        price_ary = np.array(price_ary)
        tech_ary = np.array(tech_ary)
        print(f'| price_ary.shape: {price_ary.shape}, tech_ary.shape: {tech_ary.shape}')
        return price_ary, tech_ary
class StockTradingVecEnv(StockTradingEnv):
    def __init__(self, env_num, **kwargs):
        super(StockTradingVecEnv, self).__init__(**kwargs)
        self.env_num = env_num
        self.initial_capital = np.ones(self.env_num, dtype=np.float32) * self.initial_capital
        self.initial_stocks = np.tile(self.initial_stocks[np.newaxis, :], (self.env_num, 1))
        self.price_ary = np.tile(self.price_ary[np.newaxis, :], (self.env_num, 1, 1))
        self.tech_ary = np.tile(self.tech_ary[np.newaxis, :], (self.env_num, 1, 1))
        self.device = torch.device('cuda')
    def reset_vec(self):
        with torch.no_grad():
            self.day = 0
            price = self.price_ary[:, self.day]
            self.stocks = self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum(axis=1)
            self.total_asset = self.initial_capital
            self.initial_total_asset = self.total_asset
            self.gamma_reward = np.zeros(self.env_num, dtype=np.float32)
            self.stock_cd = np.zeros_like(price)
        return self.get_state(price)
    def step_vec(self, actions):
        with torch.no_grad():
            actions = (actions * self.max_stock).long().detach().cpu().numpy()
            self.day += 1
            price = self.price_ary[:, self.day]
            self.stock_cd += 64
            min_action = int(self.max_stock * self.min_stock_rate)
            for j in range(self.env_num):
                for index in np.where(actions[j] < -min_action)[0]:
                    if price[j, index] > 0:
                        sell_num_shares = min(self.stocks[j, index], -actions[j, index])
                        self.stocks[j, index] -= sell_num_shares
                        self.amount[j] += price[j, index] * sell_num_shares * (1 - self.sell_cost_pct)
                        self.stock_cd[j, index] = 0
                for index in np.where(actions[j] > min_action)[0]:
                    if price[j, index] > 0:
                        buy_num_shares = min(self.amount[j] // price[j, index], actions[j, index])
                        self.stocks[j, index] += buy_num_shares
                        self.amount[j] -= price[j, index] * buy_num_shares * (1 + self.buy_cost_pct)
                        self.stock_cd[j, index] = 0
            state = self.get_state(price)
            total_asset = self.amount + (self.stocks * price).sum(axis=1)
            reward = (total_asset - self.total_asset) * 2 ** -10
            self.total_asset = total_asset
            self.gamma_reward = self.gamma_reward * self.gamma + reward
            done = self.day == self.max_step
            if done:
                reward = self.gamma_reward
                self.episode_return = total_asset / self.initial_total_asset
                state = self.reset_vec()
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.as_tensor([done, ] * self.env_num, dtype=torch.float32, device=self.device)
        return state, reward, done, dict()
    def get_state(self, price):
        state = np.hstack((self.amount[:, np.newaxis].clip(None, 1e4) * (2 ** -12),
                           price,
                           self.stock_cd,
                           self.stocks,
                           self.tech_ary[:, self.day],
                           ))
        return torch.as_tensor(state, dtype=torch.float32, device=self.device) * (2 ** -5)
    def check_vec_env(self):
        self.reset_vec()
        a = torch.zeros((self.env_num, self.action_dim), dtype=torch.float32, device=self.device)
        s, r, d, _ = self.step_vec(a)
        print(s.shape)
        print(r.shape)
        print(d.shape)
def check_stock_trading_env():
    if_eval = True
    env = StockTradingEnv(if_eval=if_eval)
    action_dim = env.action_dim
    state = env.reset()
    print('| check_stock_trading_env, state_dim', len(state))
    from time import time
    timer = time()
    policy_name = 'Random Action 1e-2'
    step = 1
    done = False
    episode_return = 0
    while not done:
        action = rd.uniform(-1, 1, size=action_dim) * 1e-2
        next_state, reward, done, _ = env.step(action)
        episode_return += reward
        step += 1
    print()
    print(f"| {policy_name}:")
    print(f"| step {step}, UsedTime {time() - timer:.3e}")
    print(f"| gamma_reward \t\t\t{env.gamma_reward:.3e}")
    print(f"| episode return \t\t{episode_return:.3e}")
    print(f"| discount return \t\t{episode_return / step / (1 - env.gamma):.3e}")
    print(f"| env episode return \t{env.episode_return:.3e}")
    policy_name = 'Buy  4 Action'
    step = 1
    done = False
    episode_return = 0
    while not done:
        action = np.zeros(action_dim)
        action[:3] = 1
        next_state, reward, done, _ = env.step(action)
        episode_return += reward
        step += 1
    print()
    print(f"| {policy_name}:")
    print(f"| step {step}, UsedTime {time() - timer:.3e}")
    print(f"| gamma_reward \t\t\t{env.gamma_reward:.3e}")
    print(f"| episode return \t\t{episode_return:.3e}")
    print(f"| discount return \t\t{episode_return / step / (1 - env.gamma):.3e}")
    print(f"| env episode return \t{env.episode_return:.3e}")
    '''draw_cumulative_return'''
    from elegantrl2.agent import AgentPPO
    from elegantrl2.run import Arguments
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.env = StockTradingEnv(if_eval=True)
    args.if_remove = False
    args.cwd = './StockTradingEnv-v1_AgentPPO'
    args.init_before_training()
    env.draw_cumulative_return(args, torch)
class YahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
    def fetch_data(self) -> pd.DataFrame:
        import yfinance as yf
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
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
class FeatureEngineer:
    def __init__(
            self,
            use_technical_indicator=True,
            tech_indicator_list=None,
            use_turbulence=False,
            user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
    def preprocess_data(self, df):
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df
    def add_technical_indicator(self, data):
        from stockstats import StockDataFrame as Sdf
        df = data.copy()
        df = df.sort_values(by=['tic', 'date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()
        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        df = df.sort_values(by=['date', 'tic'])
        return df
    def add_turbulence(self, data):
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df
    @staticmethod
    def add_user_defined_feature(data):
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        return df
    @staticmethod
    def calculate_turbulence(data):
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        df_price_pivot = df_price_pivot.pct_change()
        unique_date = df.date.unique()
        start = 252
        turbulence_index = [0] * start
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
                ]
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)
            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index
if __name__ == '__main__':
    check_stock_trading_env()
