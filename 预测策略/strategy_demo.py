# -*- coding: utf-8 -*-

config_str = {
    'register_center_address': 'localhost:50051',
    'location_ip': '0.0.0.0',
    'grpc_address': 'localhost:50056'
}

config_int = {
    'log_level': 0,
    'heartbeat_every_seconds': 300
}

config_double = {}

from pystrategy import *
# help(strategy)

# st = strategy("python_strategy_demo", config_str, config_int, config_double)

import pandas as pd


def get_bid_cv(bv, sbv, dbp):
    if dbp == 0:
        return bv - sbv
    elif dbp < 0:
        return 0
    else:
        return bv


def get_ask_cv(av, sav, dap):
    if dap == 0:
        return av - sav
    elif dap < 0:
        return av
    else:
        return 0


lags = 5


# min length: test_period + 1
# required fields: AskPrice1,BidPrice1,BidVolume1,AskVolume1,Volume,Turnover,
def mak_OrderImbalance_feature(df_one, contract_multiplier=10, test_period=20):
    try:
        df_one = df_one.copy()
        print('mak_OrderImbalance_feature', df_one)
        df_one["spread"] = df_one["AskPrice1"] - df_one["BidPrice1"]
        df_one["mid_price"] = (df_one['BidPrice1'] + df_one['AskPrice1']) / 2
        df_one["dmid_price"] = df_one["mid_price"].diff()

        # 之后20 个tick的平均价格
        df_one['AveragePriceChange'] = df_one['mid_price'].rolling(test_period).mean().shift(-test_period) - df_one[
            'mid_price']

        df_one["dBid_price"] = df_one['BidPrice1'].diff()
        df_one["dAsk_price"] = df_one['AskPrice1'].diff()

        df = pd.concat(
            [df_one['BidVolume1'], df_one['BidVolume1'].shift(1).fillna(0), df_one["dBid_price"]],
            axis=1)
        df.columns = ['bv', 'sbv', 'dbp']

        df['bid_CV'] = df.apply(lambda row: get_bid_cv(row['bv'], row['sbv'], row['dbp']), axis=1)
        df_one["bid_CV"] = df['bid_CV']

        # 为了方便在同一个row里面计算而组装
        df = pd.concat(
            [df_one['AskVolume1'], df_one['AskVolume1'].shift(1).fillna(0), df_one["dAsk_price"]],
            axis=1)
        df.columns = ['av', 'sav', 'dap']

        df['ask_CV'] = df.apply(lambda row: get_ask_cv(row['av'], row['sav'], row['dap']), axis=1)
        df_one["ask_CV"] = df['ask_CV']
        df_one["VOI"] = df_one["bid_CV"] - df_one["ask_CV"]
        df_one['DeltaVOI'] = df_one.VOI.diff()

        df_one.dropna(inplace=True)

        # 计算 OIR
        df_one.loc[:, "OIR"] = (df_one['BidVolume1'] - df_one['AskVolume1']) / (
                df_one['BidVolume1'] + df_one['AskVolume1'])
        df_one.loc[:, 'DeltaOIR'] = df_one['OIR'].diff()

        df_one.dropna(inplace=True)

        # 计算成交均价
        df_one.loc[:, "dVol"] = df_one['Volume'].diff()
        # 有可能两次tick之间没有成交量，dVol 为0， 后面计算均价会出现 除零 错误, 产生 inf 。
        df_one = df_one[df_one['dVol'] > 0]

        df_one.loc[:, "dTO"] = df_one['Turnover'].diff()

        df_one.loc[:, "AvgTrade_price"] = df_one["dTO"] / df_one["dVol"] / contract_multiplier
        df_one.loc[:, "AvgTrade_price"] = df_one["AvgTrade_price"].fillna(method='ffill').fillna(method='bfill')
        rolling_mean = df_one["mid_price"].rolling(center=False, window=2).mean()
        rolling_mean.iloc[0] = df_one["mid_price"].iloc[0]
        # 计算 Rt  Page_19
        #  The factor Rt, which we call the mid-price basis (MPB)
        df_one.loc[:, "MPB"] = df_one["AvgTrade_price"] - rolling_mean

        # df_one.dropna(inplace=True)

        # In[319]:

        # Run the final Regression model on which trading is based on
        df_one.loc[:, 'VOI0'] = df_one['VOI'] / df_one['spread']
        df_one.loc[:, 'OIR0'] = df_one['OIR'] / df_one['spread']
        df_one.loc[:, 'R0'] = df_one['MPB'] / df_one['spread']

        for i in range(1, lags + 1):
            VOIString = 'VOI' + str(i)
            OIRString = 'OIR' + str(i)

            df_one.loc[:, VOIString] = df_one['VOI'].shift(i) / df_one['spread']
            df_one.loc[:, OIRString] = df_one['OIR'].shift(i) / df_one['spread']

        df_one.dropna(inplace=True)

        return df_one
    except ValueError as err:
        print(err)
        return pd.DataFrame()


import pickle
from hmmlearn.hmm import GMMHMM, GaussianHMM
import datetime

featureList = ['VOI0', 'VOI1', 'VOI2', 'VOI3', 'VOI4', 'VOI5', 'OIR0', 'OIR1', 'OIR2', 'OIR3', 'OIR4', 'OIR5', 'R0']

'''
    BsFlagBuy = '1'
    BsFlagSell = '2'
    BsFlagUnknown = '0'
    DirectionLong = '0'
    DirectionShort = '1'
    OffsetClose = '1'
    OffsetCloseToday = '2'
    OffsetCloseYesterday = '3'
    OffsetOpen = '0'
    OrderStatusCancelled = '3'
    OrderStatusError = '4'
    OrderStatusFilled = '5'
    OrderStatusPartialFilledActive = '7'
    OrderStatusPartialFilledNotActive = '6'
    OrderStatusPending = '2'
    OrderStatusSubmitted = '1'
    OrderStatusUnknown = '0'
    PriceTypeAny = '0'
    PriceTypeBest5 = '2'
    PriceTypeForwardBest = '4'
    PriceTypeLimit = '3'
    PriceTypeReverseBest = '5'
    SideBuy = '0'
    SideSell = '1'
    TimeConditionGFD = '1'
    TimeConditionGTC = '2'
    TimeConditionIOC = '0'
    VolumeConditionAll = '2'
    VolumeConditionAny = '0'
    VolumeConditionMin = '1'

'''


class MyDemoStrategy(strategy):
    def __init__(self, name, config_str, config_int, config_double):
        strategy.__init__(self, name, config_str, config_int, config_double)
        self.name = name
        self.df = data = pd.DataFrame()
        self.last_volumn = 0

        # hc 's tick price is 1
        self.tick_price = 1
        self.take_profit_tick_price = 1 * self.tick_price
        self.stop_loss_tick_price = 5 * self.tick_price

        self.order_id = -1
        self.close_order_id = -1
        self.direction = -1
        self.order_state = OrderStatusUnknown
        self.buy_price = 0
        self.sell_price = 0

        self.UP_STATE = [8, 8]
        self.DOWN_STATE = [0, 0]
        self.instrument_id = "hc2001"
        self.exchange_id = "SHFE"
        self.account_id = "101065"
        self.limit_up_max = 1 + 0.04
        self.limit_down_max = 1 - 0.04

        pkl_filename = "./hmm_model.pkl"
        with open(pkl_filename, 'rb') as filein:
            self.model = pickle.load(filein)


    def init(self):
        print('init')
        self.add_md("ctp")
        self.subscribe("ctp", [self.instrument_id], "")
        self.add_account("ctp", self.account_id, 10000.0)

    def is_price_avaliable(self, quote):
        return quote.last_price > quote.open_price * self.limit_down_max and quote.last_price < quote.open_price * self.limit_up_max


    def on_quote(self, quote):
        print(quote.instrument_id, quote.trading_day, quote.data_time, \
              quote.ask_price[0], quote.ask_volume[0], \
              quote.bid_price[0], quote.bid_volume[0], \
              quote.open_interest, quote.turnover, \
              quote.volume, quote.pre_close_price, \
              quote.open_price
              )

        datetime_index = datetime.datetime.fromtimestamp(quote.data_time / 1000000000.0)
        series = pd.Series({"AskPrice1": quote.ask_price[0], "BidPrice1": quote.bid_price[0], \
                            "AskVolume1": quote.ask_volume[0], "BidVolume1": quote.bid_volume[0], \
                            "Volume": quote.volume, "Turnover": quote.turnover}, name=datetime_index)
        if self.last_volumn != quote.volume:
            self.df = self.df.append(series)
            self.last_volumn = quote.volume
        else:
            return
        print(self.df)
        print('length:', len(self.df))
        if len(self.df) > 30:
            test_set = mak_OrderImbalance_feature(self.df.copy())
            X_test = test_set[featureList]
            print(X_test)
            hidden_states = self.model.predict(X_test)
            # print(hidden_states)
            print("find status:", hidden_states[-1])
            self.handler_statue(hidden_states[-1], quote)
            self.df = self.df[-31:]


    def on_trade(self, trade):
        print('on trade:', trade)


    def on_order(self, order):
        print('on order')
        if order.status == OrderStatusCancelled:
            if order.order_id == self.order_id:
                # cancel the original order
                self.order_id = -1
                self.direction = -1
                self.order_state = OrderStatusUnknown
            elif order.order_id == self.close_order_id:
                # cancel the close order, reclose
                if order.side == SideBuy:  # buy close
                    self.insert_limit_order(order.instrument_id, order.exchange_id, order.account_id,
                                            self.get_last_md(self.instrument_id, '').last_price + self.stop_loss_tick_price,
                                            order.volume,
                                            SideBuy, OffsetCloseToday)  # or use  OffsetClose when not SHFE exchange

                if order.side == SideSell:  # sell close
                    self.insert_limit_order(order.instrument_id, order.exchange_id, order.account_id,
                                            self.get_last_md(self.instrument_id, '').last_price - self.stop_loss_tick_price,
                                            order.volume,
                                            SideSell, OffsetCloseToday)  # or use  OffsetClose when not SHFE exchange
                self.order_id = -1
                self.close_order_id = -1
                self.direction = -1
                self.order_state = OrderStatusUnknown

        if order.status == OrderStatusFilled or order.status == OrderStatusPartialFilledActive:
            if order.order_id == self.order_id:
                self.order_state = OrderStatusFilled
                if order.side == SideBuy:  # buy open
                    self.close_order_id = self.insert_limit_order(order.instrument_id, order.exchange_id, order.account_id,
                                                                  order.limit_price + self.take_profit_tick_price,
                                                                  order.volume_traded,
                                                                  SideSell,
                                                                  OffsetCloseToday)  # or use  OffsetClose when not SHFE exchange
                else:  # sell open
                    self.close_order_id = self.insert_limit_order(order.instrument_id, order.exchange_id, order.account_id,
                                                                  order.limit_price - self.take_profit_tick_price,
                                                                  order.volume_traded,
                                                                  SideBuy,
                                                                  OffsetCloseToday)  # or use  OffsetClose when not SHFE exchange

            elif order.order_id == self.close_order_id:
                # win once, reset all
                self.close_order_id = -1
                self.order_id = -1
                self.direction = -1
                self.order_state = OrderStatusUnknown


        print('on_order', order.order_id, order.status)


    def has_no_order(self):
        return self.order_id == -1


    def buy_open(self, quote):
        if not self.is_price_avaliable(quote):
            return

        print('buy_open')
        self.order_id = \
            self.insert_limit_order(self.instrument_id, self.exchange_id, self.account_id, self.buy_price, 1,
                                    SideBuy, OffsetOpen)
        if self.order_id != -1:
            self.direction = DirectionLong
        else:
            self.buy_price = 0


    def sell_open(self, quote):
        if not self.is_price_avaliable(quote):
            return

        print('sell_open')
        self.order_id = \
            self.insert_limit_order(self.instrument_id, self.exchange_id, self.account_id, self.sell_price, 1,
                                    SideSell, OffsetOpen)
        if self.order_id != -1:
            self.direction = DirectionShort
        else:
            self.sell_price = 0


    def cancel_pending_order(self):
        print('cancel_pending_order')
        if self.order_state == OrderStatusFilled:
            # cancel closing and re close, maybe cancel is later then the exchange.
            self.cancel_order(self.close_order_id)
        else:
            # not fill yet? maybe cancel is later then the exchange.
            self.cancel_order(self.order_id)


    def handler_statue(self, state, quote):
        if self.has_no_order():
            if state in self.UP_STATE:
                self.buy_price = quote.bid_price[0]
                self.buy_open(quote)
            elif state in self.DOWN_STATE:
                self.sell_price = quote.ask_price[0]
                self.sell_open(quote)
        else:
            if state in self.UP_STATE:
                if self.direction == DirectionLong:
                    # same direction do nothing
                    pass
                elif self.direction == DirectionShort:
                    self.cancel_pending_order()
            elif state in self.DOWN_STATE:
                if self.direction == DirectionShort:
                    # same direction do nothing
                    pass
                elif self.direction == DirectionLong:
                    self.cancel_pending_order()


st = MyDemoStrategy("python_strategy_demo", config_str, config_int, config_double)
st.init()
st.run()
print("done")
