
# coding: utf-8

# # 策略发现1 均值回归：短时间成交价上涨或者下跌超过一定幅度后，必然回调
# 
# ### 具体例子: 统计1分钟上涨或者下跌超过15个点tick price,在有限制的时间内，发生回调不少于2个tickprice的概率

# In[280]:

# -*- coding: utf-8 -*-

# get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
import talib
import os
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from numba import *
import numpy as np

mpl.style.use('seaborn-whitegrid')
pd.set_option('display.width', 21250)
pd.set_option('display.max_columns', 21250)
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# ## 获得tick数据

# In[281]:


@jit
def get_tick_csv(tick_file):
    
    print(tick_file)
    df_days_instrument_one_by_one = pd.read_csv(tick_file, header=0,

                                                error_bad_lines=False, low_memory=False,
                                                na_filter=True, verbose=False,
                                                skip_blank_lines=True,
                                                engine='c',
                                                warn_bad_lines=True, chunksize=100000,
                                                iterator=True)
    df = pd.concat(df_days_instrument_one_by_one, ignore_index=True)
    print(df)
    df = df[["logtime", "InstrumentID", "LastPrice", "Volume", "Turnover", "OpenInterest",
             "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", "UpdateTime"]]
    
    index = pd.DatetimeIndex(pd.to_datetime(df['logtime'], format="%Y%m%d%H%M%S%f", errors='coerce'))
    df.set_index(index, inplace= True)
    df.index.set_names(['datetime'], inplace=True)
    df.dropna(inplace=True)
    
    return df


@jit
def remark_data_time_phase(row):
    if (row.UpdateTime >= "09:00:00" and row.UpdateTime < "10:15:00")         or (row.UpdateTime >= "10:30:00" and row.UpdateTime < "11:30:00"):
        return 1

    elif (row.UpdateTime >= "13:30:00" and row.UpdateTime < "15:00:00"):
        return 2
    elif (row.UpdateTime >= "21:00:00" and row.UpdateTime <= "22:59:59"):
        #or (row.UpdateTime >= "00:00:00" and row.UpdateTime <= "02:30:00"):
        return 3
    else:
        return -1

@jit
def remark_date_time_by_index(row):
    timeStr = str(row["logtime"])
    dt = datetime.datetime.strptime(timeStr, '%Y%m%d%H%M%S%f')
    log_datetime = int(dt.strftime('%Y%m%d'))
    return log_datetime

@jit
def secondofday(time):
    t = time.split(':')
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])


# In[300]:
@jit
def cal_max_direction_and_reverse(df_one, instrumentID, idx, next_reverse_tick_number):
    i = int(idx)
    # 一定idx之后的，价格和现在价格的差值
    df_one["mid_price_roll_max"] = df_one["mid_price"].rolling(i).max()
    df_one["mid_price_roll_min"] = df_one["mid_price"].rolling(i).min()
    # 取最大最小
    df_one["mid_price_diff_max"] = df_one["mid_price_roll_max"].shift(-i) - df_one["mid_price"]
    df_one["mid_price_diff_min"] = df_one["mid_price_roll_min"].shift(-i) - df_one["mid_price"]
    
    max_value = df_one["mid_price_diff_max"].max()
    min_value = df_one["mid_price_diff_min"].min()
    #print("mid_price_diff_max" + str(i) + " :", max_value , min_value)
    # 找到最大值的发生位置
    find_max_index = np.where(df_one["mid_price_diff_max"].values == max_value)
    #print("max index", find_index[0])
    touched_index = 0
    # 遍历起始位start_idx, 找到的上升或者下降最大值位置是 start_idx + i
    # 之后从  start_idx + i 到start_idx + i+ next_reverse_tick_number 看回调的位置
    for start_idx in find_max_index[0]:
        #如果数据足够
        #print("start_idx:", start_idx)
        #print("len(df_one):", len(df_one))
        if start_idx < touched_index:
            continue
        if start_idx + i + next_reverse_tick_number < len(df_one):
            # 计算回归的时候使用, next_reverse_tick_number以后的价格，减去现在价格
            #print("start_idx:", start_idx, " end:", start_idx + next_reverse_tick_number)
            #print(df_one[start_idx+ i: start_idx + i+ next_reverse_tick_number][["mid_price", "mid_price_diff_max", "mid_price_diff_min"]] )
            mid_price_reverse = df_one[start_idx + i: start_idx + i + next_reverse_tick_number]["mid_price"] - df_one[start_idx + i: start_idx + i + 1]["mid_price"].values[0]
            touched_index = start_idx + next_reverse_tick_number
            print(instrumentID,",",df_one[start_idx:start_idx+1].index[0],",tick_numbers,",idx,",wait_tick_numbers,",next_reverse_tick_number ,",max_value,", max_value,",goon_up," ,mid_price_reverse.max(), ",reverse_down," ,mid_price_reverse.min(), file =f)
            
            
    # 找到最小值的发生位置
    find_min_index = np.where(df_one["mid_price_diff_min"].values == min_value)
    #print("max index", find_min_index[0])
    touched_index = 0
    # 遍历起始位
    for start_idx in find_min_index[0]:
        #如果数据足够
        #print("start_idx:", start_idx)
        #print("len(df_one):", len(df_one))
        if start_idx < touched_index:
            continue
        if start_idx + i + next_reverse_tick_number < len(df_one):
            # 计算回归的时候使用, next_reverse_tick_number以后的价格，减去现在价格
            #print("start_idx:", start_idx, " end:", start_idx + next_reverse_tick_number)
            #print(df_one[start_idx + i: start_idx + i + next_reverse_tick_number][["mid_price", "mid_price_diff_max", "mid_price_diff_min"]] )
            mid_price_reverse = df_one[start_idx + i: start_idx + i + next_reverse_tick_number]["mid_price"] - df_one[start_idx + i: start_idx + i + 1]["mid_price"].values[0]
            touched_index = start_idx + next_reverse_tick_number
            print(instrumentID,",",df_one[start_idx:start_idx+1].index[0],",tick_numbers,",idx,",wait_tick_numbers,",next_reverse_tick_number ,",min_value,", min_value,",goon_down," ,mid_price_reverse.min(), ",reverse_up," ,mid_price_reverse.max(), file =f)


@jit
def make_max_up_down(tick_file):

    tick_df1 = get_tick_csv(tick_file)

    tick_df1["date_phase"] = tick_df1.apply(remark_date_time_by_index, axis=1)

    # ### 按交易时间来分段

    # In[286]:

    tick_df1["SecondOfDay"] = tick_df1["UpdateTime"].apply(secondofday)

    tick_df1["date_time_phase"] = tick_df1.apply(remark_data_time_phase, axis=1)
    tick_df1 = tick_df1[tick_df1.date_time_phase > 0]  # only keep valid data

    tick_date_phase = tick_df1.groupby(tick_df1["date_phase"])
    for df_phase_date, df_phase_date_ticks in tick_date_phase:
        #print(df_phase_date, df_phase_date_ticks)
        df_phase_date_ticks_cp = df_phase_date_ticks#.copy()
        tick_df1_date_time_phase = df_phase_date_ticks_cp.groupby(df_phase_date_ticks_cp["date_time_phase"])
        for df_phase_date_time, df_phase_date_time_ticks in tick_df1_date_time_phase:
            #print("df_phase_date_time,", df_phase_date_time,",df_phase_date_time_ticks: \n",df_phase_date_time_ticks[:1])
            #todo 本来应该是循环，此处仅适用最后一个时间段
            tick_df1_one = df_phase_date_time_ticks

            if len(tick_df1_one) < 100:
                continue
            # tick_df1_one = tick_df1[(tick_df1["date_time_phase"] == 1) & (tick_df1["date_phase"] == 20191022)]

            # ### 使用少量数据演示

            # In[290]:

            # df_one = tick_df1_one[:1000]
            df_one = tick_df1_one.copy()


            df_one["mid_price"] = (df_one['BidPrice1'] + df_one['AskPrice1']) / 2
            df_one["dmid_price"] = df_one["mid_price"].diff().fillna(0)

            # 最大1,2,3,4,5分钟的连续增长数据
            ranges = list(np.array(range(60 * 2, 60 * 2 * 6, 120)))

            next_reverse_tick_number = 60*2*5


            # plt.figure(1, figsize=(20,10))

            for idx in ranges:
            #for idx in [120]:
                i = int(idx)
                cal_max_direction_and_reverse(df_one, instrumentID, idx, next_reverse_tick_number)

productName = "FG"
f = open(r"C:\Anaconda3\Scripts\fintech\mean_reverse_%s.csv" % (productName), "w")
# In[ ]:
if __name__ == '__main__':

    for parent, dirs, files in os.walk(r"O:\all_tick_instruments\"):
        for file_name in files:
            stock_file_name = os.path.join(parent, file_name)
            ind = file_name.find(".")
            instrumentID = file_name[: ind]
            if instrumentID.startswith("%s"%(productName)):
                print(file_name, instrumentID)
                # print(stock_file_name)



                # instrumentID = "rb2001"
                # In[285]:
                # tick_file1 = r'O:\all_tick_instruments\rb1910.csv'
                tick_file1 = r'O:\all_tick_instruments\mongodb_data_instrumentids\%s.csv' % (instrumentID)
                # tick_file1 = r'O:\fintech\qq_yun\rb2001.csv'
                make_max_up_down(tick_file1)