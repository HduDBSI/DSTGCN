"""
ARIMA算法时序预测
mae 5.456923947388564
"""

import pandas as pd
from utils import timeseries_plot, bucket_avg, preprocess, config_plot, get_df
from Arima import *
import matplotlib.pyplot as plt
# 多进程
import multiprocessing
# 时间
import time
# os
import os
from tqdm import tqdm

# 进程的互斥锁 
lock = multiprocessing.Lock()

'''参数'''
# 数据文件名
filename = "../data/zhengzhou/V_716.csv"
# 道路数
n_route = 30
# 开始的道路
offset = 130
# 训练序列长度, 预测序列长度
source_frame = 12
target_frame = 3
n_frame = source_frame + target_frame
# 每天的序列长度
day_slot = 3 * 24
# 训练数据的天数, test数据的天数, valid数据的天数
train_days, test_days, valid_days = 5, 1, 1
# 进程个数K
K = 3

"""
道路流量ARIMA预测
params:
    idx:道路的下标
"""
def traffic_main(idx):

    # 设置绘图参数
    config_plot()

    # 读取数据, 返回dataframe
    df = get_df(filename)
    # print(df.shape)

    # 第一条道路的序列
    road_seq = pd.to_numeric(df[idx])
    # print(road_seq.shape)
    
    # 训练数据
    road_seq_train = road_seq[:day_slot*train_days]
    # print(road_seq_train.shape)
     
    # time series plot of one-minute sampling rate data
    # timeseries_plot(y=road_seq, color='g', y_label='road_seq')

    ''' "Grid search" of seasonal ARIMA model '''
    ''' the seasonal periodicy 24 hours, i.e. S=24*60/10 = 144 samples '''
    '''一个周期为一天的记录个数'''
    periodicy = 72
    # p d q的取值是0或1
    arima_para = {}
    arima_para['p'] = range(2)
    arima_para['d'] = range(2)
    arima_para['q'] = range(2)
    # # the seasonal periodicy is  24 hours
    seasonal_para = periodicy
    # 构建ARIMA模型

    arima = Arima_Class(arima_para, seasonal_para)

    # 拟合数据
    arima.fit(road_seq_train)

    # Prediction on observed data starting on pred_start
    # observed and prediction starting dates in plots
    plot_start = 0
    pred_start = day_slot*train_days

    # ax = road_seq[plot_start:].plot(label='observed', figsize=(15, 10))
    # ax.set_ylabel("flow")
    # plt.show()

    # One-step ahead forecasts
    # dynamic=False 表示每一步的预测都会使用上这个点之前的所有序列
    # dynamic = False
    # arima.pred(road_seq, plot_start, pred_start, dynamic, ts_label="flow")

    # # # Dynamic forecasts
    # # dynamic=True 表示预测只会使用pred_start之前的数据
    # dynamic = True
    # arima.pred(road_seq, plot_start, pred_start, dynamic, ts_label="flow")

    # Forecasts to unseen future data
    n_steps = day_slot*(valid_days+test_days)
    # mae, rmse = arima.forcast(road_seq, n_steps, ts_label="flow")
    rmse = arima.forcast(road_seq, n_steps, ts_label="flow")
    # print("MAE", mae)
    # print("RMSE", rmse)
    # return mae, rmse
    return rmse

"""
进程类, 执行轨迹ARIMA预测任务
"""
class arima_process(multiprocessing.Process):

    def __init__(self, seq_idx_range, MAE_list, RMSE_list):
        multiprocessing.Process.__init__(self)
        self.seq_idx_range = seq_idx_range
        self.MAE_list = MAE_list
        self.RMSE_list = RMSE_list

    def run(self):
        print("开始进程：", os.getpid())
        main_process_task(self.seq_idx_range, self.MAE_list, self.RMSE_list)
        print("退出进程：", os.getpid())

def main_process_task(seq_idx_range, MAE_list, RMSE_list):
    
    # sum_mae = 0
    # sum_rmse = 0
    for i in range(seq_idx_range[0], seq_idx_range[1]):
        # p, q = traffic_main(i)
        q = traffic_main(i)
        # sum_mae += p
        # sum_rmse += q
        RMSE_list.append(q)

    # print(sum_mae)
    # print(sum_rmse)
    lock.acquire()
    # MAE_list.append(sum_mae)
    # RMSE_list.append(sum_rmse)
    lock.release()


'''多进程'''
def multi_process():

    print("多进程ARIMA")
    start_time = time.time()

    # mae_list, 每个进程计算的mae总和, 多进程默认不能共享全局变量
    MAE_list = multiprocessing.Manager().list()
    RMSE_list = multiprocessing.Manager().list()

    # 创建进程列表
    process_list = list()
    # 序列个数
    SEQ_NUM = n_route
    # 每个进程的序列书
    work = SEQ_NUM//K
    for i in range(K):
        # 计算任务范围
        seq_idx_range = [offset + i*work, offset + (i+1)*work]
        # if i == K-1:
        #     seq_idx_range = [int(i*SEQ_NUM/K), SEQ_NUM]
        # print(seq_idx_range)
        # print(seq_idx_range)
        # 加入进程队列
        process_list.append(arima_process(seq_idx_range, MAE_list, RMSE_list))
        # 进程开始执行任务
        process_list[-1].start()

    # 等待每个进程执行完毕    
    for i in range(K):
        process_list[i].join()

    # print(MAE_list)
    print(RMSE_list)

    # print("mae", sum(MAE_list)/n_route)
    # print("rmse", sum(RMSE_list)/n_route)

    print("消耗时间", time.time() - start_time)
    print("退出主进程")

'''单进程'''
def single_process():

    print("单进程ARIMA")
    sum_mae = 0
    for i in tqdm(range(n_route)):
        sum_mae += traffic_main(i)
    print(sum_mae/n_route)


if __name__ == "__main__":

    # 家用电量消耗数据
    # power_main()

    multi_process()

    # single_process()

