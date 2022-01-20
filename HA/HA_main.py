"""
HA算法时序预测
"""
from tkinter import PROJECTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sqrt

'''参数'''
# 数据文件名
file_name = "../data/zhengzhou/V_716.csv"
# 训练数据的天数, test数据的天数, valid数据的天数
train_days, test_days, valid_days = 5, 1, 1
# 每天序列的长度
day_slot = 72

"""
读取数据, 返回dataframe
params:
    filename:文件名
"""
def get_df(filename):
    df = pd.read_csv(filename, header=None, encoding='utf-8', parse_dates=True)
    return df


"""
计算单个序列的MAE
"""
def MAE(ground_truth, pred) -> float:
    # print(type(ground_truth), type(pred))
    print(ground_truth.shape, pred.shape)
    # return sum([abs(ground_truth[i] - pred[i]) for i in range(seq_len, len(ground_truth))])/seq_len
    return np.average(np.abs(ground_truth-pred))

"""
计算单个序列的RMSE
"""
def RMSE(ground_truth, pred) -> float:
    # print(type(ground_truth), type(pred))
    print(ground_truth.shape, pred.shape)
    # return sqrt(sum([pow(ground_truth[i] - pred[i], 2) for i in range(seq_len, len(ground_truth))])/seq_len)
    return np.sqrt(np.average(pow(ground_truth - pred, 2)))

"""
使用过往数据的平均值作为当前的预测
数据是使用10分钟聚合的
为了包容时间段的偏移, 因此使用30分钟内的均值(前10分钟, 当前10分钟, 后10分钟)
"""
def forcast():
    
    # 读取数据, 返回dataframe
    df = get_df(file_name)
    print(df.shape)
    # 道路数
    road_num = df.shape[1]
    mae_list = list()
    rmse_list = list()

    preds = list()
    truth = list()

    for idx in tqdm(range(road_num)):
        # 读出序列
        road_seq = pd.to_numeric(df[idx])
        # 绘制序列
        # if idx in [k for k in range(10, 100, 10)]:
        #     plt.plot(road_seq)
        #     plt.show()
        # 计算训练序列的均值
        road_seq = road_seq.to_list()
        # 训练序列 0~144*3, 测试序列 144*3~144*4
        road_seq = np.array(road_seq, dtype=np.int64)
        # 分割训练, 测试
        seq_train  = road_seq[:day_slot*train_days]
        seq_valid = road_seq[day_slot*train_days : day_slot*(train_days+valid_days)]
        seq_test = road_seq[day_slot*(train_days+valid_days):]
        # print(seq_train.shape)
        # 训练序列 (day_slot*train_days, 1) -> (day_slot, train_days)
        seq_train = seq_train.reshape((day_slot, train_days))
        # print(seq_train.shape)
        # 计算预测结果
        seq_pred = aggregate_avg(seq_train)
        # 计算MAE
        mae = MAE(seq_test, seq_pred)
        mae_list.append(mae)
        # print(mae)

        
        # rmse = RMSE(seq_test, seq_pred)
        # rmse_list.append(rmse)
        
        '''RMSE要统一起来一起算才对'''
        # print(seq_pred.shape, seq_test.shape)
        preds.append(seq_pred)
        truth.append(seq_test)
    
    # print(mae_list)
    print("mae", sum(mae_list)/road_num)

    # print("rmse", sum(rmse_list)/road_num)
    print("rmse", RMSE(np.array(truth), np.array(preds)))

"""
求聚合的平均值, 相邻时刻平均
params:
    seq_train:训练序列, shape=(144, 3), 144为每日的序列长度, 3为3日
"""
def aggregate_avg(seq_train):

    # 每日的序列长, 日期数
    day_slot, day = seq_train.shape
    # 三日序列的和
    seq_sum = np.sum(seq_train, axis=1)
    # 聚合后的序列
    seq_pred = list()
    for i in range(day_slot):
        # 一天中的第一个记录
        if i == 0:
            seq_pred.append(sum(seq_sum[:1])/day)
        # 一天中的最后一个记录
        elif i == day_slot - 1:
            seq_pred.append(sum(seq_sum[day_slot-1:])/day)
        # 一天中的其他记录
        else:
            seq_pred.append(sum(seq_sum[i-1:i+2])/(day*3))
    return np.array(seq_pred)

if __name__ == "__main__":
    # HA预测
    forcast()
    pass