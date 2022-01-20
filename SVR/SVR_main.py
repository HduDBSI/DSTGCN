"""
SVR算法时序预测
"""
from time import process_time
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.accessor import CachedAccessor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor

'''参数'''
# 数据文件名
file_name = "../data/zhengzhou/V_716.csv"
# 道路数
n_route = 716
# 训练序列长度, 预测序列长度
source_frame = 12
target_frame = 3
n_frame = source_frame + target_frame
# 每天的序列长度
day_slot = 3 * 24
# 训练数据的天数, test数据的天数, valid数据的天数
train_days, test_days, valid_days = 5, 1, 1

def MAE(ground_truth, pred) -> float:
    # print(ground_truth.shape, pred.shape)
    # print(ground_truth - pred)
    # return np.sum(np.abs(ground_truth - pred))/len(ground_truth)
    return np.average(np.abs(ground_truth - pred))

def RMSE(ground_truth, pred) -> float:
    # print(ground_truth, pred)
    # print(ground_truth - pred)
    # return np.sqrt(np.sum(pow(ground_truth - pred, 2))/len(ground_truth))
    return np.sqrt(np.average(pow(ground_truth - pred, 2)))

'''
case study的数据
'''
def case_study_data(file_path):
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    pass

    tmp_seq = np.zeros((72, 12+1, 716, 1))

    offset = 72 * 6 - 12
    for i in range(72):
        sta = offset + i
        tmp_seq[i, :, :, :] = data_seq[sta:sta+13].reshape(13, 716, 1)

    return tmp_seq

def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''

    # 每一天分割出的序列数
    n_slot = day_slot - 15 + 1
    # 输出 tmp_seq.shape = [每一天序列数*train/val/test的天数, 每一个序列长度, 传感器个数, 通道数]
    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq

def data_gen_zhengzhou(file_path, n_route, n_frame, day_slot):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''

    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # 生成训练序列
    seq_train = seq_gen(train_days, data_seq, 0, n_frame, n_route, day_slot)
    # 生成验证序列
    seq_val = seq_gen(valid_days, data_seq, train_days, n_frame, n_route, day_slot)
    # 生成测试序列
    seq_test = seq_gen(test_days, data_seq, train_days + valid_days, n_frame, n_route, day_slot)

    return seq_train, seq_val, seq_test

# """
# 使用SVR模型预测
# """
# def forcast():

#     # 加载数据
#     seq_train, seq_val, seq_test = \
#      data_gen_zhengzhou(file_path = file_name, n_route=n_route, n_frame=n_frame, day_slot=day_slot)

#     mae_list = list()
#     rmse_list = list()

#     for idx in tqdm(range(n_route)):
#         # if idx == 3:
#         #     break
#         '''SVR模型, SVR的target必须是1, 因此必须递归预测'''
#         model=SVR(kernel='linear')
#         X_train, Y_train = seq_train[:, :source_frame, idx, 0], seq_train[:, source_frame, idx, 0]
#         model.fit(X_train, Y_train)
        
#         '''测试集上测试'''
#         # 预测若干步
#         mae_steps = 0 #多步预测结果的总mae
#         rmse_steps = 0 #多步预测结果的总mae
#         X_test, ground_truth = seq_test[:, :source_frame, idx, 0], seq_test[:, source_frame, idx, 0]
#         # print(seq_test.shape)
#         for i in range(target_frame):
#             pred = model.predict(X_test)
#             # print(ground_truth.shape, pred.shape)
#             # 计算MAE
#             mae = MAE(ground_truth, pred)
#             rmse = RMSE(ground_truth, pred)
#             mae_steps += mae
#             rmse_steps += rmse

#             if i == target_frame - 1:
#                 break
#             # 将预测结果作为下一步的结果, 添加到X_test
#             X_test[:, :-1] = X_test[:, 1:]
#             X_test[:, -1] = pred
#             ground_truth = seq_test[:, source_frame+i+1, idx, 0]
        
#         mae_list.append(mae_steps/target_frame)
#         # print(mae)
#         rmse_list.append(rmse_steps/target_frame)

#     print(mae_list)
#     print("mae", sum(mae_list)/n_route)
#     print("rmse", sum(rmse_list)/n_route)

"""
使用SVR模型预测, multisteps
"""
def forcast():

    # 加载数据
    seq_train, seq_val, seq_test = \
     data_gen_zhengzhou(file_path = file_name, n_route=n_route, n_frame=n_frame, day_slot=day_slot)

    # 加载case study数据集
    seq_ex = case_study_data(file_path = file_name)

    mae_list = list()
    rmse_list = list()

    preds = list()
    truth = list()

    # '''case study结果'''
    # preds = np.zeros(shape=(72, 716))

    for idx in tqdm(range(n_route)):
        # if idx == 3:
        #     break
        model=MultiOutputRegressor(SVR(kernel='linear'))
        X_train, Y_train = seq_train[:, :source_frame, idx, 0], seq_train[:, source_frame:source_frame + target_frame, idx, 0]
        model.fit(X_train, Y_train)
        
        # '''case study上测试'''
        # X_ex = seq_ex[:, :source_frame, idx, 0]
        # ground_truth =  seq_ex[:, source_frame:source_frame + target_frame, idx, 0]
        # # print(X_ex.shape)
        # pred = model.predict(X_ex)
        # # print(ground_truth.shape, pred.shape)
        # # 计算MAE
        # # mae = MAE(ground_truth, pred)
        # # print(mae)
        # preds[:, idx:idx+1] = pred


        '''测试集上测试'''
        X_test, ground_truth = seq_test[:, :source_frame, idx, 0], seq_test[:, source_frame:source_frame + target_frame, idx, 0]
        # print(seq_test.shape)
        pred = model.predict(X_test)
        # print(ground_truth.shape, pred.shape)
        # 计算MAE
        mae = MAE(ground_truth, pred)
        mae_list.append(mae)
        # print(mae)

        # rmse = RMSE(ground_truth, pred)
        # rmse_list.append(rmse)

        '''RMSE要统一起来一起算才对'''
        preds.append(pred)
        truth.append(ground_truth)

    # print(mae_list)
    print("mae", sum(mae_list)/n_route)

    print("rmse", RMSE(np.array(truth), np.array(preds)))
    # print("rmse", sum(rmse_list)/n_route)

    # '''保存case study结果'''
    # np.savetxt(X=preds, fname="SVR预测结果.txt")

"""画出对每条道路预测结果的误差"""
def plot_MAE4roads(mae_list):
    plt.plot(mae_list)
    plt.savefig("每条道路的平均MAE")
    pass

if __name__ == "__main__":
    # SVR预测
    forcast()