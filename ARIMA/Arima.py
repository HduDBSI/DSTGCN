import itertools
# from typing import get_origin
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math


class Arima_Class:
    def __init__(self, arima_para, seasonal_para):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para['p']
        d = arima_para['d']
        q = arima_para['q']
        # Generate all different combinations of p, q and q triplets
        # 列出pdq取值的所有组合
        self.pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                             for x in list(itertools.product(p, d, q))]

    """
    拟合数据
    params:
        ts:时间序列
    """
    def fit(self, ts):
        warnings.filterwarnings("ignore")
        # 每个元素为 (参数, 季节性参数, AIC)
        # results_list = list()
        # # 网格化参数实验, 分别为ARIMA参数和季节性参数
        # for param in self.pdq:
        #     for param_seasonal in self.seasonal_pdq:
        #         try:
        #             model = sm.tsa.statespace.SARIMAX(ts,
        #                                             order=param,
        #                                             seasonal_order=param_seasonal,
        #                                             enforce_stationarity=False,
        #                                             enforce_invertibility=False)

        #             results = model.fit()

        #             print('ARIMA{}x{}seasonal - AIC:{}'.format(param, 
        #                                                     param_seasonal, results.aic))
        #             results_list.append([param, param_seasonal, results.aic])
        #         except:
        #             continue
        # # 不同参数的结果列表, 
        # results_list = np.array(results_list)
        # # 选出最低的AIC熵对应的参数
        # lowest_AIC = np.argmin(results_list[:, 2])
        # # 打印模型参数
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print('ARIMA{}x{}seasonal with lowest_AIC:{}'.format(
        #     results_list[lowest_AIC, 0], results_list[lowest_AIC, 1], results_list[lowest_AIC, 2]))
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # 模型选择最合适的参数
        # model = sm.tsa.statespace.SARIMAX(ts,
        #                                 order=results_list[lowest_AIC, 0],
        #                                 seasonal_order=results_list[lowest_AIC, 1],
        #                                 enforce_stationarity=False,
        #                                 enforce_invertibility=False)

        # 最合适的参数ARIMA(1, 0, 1)x(0, 1, 1, 144)seasonal with lowest_AIC:932.074
        model = sm.tsa.statespace.SARIMAX(ts,
                                        order=(1, 0, 1),
                                        seasonal_order=(0, 1, 1, 144),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        # 使用新模型拟合数据, disp为是否显示fit信息
        self.final_result = model.fit(disp=False)
        # print('Final model summary:')
        # print(self.final_result.summary().tables[1])
        # print('Final model diagnostics:')
        # self.final_result.plot_diagnostics(figsize=(15, 12))
        # plt.tight_layout()
        # plt.savefig('model_diagnostics.png', dpi=300)
        # plt.show()


    """
    时序预测, 训练数据需要包含整个序列
    params:
        ts:时间序列
        plot_start:绘图开始位置
        pred_start:预测开始位置
        dynamic:动态表示, 预测使用的历史序列范围, True表示只是用pred_start之前的数据
             false表示每一步之前的所有数据
        ts_label:绘图的标签
    """
    def pred(self, ts, plot_start, pred_start, dynamic, ts_label):

        # pred_dynamic = self.final_result.get_prediction(
        #     start=pd.to_datetime(pred_start), dynamic=dynamic, full_results=True)
        pred_dynamic = self.final_result.get_prediction(
            start=pred_start, dynamic=dynamic, full_results=True)
        pred_dynamic_ci = pred_dynamic.conf_int()
        print(ts[plot_start:])
        ax = ts[plot_start:].plot(label='observed', figsize=(15, 10))
        print(pred_dynamic.predicted_mean)
        if dynamic == False:
            pred_dynamic.predicted_mean.plot(
                label='One-step ahead Forecast', ax=ax)
        else:
            pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

        ax.fill_between(pred_dynamic_ci.index,
                        pred_dynamic_ci.iloc[:, 0],
                        pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        # ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(plot_start), ts.index[-1],
        #                  alpha=.1, zorder=-1)
        ax.fill_betweenx(ax.get_ylim(), plot_start, ts.index[-1],
                            alpha=.1, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel(ts_label)
        plt.ylim((0, max(ts)+1))
        plt.legend()
        plt.tight_layout()
        if dynamic == False:
            plt.savefig(ts_label + '_one_step_pred.png', dpi=300)
        else:
            plt.savefig(ts_label + '_dynamic_pred.png', dpi=300)
        plt.show()

    """
    预测, 训练数据可以不包含整个序列
    params:
        ts:时间序列
        n_steps:预测的序列长度
        ts_label:绘图的标签
    """
    def forcast(self, ts, n_steps, ts_label):
        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)

        # Get confidence intervals of forecasts
        # pred_ci = pred_uc.conf_int()
        # ax = ts.plot(label='observed', figsize=(15, 10))
        # pred_uc.predicted_mean.plot(ax=ax, label='Forecast in Future')
        # ax.fill_between(pred_ci.index,
        #                 pred_ci.iloc[:, 0],
        #                 pred_ci.iloc[:, 1], color='k', alpha=.25)
        # ax.set_xlabel('Time')
        # ax.set_ylabel(ts_label)
        # plt.tight_layout()
        # plt.savefig(ts_label + '_forcast.png', dpi=300)
        # plt.legend()
        # plt.show()

        # 计算MAE
        # print(type(ts), type(pred_uc))
        # return MAE(ts.to_list()[len(ts)-n_steps:], pred_uc.predicted_mean.to_list()), RMSE(
        #     ts.to_list()[len(ts) - n_steps:], pred_uc.predicted_mean.to_list())

        return RMSE(ts.to_list()[len(ts) - n_steps:], pred_uc.predicted_mean.to_list())

"""
计算单个序列的MAE, 前一半是valid, 后一半是test
"""
def MAE(ground_truth, pred) -> float:
    # plt.plot(ground_truth)
    # plt.plot(pred)
    # plt.show()
    # print(ground_truth)
    # print(type(ground_truth), type(pred))
    # 只取后一半作为预测
    seq_len = int(len(ground_truth) / 2)
    # 
    return sum([abs(ground_truth[i] - pred[i]) for i in range(seq_len, len(ground_truth))])/seq_len

"""
计算单个序列的RMSE, 前一半是valid, 后一半是test
"""
def RMSE(ground_truth, pred) -> float:
    # plt.plot(ground_truth)
    # plt.plot(pred)
    # plt.show()
    # print(ground_truth)
    # print(type(ground_truth), type(pred))
    # 只取后一半作为预测
    seq_len = int(len(ground_truth) / 2)
    #
    return math.sqrt(sum([pow(ground_truth[i] - pred[i], 2) for i in range(seq_len, len(ground_truth))])/seq_len)