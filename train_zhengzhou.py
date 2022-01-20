import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import *
import shutil
import random
import pickle
from util import z_inverse

# 选取显存最大的显卡
import pynvml
pynvml.nvmlInit()
# 最小占用空间, 显卡编号
min_used = sys.maxsize
best_gpu = -1
for i in range(4):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if meminfo.used < min_used:
        min_used = meminfo.used
        best_gpu = i
# 设置显卡的可见性
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
print("使用%d号显卡"%(best_gpu))

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
# parser.add_argument('--device',type=str,default='cpu',help='')

# 双向邻接矩阵, 可能考虑到有向图?
# parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
# 单向邻接矩阵, 可能倒履到无向图
parser.add_argument('--adjtype', type=str, default='transition', help='adj type')
# 序列长度, 训练序列长度?
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--target_length', type=int, default=1, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
# 输入维度?
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
# JiNan num_nodes = 561 cluster_nodes = 20
# XiAN num_nodes = 792 cluster_nodes = 40
# PEMS num_nodes = 228 cluster_nodes = None
# parser.add_argument('--num_nodes', type=int, default=228, help='number of nodes')
# 郑州的节点个数 2000
# parser.add_argument('--num_nodes', type=int, default=1968, help='number of nodes')
parser.add_argument('--num_nodes', type=int, default=716, help='number of nodes')
# 批次大小
parser.add_argument('--batch_size', type=int, default=58, help='batch size')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
# dropout比例
# parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
# 学习率衰减系数
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
# 迭代次数
parser.add_argument('--epochs', type=int, default=100, help='')
# 每多少个epoch打印一次
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--force', type=str, default=False, help="remove params dir", required=False)
# 模型参数保存路径
parser.add_argument('--save', type=str, default='./garage/Zhengzhou', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
# gwnet
# parser.add_argument('--model',type=str,default='gwnet',help='adj type')
# Gated_STGCN
# parser.add_argument('--model', type=str, default='Gated_STGCN', help='adj type')
parser.add_argument('--model', type=str, default='ADGCN', help='adj type')
# parser.add_argument('--model', type=str, default='TGCN', help='adj type')
# Attention model
# parser.add_argument('--model', type=str, default='ASTGCN_Recent', help='adj type')
# H_GCN_wh
# parser.add_argument('--model',type=str,default='H_GCN_wh',help='adj type')
parser.add_argument('--decay', type=float, default=0.95, help='decay rate of learning rate ')

args = parser.parse_args()
##model repertition
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 主训练函数
def main():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # 加载邻接矩阵
    # adj_mx = util.pems_load_adj("./HGCN_data/PEMSD7(M)/hazdzz-w.csv")
    # 加载zhengzhou邻接矩阵(原始矩阵记录的是距离)
    adj_mx = util.zhengzhou_load_adj("./HGCN_data/zhengzhou/W_716.csv")

    # PEMS 数据集, x_stats 包含了数据的均值与标准差
    # dataloader, x_stats = util.pems_load_dataset(dataset_dir="./HGCN_data/PEMSD7(M)/V_228.csv",
    #                                              batch_size=args.batch_size, valid_batch_size=args.batch_size,
    #                                              test_batch_size=args.batch_size)
    # 郑州数据集
    dataloader, x_stats = util.zhengzhou_load_dataset(dataset_dir="./HGCN_data/zhengzhou/V_716.csv",
                                                      n_route=args.num_nodes,
                                                      batch_size=args.batch_size,
                                                      valid_batch_size=args.batch_size,
                                                      test_batch_size=args.batch_size)

    # 预测可视化样例实验的数据
    ex_dataloader, ex_x_stats = util.exp_dataloader(dataset_dir="./HGCN_data/zhengzhou/V_716.csv")

    # 邻接矩阵放到显存上
    supports = [torch.tensor(adj_mx).cuda()]

    print(args)
    if args.model == 'gwnet':
        engine = trainer1(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'ASTGCN_Recent':
        engine = trainer2(args.in_dim, args.seq_length, args.target_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                          )
    elif args.model == 'GRCN':
        engine = trainer3(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    # 传入训练数据的统计信息
    elif args.model == 'Gated_STGCN':
        print("Gated_STGCN")
        engine = trainer4(args.in_dim, args.seq_length, args.target_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                          )
    elif args.model == 'H_GCN_wh':
        engine = trainer5(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )

    elif args.model == 'OGCRNN':
        engine = trainer8(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'OTSGGCN':
        engine = trainer9(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'LSTM':
        engine = trainer10(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay
                           )
    elif args.model == 'GRU':
        engine = trainer11(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay
                           )
    elif args.model == 'TGCN':
        engine = trainer12(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                           )

    # attention dynamic graph convolution network
    elif args.model == 'ADGCN':
        engine = trainer13(args.in_dim, args.seq_length, args.target_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                           )


    # check parameters file
    '''保存网络保存参数'''
    params_path = args.save + "/" + args.model
    if os.path.exists(params_path) and not args.force:
        pass
        # raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))


    '''开始训练'''
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    '''train loss 和 valid loss'''
    tr_loss_ = list()
    va_loss_ = list()

    for i in range(1, args.epochs + 1):
        # if i == 3:
        #     print("epoch 3")
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        # shuffle
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :args.target_length])
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            # if iter % args.print_every == 0 :
            #   log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            #  print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()

        '''验证'''
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :args.target_length])
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(),
                   params_path + "/" + args.model + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

        '''集中输出loss'''
        tr_loss_.append(mtrain_loss)
        va_loss_.append(mvalid_loss)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("train loss", tr_loss_)
    print("valid loss", va_loss_)

    # 最佳模型
    bestid = np.argmin(his_loss)

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    '''test'''
    # 加载模型
    engine.model.load_state_dict(torch.load(
        params_path + "/" + args.model + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    engine.model.eval()

    test_loss = []
    test_mae = []
    test_mape = []
    test_rmse = []

    '''case study实验'''
    for iter, (x, y) in enumerate(ex_dataloader['ex_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        with torch.no_grad():
            mae, rmse, predicts = engine.ex_eval(testx, testy[:, 0, :, :args.target_length])
            print(predicts.shape)
            ''' 保存真实值和预测结果'''
            '''预测值需要反归一化'''
            predicts = z_inverse(predicts, x_stats['mean'], x_stats['std'])
            np.savetxt(X=predicts.squeeze().cpu().numpy(), fname=args.model + "预测结果" + ".txt")
            np.savetxt(X=testy.squeeze().cpu().numpy(), fname=args.model + "真实值" + ".txt")

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        with torch.no_grad():
            metrics = engine.eval(testx, testy[:, 0, :, :args.target_length])
            test_loss.append(metrics[0])
            test_mae.append(metrics[1])
            test_mape.append(metrics[2])
            test_rmse.append(metrics[3])

    mtest_loss = np.mean(test_loss)
    mtest_mae = np.mean(test_mae)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(mtest_mae, mtest_rmse))

    return mtest_mae, mtest_rmse

'''加载邻接表'''
def load_adj_list(file_path):
    adj_list = list()
    with open(file_path, "rb") as fr:
        adj_list = pickle.load(fr)
    return adj_list

'''使用流量转移数据的main函数'''
def main_transiton():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)

    # 加载zhengzhou邻接矩阵(原始矩阵记录的是距离)
    adj_mx = util.zhengzhou_load_adj("./HGCN_data/zhengzhou/W_716.csv")

    # 加载邻接表, 邻接表记录了邻接关系
    adj_list = load_adj_list("./HGCN_data/zhengzhou/small_adjacency_list.pickle")

    # 郑州数据集, 带转移流量
    dataloader, x_stats = util.zhengzhou_load_dataset_transition(dataset_dir="./HGCN_data/zhengzhou/V_716.csv",
                                                                 tpl_dir="./HGCN_data/zhengzhou/TPL.pickle",
                                                                 n_route=args.num_nodes,
                                                                 batch_size=args.batch_size,
                                                                 valid_batch_size=args.batch_size,
                                                                 test_batch_size=args.batch_size)

    # 邻接矩阵放到显存上
    supports = [torch.tensor(adj_mx).cuda()]

    # 预测可视化样例实验的数据
    ex_dataloader_transition = util.exp_dataloader_transition(dataset_dir="./HGCN_data/zhengzhou/V_716.csv",
                                                               tpl_dir="./HGCN_data/zhengzhou/TPL.pickle")

    print(args)
    if args.model == 'gwnet':
        engine = trainer1(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'ASTGCN_Recent':
        engine = trainer2(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                          )
    elif args.model == 'GRCN':
        engine = trainer3(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    # 传入训练数据的统计信息
    elif args.model == 'Gated_STGCN':
        print("Gated_STGCN")
        engine = trainer4(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                          )
    elif args.model == 'H_GCN_wh':
        engine = trainer5(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )

    elif args.model == 'OGCRNN':
        engine = trainer8(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'OTSGGCN':
        engine = trainer9(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'LSTM':
        engine = trainer10(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay
                           )
    elif args.model == 'GRU':
        engine = trainer11(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay
                           )
    elif args.model == 'TGCN':
        engine = trainer12(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                           )

    # attention dynamic graph convolution network
    elif args.model == 'ADGCN':
        engine = trainer13(args.in_dim, args.seq_length, args.target_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats, adj_list
                           )


    # check parameters file
    '''保存网络保存参数'''
    params_path = args.save + "/" + args.model
    if os.path.exists(params_path) and not args.force:
        pass
        # raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))


    '''开始训练'''
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    '''train loss 和 valid loss'''
    tr_loss_ = list()
    va_loss_ = list()


    for i in range(1, args.epochs + 1):
        if i == 60:
            print("epoch 60")
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        # shuffle
        dataloader['train_loader'].shuffle()
        for iter, (x, y, tpl) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            metrics = engine.train(trainx, trainy[:, 0, :, :args.target_length], tpl)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            # if iter % args.print_every == 0 :
            #   log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            #  print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()

        '''验证'''
        for iter, (x, y, tpl) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            '''验证过程中的指标'''
            metrics = engine.eval(testx, testy[:, 0, :, :args.target_length], tpl)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, (t2 - t1)), flush=True)

        if len(va_loss_)==0 or len(va_loss_) > 0 and mvalid_loss < min(va_loss_):
            torch.save(engine.model.state_dict(),
                       params_path + "/" + args.model + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

        '''集中输出loss'''
        tr_loss_.append(mtrain_loss)
        va_loss_.append(mvalid_loss)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("train loss", tr_loss_)
    print("valid loss", va_loss_)

    # 最佳模型
    bestid = np.argmin(his_loss)

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    '''test'''
    # 加载模型
    engine.model.load_state_dict(torch.load(
        params_path + "/" + args.model + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    engine.model.eval()

    test_loss = []
    test_mae = []
    test_mape = []
    test_rmse = []

    '''case study实验'''
    for iter, (x, y, tpl) in enumerate(ex_dataloader_transition['ex_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        with torch.no_grad():
            mae, rmse, predicts = engine.ex_eval(testx, testy[:, 0, :, :args.target_length], tpl)
            print(predicts.shape)
            ''' 保存真实值和预测结果'''
            '''预测值需要反归一化'''
            predicts = z_inverse(predicts, x_stats['mean'], x_stats['std'])
            np.savetxt(X=predicts.squeeze().cpu().numpy(), fname=args.model + "预测结果" + ".txt")
            np.savetxt(X=testy.squeeze().cpu().numpy(), fname=args.model + "真实值" + ".txt")

    for iter, (x, y, tpl) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        with torch.no_grad():
            metrics = engine.eval(testx, testy[:, 0, :, :args.target_length], tpl)
            test_loss.append(metrics[0])
            test_mae.append(metrics[1])
            test_mape.append(metrics[2])
            test_rmse.append(metrics[3])

    mtest_loss = np.mean(test_loss)
    mtest_mae = np.mean(test_mae)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(mtest_mae, mtest_rmse))

    return mtest_mae, mtest_rmse

    # # testing
    # bestid = np.argmin(his_loss)
    # print("最佳模型位于", bestid+1, "epoch")
    # engine.model.load_state_dict(torch.load(
    #     params_path + "/" + args.model + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    # engine.model.eval()
    #
    # outputs = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy.transpose(1, 3)[:, 0, :, :]
    #
    # for iter, (x, y, tpl) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1, 3)
    #     with torch.no_grad():
    #         preds, spatial_at, parameter_adj = engine.model(testx, tpl)
    #         # preds = preds.transpose(1, 3)
    #     # outputs.append(preds.squeeze())
    #     outputs.append(preds)
    #
    # yhat = torch.cat(outputs, dim=0)
    # # yhat = yhat[:realy.size(0), ...]
    #
    # print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    #
    # amae = []
    # amape = []
    # armse = []
    # prediction = yhat
    # '''预测长度为, PEMS设置为 3/6/9'''
    # '''测试集上在不同步的最佳结果'''
    # for i in range(1):
    #     pred = prediction[:, :, i]
    #     # 取整数个batch
    #     num = realy.shape[0] // args.batch_size * args.batch_size
    #     real = realy[:num, :, i]
    #     metrics = util.pems_metric(pred, real, x_stats)
    #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
    #     amae.append(metrics[0])
    #     amape.append(metrics[1])
    #     armse.append(metrics[2])
    #
    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    # torch.save(engine.model.state_dict(), params_path + "/" + args.model + "_exp" + str(args.expid) + "_best_" + str(
    #     round(his_loss[bestid], 2)) + ".pth")
    # prediction_path = params_path + "/" + args.model + "_prediction_results"
    # ground_truth = realy.cpu().detach().numpy()
    # prediction = prediction.cpu().detach().numpy()
    # spatial_at = spatial_at.cpu().detach().numpy()
    # parameter_adj = parameter_adj.cpu().detach().numpy()
    # np.savez_compressed(
    #     os.path.normpath(prediction_path),
    #     prediction=prediction,
    #     spatial_at=spatial_at,
    #     parameter_adj=parameter_adj,
    #     ground_truth=ground_truth
    # )
    #
    # return np.mean(amae), np.mean(armse)


"""
选择出来的15条道路的下标, 用于绘制静态加权邻接矩阵
[312, 405, 407, 417, 418, 419, 455, 563, 588, 667, 668, 669, 670, 671, 672]
"""
def draw_weighted_adjacency_matrix():
    from d2l import torch as d2l
    import matplotlib.pyplot as plt
    selected_roads_idx = [312, 405, 407, 417, 418, 419, 455, 563, 588, 667, 668, 669, 670, 671, 672]
    # 加载加权邻接矩阵
    adj_mx = util.zhengzhou_load_adj("./HGCN_data/zhengzhou/W_716.csv")
    selected_adj_mx = adj_mx[selected_roads_idx]
    selected_adj_mx = selected_adj_mx[:, selected_roads_idx]
    d2l.show_heatmaps(torch.tensor(selected_adj_mx).reshape(1, 1, len(selected_roads_idx), len(selected_roads_idx)),
        xlabel='', ylabel='')
    plt.savefig("加权矩阵.pdf", pad_inches="tight")
    pass

if __name__ == "__main__":

    # t1 = time.time()
    # main()
    # # 是同ADGCN
    # # main_transiton()
    # t2 = time.time()
    # print("Total time spent: {:.4f}".format(t2 - t1))

    # 重复跑若干次
    mae_list = list()
    rmse_list = list()

    for i in range(1):
        # mae, rmse = main()
        mae, rmse = main_transiton()
        mae_list.append(mae)
        rmse_list.append(rmse)

    print("avg_MAE = %.2f, avg_RMSE = %.2f" % (sum(mae_list)/len(mae_list), sum(rmse_list)/len(rmse_list)))

    # 在一次训练结束后清空一个目录 params_path + "/"
    # import shutil

    # shutil.rmtree(args.save + "/" + args.model + "/")  # 能删除该文件夹和文件夹下所有文件

    # '''选择出来的15条道路的下标, 用于绘制静态加权邻接矩阵'''
    # draw_weighted_adjacency_matrix()
