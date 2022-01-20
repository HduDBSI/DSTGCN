import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
import pandas as pd
from scipy.sparse import linalg


"""
使用了流量转移数据的dataloader
"""
class DataLoader_transition(object):
    def __init__(self, xs, ys, tpls, batch_size, pad_with_last_sample=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            tpl_padding = np.repeat(tpls[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            tpls = np.concatenate([tpls, tpl_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.tpls = tpls

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        tpls = self.tpls[permutation]
        self.xs = xs
        self.ys = ys
        self.tpls = tpls

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                tpl_i = self.tpls[start_ind: end_ind, ...]
                yield (x_i, y_i, tpl_i)
                self.current_ind += 1

        return _wrapper()

class DataLoader(object):
    ''''样本补全会使指标下降'''
    # def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size. 让训练数据能被批次大小整除
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys


    '''打乱顺序'''
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    
    
class DataLoader_cluster(object):
    def __init__(self, xs, ys, xc, yc, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            num_padding = (batch_size - (len(xc) % batch_size)) % batch_size
            x_padding = np.repeat(xc[-1:], num_padding, axis=0)
            y_padding = np.repeat(yc[-1:], num_padding, axis=0)
            xc = np.concatenate([xc, x_padding], axis=0)
            yc = np.concatenate([yc, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, xc, yc = self.xs[permutation], self.ys[permutation], self.xc[permutation], self.yc[permutation]
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_c = self.xc[start_ind: end_ind, ...]
                y_c = self.yc[start_ind: end_ind, ...]
                yield (x_i, y_i, x_c, y_c)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


"""
对称标准化邻接矩阵
和下面asym_adj的区别是, 分别为 D^-1/2*A*D^-1/2, D^-1*A*D^-1
"""
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)    # 采用三元组(row, col, data)的形式存储稀疏邻接矩阵
    rowsum = np.array(adj.sum(1))   # 按行求和得到rowsum, 即每个节点的度
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()   # (行和rowsum)^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.   # isinf部分赋值为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)   # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense() # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)

"""
另一种对称标准化邻接矩阵
"""
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

"""
计算scaled拉普拉斯矩阵, 默认为无向图
"""
def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


'''
加载邻接矩阵
HGCN数据集
params:
    pkl_filename:
    adjtype:
'''
def load_adj(pkl_filename, adjtype):
    # 加载 pkl 文件
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]    # 返回的是 [D^-1/2*A*D^-1/2, D_1^-1/2*A_1*D_1^-1/2], A_1 为 A的转置, JiNan数据集里邻接矩阵不是对称矩阵
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

"""
加载邻接矩阵
PEMS数据集
"""
def pems_load_adj(file_path):
    adj = pd.read_csv(file_path, header=None, dtype=np.float32).values
    return adj

"""
加载郑州数据集邻接矩阵
"""
def zhengzhou_load_adj(file_path):
    # 距离矩阵
    adj = pd.read_csv(file_path, header=None, dtype=np.float32).values
    # 计算 exp(-adj^2/theta^2)
    adj = np.exp(- adj * adj/ np.var(adj))
    # 保留1022*1022
    # adj = adj[:1022, :1022]
    return adj


"""
HGCN数据集的加载
节点速度
"""
def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    #scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    #for category in ['train', 'val', 'test']:
        #data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    #data['scaler'] = scaler
    return data

"""
PEMSD7(M)数据集加载
导入的数据集是csv格式
Params:
    dataset_dir:数据集路径
    batch_size:batch大小
    n_route:图节点个数
    n_his:历史序列长度
    n_frame:历史序列+目标序列长度
    day_slot:每天可以分隔出的记录个数, 288 = 24(小时) * 60(分钟) / 5(间隔分钟)
    valid_batch_size:验证集batch大小
    test_batch_size:测试集batch大小
"""
def pems_load_dataset(dataset_dir, batch_size, n_route=228, n_his=12, n_frame=15, day_slot=288, valid_batch_size= None, test_batch_size=None):

    # 数据集划分, train,val,test所占的天数
    n_train, n_val, n_test = 34, 5, 5
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(dataset_dir, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {dataset_dir}.')

    '''序列划分, 与STGCN-tensorflow一致'''
    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    '''数据的统计信息应设置为所有数据的统计信息'''
    # x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}
    x_stats = {'mean': np.mean(data_seq), 'std': np.std(data_seq)}

    '''标准化'''
    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    data = {}

    # 将数据分为 his 和 target
    data['x_train'], data['y_train'] = x_train[:, :n_his, :, :], x_train[:, n_his:, :, :]
    data['x_val'], data['y_val'] = x_val[:, :n_his, :, :], x_val[:, n_his:, :, :]
    data['x_test'], data['y_test'] = x_test[:, :n_his, :, :], x_test[:, n_his:, :, :]

    # 按照本代码的方式封装
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    #data['scaler'] = scaler
    return data, x_stats

"""
郑州数据集加载
导入的数据集是csv格式
Params:
    dataset_dir:数据集路径
    batch_size:batch大小
    n_route:图节点个数
    n_his:历史序列长度
    n_frame:历史序列+目标序列长度
    day_slot:每天可以分隔出的记录个数, 72 = 24(小时) * 60(分钟) / 20(间隔分钟)
    valid_batch_size:验证集batch大小
    test_batch_size:测试集batch大小
"""
def zhengzhou_load_dataset(dataset_dir, batch_size, n_route = None, n_his = 12, n_frame = 15, day_slot = 72, valid_batch_size = None, test_batch_size = None):

    # 数据集划分, train,val,test所占的天数
    n_train, n_val, n_test = 5, 1, 1
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(dataset_dir, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {dataset_dir}.')

    # '''只保留前1022个节点'''
    # data_seq = data_seq[:, :1022]

    '''序列划分, 与STGCN-tensorflow一致'''
    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    # FIXME 应该是训练集和验证集的统计信息
    '''数据的统计信息应设置为所有数据的统计信息'''
    # x_stats = {'mean': np.mean(data_seq), 'std': np.std(data_seq)}
    x_stats = {'mean': np.mean(data_seq[:(n_train + n_val) * day_slot]),
               'std': np.std(data_seq[:(n_train + n_val) * day_slot])}

    '''标准化'''
    # x_train, x_val, x_test: np.array, [sequence_num, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    data = {}

    # 将数据分为 his 和 target
    data['x_train'], data['y_train'] = x_train[:, :n_his, :, :], x_train[:, n_his:, :, :]
    data['x_val'], data['y_val'] = x_val[:, :n_his, :, :], x_val[:, n_his:, :, :]
    data['x_test'], data['y_test'] = x_test[:, :n_his, :, :], x_test[:, n_his:, :, :]

    # 按照本代码的方式封装
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    #data['scaler'] = scaler
    return data, x_stats

'''
将tpl从list(dict)转为list(list)
'''
def tpl_dict2list(tpl):

    new_tpl = list()

    for i, d in enumerate(tpl):
        tmp = list()
        for k in d.keys():
            x = [v for v in d[k].values()]
            x += [0 for i in range(10 - len(x))]
            tmp.append(x)
        new_tpl.append(tmp)

    return new_tpl

"""
郑州数据集加载, 包含流量转移数据
导入的数据集是csv格式
Params:
    dataset_dir:数据集路径
    tpl_dir:TPL数据路径
    batch_size:batch大小
    n_route:图节点个数
    n_his:历史序列长度
    n_frame:历史序列+目标序列长度
    day_slot:每天可以分隔出的记录个数, 72 = 24(小时) * 60(分钟) / 20(间隔分钟)
    valid_batch_size:验证集batch大小
    test_batch_size:测试集batch大小
"""
def zhengzhou_load_dataset_transition(dataset_dir, tpl_dir, batch_size, n_route = None, n_his = 12, n_frame = 15, day_slot = 72, valid_batch_size = None, test_batch_size = None):

    # 数据集划分, train,val,test所占的天数
    n_train, n_val, n_test = 5, 1, 1
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(dataset_dir, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {dataset_dir}.')

    # 读取TPL数据
    TPL = list()
    '''快速ADGCN模型需要把list(dict)类型TPL转换成矩阵形式TPL'''
    with open(tpl_dir, "rb") as fr:
        TPL = pickle.load(fr)
        # TODO 加工TPL
        TPL = tpl_dict2list(TPL)

    TPL_train = seq_gen_tpl(n_train, TPL, 0, n_frame, day_slot)
    TPL_val = seq_gen_tpl(n_val, TPL, n_train, n_frame, day_slot)
    TPL_test = seq_gen_tpl(n_test, TPL, n_train + n_val, n_frame, day_slot)

    '''序列划分, 与STGCN-tensorflow一致'''
    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.

    '''统计信息只能是训练集和验证集的数据计算'''
    x_stats = {'mean': np.mean(data_seq[:(n_train+n_val)*day_slot]), 'std': np.std(data_seq[:(n_train+n_val)*day_slot])}

    '''标准化'''
    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    data = {}

    # 将数据分为 his 和 target
    data['x_train'], data['y_train'] = x_train[:, :n_his, :, :], x_train[:, n_his:, :, :]
    data['x_val'], data['y_val'] = x_val[:, :n_his, :, :], x_val[:, n_his:, :, :]
    data['x_test'], data['y_test'] = x_test[:, :n_his, :, :], x_test[:, n_his:, :, :]

    data['tpl_train'] = TPL_train[:, :n_his]
    data['tpl_val'] = TPL_val[:, :n_his]
    data['tpl_test'] = TPL_test[:, :n_his]

    # 按照本代码的方式封装
    data['train_loader'] = DataLoader_transition(data['x_train'], data['y_train'], data['tpl_train'], batch_size)
    data['val_loader'] = DataLoader_transition(data['x_val'], data['y_val'], data['tpl_val'], valid_batch_size)
    data['test_loader'] = DataLoader_transition(data['x_test'], data['y_test'], data['tpl_test'], test_batch_size)

    return data, x_stats

"""
case study实验的测试数据生成
无transition
"""
def exp_dataloader(dataset_dir):
    try:
        data_seq = pd.read_csv(dataset_dir, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {dataset_dir}.')
    # 数据集划分, train,val,test所占的天数
    n_train, n_val, n_test = 5, 1, 1
    day_slot = 72
    '''统计信息只能是训练集和验证集的数据计算'''
    x_stats = {'mean': np.mean(data_seq[:(n_train+n_val)*day_slot]), 'std': np.std(data_seq[:(n_train+n_val)*day_slot])}

    '''数据是12->1'''
    tmp_seq = np.zeros((72, 12+1, 716, 1))

    offset = 72 * 6 - 12
    for i in range(72):
        sta = offset + i
        tmp_seq[i, :, :, :] = data_seq[sta:sta+13].reshape(13, 716, 1)

    ex_test = z_score(tmp_seq, x_stats['mean'], x_stats['std'])
    data = {}
    '''这里需要用归一化的数据作为输入'''
    data['ex_x'], data['ex_y'] = ex_test[:, :12, :, :], tmp_seq[:, 12:, :, :]
    data['ex_loader'] = DataLoader(data['ex_x'], data['ex_y'], 72)
    return data, x_stats

"""
case study实验的测试数据生成
有transition
"""
def exp_dataloader_transition(dataset_dir, tpl_dir):
    try:
        data_seq = pd.read_csv(dataset_dir, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {dataset_dir}.')

    # 读取TPL数据
    TPL = list()
    '''快速ADGCN模型需要把list(dict)类型TPL转换成矩阵形式TPL'''
    with open(tpl_dir, "rb") as fr:
        TPL = pickle.load(fr)
        # TODO 加工TPL
        TPL = tpl_dict2list(TPL)

    TPL = np.array(TPL)
    ex_TPL = np.zeros((72, 12+1, 716, 10))

    offset = 72 * 6 - 12
    for i in range(72):
        sta = offset + i
        ex_TPL[i, :, :, :] = TPL[sta:sta+13]

    ex_TPL = np.array(ex_TPL)

    # 数据集划分, train,val,test所占的天数
    n_train, n_val, n_test = 5, 1, 1
    day_slot = 72
    '''统计信息只能是训练集和验证集的数据计算'''
    x_stats = {'mean': np.mean(data_seq[:(n_train+n_val)*day_slot]), 'std': np.std(data_seq[:(n_train+n_val)*day_slot])}

    '''数据是12->1'''
    tmp_seq = np.zeros((72, 12+1, 716, 1))

    offset = 72 * 6 - 12
    for i in range(72):
        sta = offset + i
        tmp_seq[i, :, :, :] = data_seq[sta:sta+13].reshape(13, 716, 1)

    ex_x = z_score(tmp_seq, x_stats['mean'], x_stats['std'])
    data = {}
    '''这里需要用归一化的数据作为输入'''
    data['ex_x'], data['ex_y'] = ex_x[:, :12, :, :], tmp_seq[:, 12:, :, :]
    data['ex_tpl'] = ex_TPL[:, :12]
    data['ex_loader'] = DataLoader_transition(data['ex_x'], data['ex_y'], data['ex_tpl'], 72)
    return data

"""
z_score
"""
def z_score(x, mean, std):
    return (x - mean) / std

"""
预测时需要z-score的逆过程运算
"""
def z_inverse(x, mean, std):
    return x * std + mean

"""
PEMSD7(M)生成序列
Params:
    len_seq:train/val/test的天数
    data_seq:csv读出数据
    offset:train/val/test的起始下标
    n_frame:history+target长度
    n_route:传感器个数
    day_slot:每天的记录个数 = 24*60/interval
"""
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
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq

"""
TPL生成序列, 
返回的是数据类型为dict的ndarray, 实例数 * n_frame * n_route * 1
Params:
    len_seq:train/val/test的天数
    tpl:[dict, dict, .....]
    offset:train/val/test的起始下标
    n_frame:history+target长度
    n_route:传感器个数
    day_slot:每天的记录个数 = 24*60/interval
"""
def seq_gen_tpl(len_seq, tpl, offset, n_frame, day_slot, C_0=1):

    # 每天的实例个数
    n_slot = day_slot - n_frame + 1

    # tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    tpl_instance = list()

    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tpl_instance.append(tpl[sta:end])

    return np.array(tpl_instance)

"""
HGCN数据集的加载
聚类区域流量
"""
def load_dataset_cluster(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        print(category)
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    print("ok")
    for category in ['train_cluster', 'val_cluster', 'test_cluster']:
        print(category)
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['train_loader_cluster'] = DataLoader_cluster(data['x_train'], data['y_train'],data['x_train_cluster'], data['y_train_cluster'], batch_size)
    data['val_loader_cluster'] = DataLoader_cluster(data['x_val'], data['y_val'],data['x_val_cluster'], data['y_val_cluster'], valid_batch_size)
    data['test_loader_cluster'] = DataLoader_cluster(data['x_test'], data['y_test'],data['x_test_cluster'], data['y_test_cluster'], test_batch_size)
    
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    # print("标签长度", len(labels))
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

"""
PEMS数据集metric, x_stats 为数据的均值与标准差
"""
def pems_metric(pred, real, x_stats):
    mae = pems_mae(pred, real, x_stats).item()
    mape = pems_mape(pred, real, x_stats).item()
    rmse = pems_rmse(pred, real, x_stats).item()
    return mae, mape, rmse


"""
PEMS数据MAE计算, x_stats 为数据的均值与标准差
"""
def pems_mae(preds, labels, x_stats):
    # 将数据z-inverse
    preds = z_inverse(preds, x_stats['mean'], x_stats['std'])
    labels = z_inverse(labels, x_stats['mean'], x_stats['std'])
    loss = torch.abs(preds - labels)
    return torch.mean(loss)

"""
PEMS数据MAPE计算, x_stats 为数据的均值与标准差
"""
def pems_mape(preds, labels, x_stats):
    # 将数据z-inverse
    preds = z_inverse(preds, x_stats['mean'], x_stats['std'])
    labels = z_inverse(labels, x_stats['mean'], x_stats['std'])
    loss = torch.abs((preds - labels)/labels)
    return torch.mean(loss)

"""
PEMS数据RMSE计算, x_stats 为数据的均值与标准差
"""
def pems_rmse(preds, labels, x_stats):
    # 样例个数
    example_num = preds.shape[0] * preds.shape[1] * preds.shape[2]
    # 将数据z-inverse
    preds = z_inverse(preds, x_stats['mean'], x_stats['std'])
    labels = z_inverse(labels, x_stats['mean'], x_stats['std'])
    # return torch.sqrt(torch.pow(torch.mean(preds-labels), 2))
    return torch.sqrt(
        torch.sum(torch.pow(preds-labels, 2))
        /(example_num)
    )