# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018

@author: gk
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d, Embedding, Linear
from sparse_activations import Sparsemax

"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze(1)#b,l,n
        
        c2 = seq.permute(0, 2, 1, 3)#b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(1)#b,c,l
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2)+self.b)
        logits = torch.matmul(self.v, logits)
        ##normalization
        a,_ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        return coefs
    
class SATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(SATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        
        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self,seq):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq.permute(0, 3, 1, 2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2)+self.b)
        logits = torch.matmul(self.v, logits)
        ##normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        return coefs

class cheby_conv_ds(nn.Module):
    def __init__(self, c_in, c_out, K):
        super(cheby_conv_ds, self).__init__()
        c_in_new = (K)*c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1, 1), bias=True)
        self.K = K
        
    def forward(self, x, adj, ds):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L0 = torch.eye(nNode).cuda()
        L1 = adj
    
        L = ds*adj
        I = ds*torch.eye(nNode).cuda()
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            L3 = ds*L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        # 通道变换其实没有邻居节点信息聚合的过程
        out = self.conv1(x)
        return out

    
###ASTGCN_block
class ST_BLOCK_0(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_0, self).__init__()

        '''这里原来论文没有先进行卷积'''
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT = TATT(c_in, num_nodes, tem_size)
        self.SATT = SATT(c_in, num_nodes, tem_size)
        self.dynamic_gcn = cheby_conv_ds(c_in, c_out, K)
        self.K = K
        # padding 是一个tuple的话, 表示高度与宽度上的padding, 这里设置的padding模式没有修改时序卷积后的长度
        self.time_conv = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, 1),
                          stride=(1, 1), bias=True)
        # self.time_conv = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, 0),
        #                   stride=(1, 1), bias=True)
        self.residual_conv = Conv2d(c_in, c_out, kernel_size=(1, 1),
                                    stride=(1, 1))
        #self.bn=BatchNorm2d(c_out)
        # self.bn = LayerNorm([c_out, num_nodes, tem_size])
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        # x_input = self.conv1(x)
        x_input = x
        # 计算时间注意力系数
        T_coef = self.TATT(x)
        T_coef = T_coef.transpose(-1, -2)
        # 使用时间注意力系数加权
        x_TAt = torch.einsum('bcnl,blq->bcnq', x, T_coef)

        # 计算空间注意力系数, 这里计算出来的是不对称的
        # TODO ASTGCN的作者, 在这里输入是x_TAt
        # S_coef = self.SATT(x)#B x N x N
        S_coef = self.SATT(x_TAt)  # B x N x N

        # 空间卷积
        # spatial_gcn = self.dynamic_gcn(x_TAt, supports, S_coef)
        spatial_gcn = self.dynamic_gcn(x, supports, S_coef)
        spatial_gcn = torch.relu(spatial_gcn)
        time_conv_output = self.time_conv(spatial_gcn)
        '''x_input与time_conv_output对齐'''
        x_input = self.residual_conv(x_input)
        x_input = x_input[:, :, :, :time_conv_output.shape[-1]]

        '''这里用LayerNorm没什么道理,为啥要让不同样本的同一维度归一化呢'''
        # out = self.bn(torch.relu(time_conv_output + x_input))
        out = torch.relu(time_conv_output + x_input)

        return out, S_coef, T_coef
     


###1
###DGCN_Mask&&DGCN_Res
class T_cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        Lap = Lap.transpose(-1,-2)
        #print(Lap)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_1(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_1,self).__init__()
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        self.dynamic_gcn=T_cheby_conv(c_out,2*c_out,K,Kt)
        self.K=K
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,supports)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,supports,T_coef
        
    
###2    
##DGCN_R  
class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv_ds,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 


    
class SATT_2(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_2,self).__init__()
        self.conv1=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.bn=LayerNorm([num_nodes,num_nodes,12])
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits,-1)
        return logits
  

class TATT_1(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT_1,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn=BatchNorm1d(tem_size)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        #print(c2.shape)
        f2 = self.conv2(c2).squeeze()#b,c,n
         
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        logits = logits.permute(0,2,1).contiguous()
        logits=self.bn(logits).permute(0,2,1).contiguous()
        coefs = torch.softmax(logits,-1)
        return coefs   


class ST_BLOCK_2_r(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_2_r,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        
        self.SATT_2=SATT_2(c_out,num_nodes)
        self.dynamic_gcn=T_cheby_conv_ds(c_out,2*c_out,K,Kt)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
        self.K=K
        self.tem_size=tem_size
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        S_coef=self.SATT_2(x_1)
        shape=S_coef.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) #2020/3/28/22:17,试验下效果
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        adj_out1=(adj_out)*supports
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,adj_out1)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,adj_out,T_coef

###DGCN_GAT
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features,out_features,length,Kt, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.length = length
        self.alpha = alpha
        self.concat = concat
        
        self.conv0=Conv2d(self.in_features, self.out_features, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        
        self.conv1=Conv1d(self.out_features*self.length, 1, kernel_size=1,
                          stride=1, bias=False)
        self.conv2=Conv1d(self.out_features*self.length, 1, kernel_size=1,
                          stride=1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, input, adj):
        '''
        :param input: 输入特征 (batch,in_features,nodes,length)->(batch,in_features*length,nodes)
        :param adj:  邻接矩阵 (batch,batch)
        :return: 输出特征 (batch,out_features)
        '''
        input=self.conv0(input)
        shape=input.shape
        input1=input.permute(0,1,3,2).contiguous().view(shape[0],-1,shape[2]).contiguous()
        
        f_1=self.conv1(input1)
        f_2=self.conv1(input1)
        
        logits = f_1 + f_2.permute(0,2,1).contiguous()
        attention = F.softmax(self.leakyrelu(logits)+adj, dim=-1)  # (batch,nodes,nodes)
        #attention1 = F.dropout(attention, self.dropout, training=self.training) # (batch,nodes,nodes)
        attention=attention.transpose(-1,-2)
        h_prime = torch.einsum('bcnl,bnq->bcql',input,attention) # (batch,out_features)        
        return h_prime,attention

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads,length,Kt):
        """
        Dense version of GAT.
        :param nfeat: 输入特征的维度
        :param nhid:  输出特征的维度
        :param nclass: 分类个数
        :param dropout: dropout
        :param alpha: LeakyRelu中的参数
        :param nheads: 多头注意力机制的个数
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid,length=length,Kt=Kt, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        #self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        fea=[]
        for att in self.attentions:
            f,S_coef=att(x, adj)
            fea.append(f)
        x = torch.cat(fea, dim=1)
        #x = torch.mean(x,-1)
        return x,S_coef



"""
Gated-STGCN(IJCAI)
切比雪夫多项式拟合
"""
class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L0 = torch.eye(nNode).cuda()
        # L0 = torch.eye(nNode)
        L1 = torch.cat([i for i in adj]).reshape(nNode, nNode)
        #L1 = adj
        Ls.append(L0)
        Ls.append(L1)
        # 阶数设置
        for k in range(2, self.K):
            # L2 = 2 *torch.matmul( adj, L1) - L0
            L2 = 2 * torch.matmul(L1, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out

"""
inception-门控时序卷积层
params:
    in_channels:输入通道大小
    out_channels:输出通道大小
    kernel_size:时序卷积核大小
"""
class TimeBlock_inception(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock_inception, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # GLU 使用的卷积核, 卷积核长度分别为 3,5,7
        self.conv_glu_3 = nn.Conv2d(out_channels, 2 * out_channels, (1, 3), padding=(0, 1))
        self.conv_glu_5 = nn.Conv2d(out_channels, 2 * out_channels, (1, 5), padding=(0, 2))
        self.conv_glu_7 = nn.Conv2d(out_channels, 2 * out_channels, (1, 7), padding=(0, 3))

        # 三个卷积结果的权重
        self.r3 = nn.Parameter(torch.randn(size=(1, 1)))
        self.r5 = nn.Parameter(torch.randn(size=(1, 1)))
        self.r7 = nn.Parameter(torch.randn(size=(1, 1)))

        # 下采样卷积核
        self.conv_down = nn.Conv2d(in_channels, out_channels, (1, 1), )

        # inception cat后的结果, 下采样
        self.ineption_down = nn.Conv2d(out_channels, out_channels, (1, 1))

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions. convert from NHWC to NCHW
        X = X.permute(0, 3, 1, 2)

        '''模仿原作者的代码, 加入通道对齐'''
        c_in = X.shape[1]

        # 等于不需要操作
        if c_in == self.out_channels:
            X_input = X
        # 小于补上0
        elif c_in < self.out_channels:
            X_input = torch.cat((X, torch.zeros([X.shape[0], self.out_channels-c_in, X.shape[2], X.shape[3]]).cuda()), dim=1)
        # 大于先下采样
        else:
            X_input = self.conv_down(X)

        '''inception结构'''
        # 卷积核为3
        temp_3 = self.conv_glu_3(X_input)
        P_3 = temp_3[:, :self.out_channels, :, :]
        Q_3 = temp_3[:, self.out_channels:, :, :]
        # 使用残差连接
        out_3 = (P_3 + X_input) * torch.sigmoid(Q_3)

        # 卷积核为5
        temp_5 = self.conv_glu_5(X_input)
        P_5 = temp_5[:, :self.out_channels, :, :]
        Q_5 = temp_5[:, self.out_channels:, :, :]
        # 使用残差连接
        out_5 = (P_5 + X_input) * torch.sigmoid(Q_5)

        # 卷积核为7
        temp_7 = self.conv_glu_7(X_input)
        P_7 = temp_7[:, :self.out_channels, :, :]
        Q_7 = temp_7[:, self.out_channels:, :, :]
        # 使用残差连接
        out_7 = (P_7 + X_input) * torch.sigmoid(Q_7)

        # 按照通道拼接
        # out = torch.cat([out_3, out_5, out_7], dim=1)
        # out = torch.cat([out_3, out_5], dim=1)
        # out = out_3

        # 通道下采样
        # out = self.ineption_down(out)

        # 三类卷积结果直接相加
        # out = out_3 + out_5 + out_7

        # 三类卷积结果加权
        # out = torch.relu(self.r3 * out_3 + self.r5 * out_5 + self.r7 * out_7)

        '''这种简单加权的方式取得了最好的效果'''
        out = self.r3 * out_3 + self.r5 * out_5 + self.r7 * out_7

        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

"""
STGCN中STBLOCK的门控时序卷积层, 使用padding, 意思是在卷积过程中, 时间序列长度不变
params:
    in_channels:输入通道大小
    out_channels:输出通道大小
    kernel_size:时序卷积核大小
"""
class TimeBlock_padding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock_padding, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # GLU 使用的卷积核, 通道变为以前的两倍, 一维卷积
        self.conv_glu = nn.Conv2d(out_channels, 2*out_channels, (1, kernel_size), padding=(0, 1))

        # 下采样卷积核
        self.conv_down = nn.Conv2d(in_channels, out_channels, (1, 1), )

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions. convert from NHWC to NCHW
        X = X.permute(0, 3, 1, 2)

        '''模仿原作者的代码, 加入通道对齐'''
        c_in = X.shape[1]

        # 等于不需要操作
        if c_in == self.out_channels:
            X_input = X
        # 小于补上0
        elif c_in < self.out_channels:
            X_input = torch.cat((X, torch.zeros([X.shape[0], self.out_channels-c_in, X.shape[2], X.shape[3]]).cuda()), dim=1)
        # 大于先下采样
        else:
            X_input = self.conv_down(X)

        # 通道变为两倍
        temp = self.conv_glu(X_input)
        P = temp[:, :self.out_channels, :, :]
        Q = temp[:, self.out_channels:, :, :]
        # 使用残差连接
        # out = (P + X_input[:, :, :, self.kernel_size-1:]) * torch.sigmoid(Q)
        out = (P + X_input) * torch.sigmoid(Q)

        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

"""
STGCN中STBLOCK的门控时序卷积层
params:
    in_channels:输入通道大小
    out_channels:输出通道大小
    kernel_size:时序卷积核大小
"""
class TimeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # GLU 使用的卷积核, 通道变为以前的两倍, 一维卷积
        self.conv_glu = nn.Conv2d(out_channels, 2*out_channels, (1, kernel_size))

        # 下采样卷积核
        self.conv_down = nn.Conv2d(in_channels, out_channels, (1, 1), )

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions. convert from NHWC to NCHW
        X = X.permute(0, 3, 1, 2)

        '''模仿原作者的代码, 加入通道对齐'''
        c_in = X.shape[1]

        # 等于不需要操作
        if c_in == self.out_channels:
            X_input = X
        # 小于补上0
        elif c_in < self.out_channels:
            X_input = torch.cat((X, torch.zeros([X.shape[0], self.out_channels-c_in, X.shape[2], X.shape[3]]).cuda()), dim=1)
        # 大于先下采样
        else:
            X_input = self.conv_down(X)

        # 通道变为两倍
        temp = self.conv_glu(X_input)
        P = temp[:, :self.out_channels, :, :]
        Q = temp[:, self.out_channels:, :, :]
        # 使用残差连接
        out = (P + X_input[:, :, :, self.kernel_size-1:]) * torch.sigmoid(Q)
        # out = P * torch.sigmoid(Q)

        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

# """
# STGCN中的st-block
# params:
#     c_in:输入维度
#     c_out:输出维度
#     num_nodes:图节点
#     tem_size:训练序列长度
#     K:空间卷积核大小
#     Kt:时序卷积核大小
# """
# class ST_BLOCK_4(nn.Module):
#
#     def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
#         super(ST_BLOCK_4,self).__init__()
#         # self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
#         #                   stride=(1,1), bias=True)
#         # 不要padding
#         self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, Kt), stride=(1,1), bias=True)
#         # 两个反斜杠表示只输出小数点前的数字, 改为不进行通道变换
#         self.gcn = cheby_conv(c_out//2, c_out, K, 1)
#         # 不要padding
#         # self.conv2 = Conv2d(c_out, c_out*2, kernel_size=(1, Kt), padding=(0,1), stride=(1,1), bias=True)
#         self.conv2 = Conv2d(c_out, c_out * 2, kernel_size=(1, Kt), stride=(1, 1), bias=True)
#         self.c_out = c_out
#         # self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1,1), bias=True)
#         self.conv_1 = Conv2d(c_in, c_out//2, kernel_size=(1, 1), stride=(1, 1), bias=True)
#         #self.conv_2=Conv2d(c_out//2, c_out, kernel_size=(1, 1),
#           #                stride=(1,1), bias=True)
#
#
#     def forward(self,x,supports):
#         # 通道变换, 用于残差连接
#         x_input1 = self.conv_1(x)[:,:,:,2:]
#         #
#         x1 = self.conv1(x)
#         filter1, gate1 = torch.split(x1, [self.c_out//2, self.c_out//2], 1)
#         x1 = (filter1+x_input1) * torch.sigmoid(gate1)
#         x2 = self.gcn(x1, supports)
#         x2 = torch.relu(x2)
#         #x_input2=self.conv_2(x2)
#         x3 = self.conv2(x2)
#         filter2, gate2 = torch.split(x3, [self.c_out, self.c_out], 1)
#         # x = (filter2 + x_input1) * torch.sigmoid(gate2)
#         x = (filter2+x2[:,:,:,2:]) * torch.sigmoid(gate2)
#         return x

"""
STGCN中的st-block
params:
    c_in:输入维度
    c_out:输出维度
    num_nodes:图节点
    tem_size:训练序列长度
    K:空间卷积阶数
    Kt:时序卷积核大小
"""
class ST_BLOCK_4(nn.Module):

    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_4,self).__init__()
        spatial_channels = 32
        # self.temporal1 = TimeBlock(in_channels=c_in,
        #                            out_channels=spatial_channels)
        # '''使用padding的时序卷积'''
        self.temporal1 = TimeBlock_padding(in_channels=c_in,
                                   out_channels=spatial_channels)
        # '''使用inception结构'''
        # self.temporal1 = TimeBlock_inception(in_channels=c_in,
        #                           out_channels=spatial_channels)
        self.Theta1 = nn.Parameter(
            torch.FloatTensor(spatial_channels, spatial_channels))
        # self.temporal2 = TimeBlock(in_channels=spatial_channels,
        #                            out_channels=c_out)
        # '''使用padding的时序卷积'''
        self.temporal2 = TimeBlock_padding(in_channels=spatial_channels,
                                   out_channels=c_out)
        # '''使用inception结构'''
        # self.temporal2 = TimeBlock_inception(in_channels=spatial_channels,
        #                           out_channels=c_out)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

        ''' drop out '''
        self.dropout = nn.Dropout(p=0.05)
        # self.dropout = nn.Dropout()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, x, supports):

        t = self.temporal1(x)
        lfs = torch.einsum("ij,jklm->kilm", [supports, t.permute(1, 0, 2, 3)])
        # 加一层残差
        t2 = F.relu(torch.matmul(lfs, self.Theta1) + t)
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t2 = t2.permute(0, 3, 1, 2)

        t3 = self.temporal2(t2)
        out = self.batch_norm(t3)
        # 加一层 dropout
        # return self.dropout(out)
        return out

###GRCN(ICLR)
class gcn_conv_hop(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ] - input of one single time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : gcn_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(gcn_conv_hop,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv1d(c_in_new, c_out, kernel_size=1,
                          stride=1, bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcn,knq->bckq', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode)
        out = self.conv1(x)
        return out 



class ST_BLOCK_5(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_5,self).__init__()
        self.gcn_conv=gcn_conv_hop(c_out+c_in,c_out*4,K,1)
        self.c_out=c_out
        self.tem_size=tem_size
        
        
    def forward(self,x,supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        c = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        out=[]
        
        for k in range(self.tem_size):
            input1=x[:,:,:,k]
            tem1=torch.cat((input1,h),1)
            fea1=self.gcn_conv(tem1,supports)
            i,j,f,o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c=c*torch.sigmoid(f)+torch.sigmoid(i)*torch.tanh(j)
            new_h=torch.tanh(new_c)*(torch.sigmoid(o))
            c=new_c
            h=new_h
            out.append(new_h)
        x=torch.stack(out,-1)
        return x

"""
ADGCN中的通道注意力
"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()

        # 输入是 (N, C, H, W)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # in_planes // 16 表示通道压缩为 1/16
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x.shape = [b, F, H, W]

        # 平均池化
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        # out = avg_out
        return self.sigmoid(out)


"""
ADGCN中的st-block, inception结构
params:
    c_in:输入维度
    c_out:输出维度
    num_nodes:图节点
    tem_size:训练序列长度
    K:空间卷积阶数
    Kt:时序卷积核大小
"""
class ST_BLOCK_8(nn.Module):

    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_8, self).__init__()
        spatial_channels = 32

        '''使用inception结构的时序卷积'''
        self.temporal1 = TimeBlock_inception(in_channels=c_in,
                                   out_channels=spatial_channels)
        # self.temporal1 = TimeBlock_padding(in_channels=c_in,
        #                                    out_channels=spatial_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(spatial_channels, spatial_channels))
        self.temporal2 = TimeBlock_inception(in_channels=spatial_channels,
                                   out_channels=c_out)
        # self.temporal2 = TimeBlock_padding(in_channels=spatial_channels,
        #                                    out_channels=c_out)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

        self.channel_attention = ChannelAttention(in_planes=spatial_channels, ratio=16)
        '''经过注意力加权后的权重'''
        self.r_c = nn.Parameter(torch.randn(size=(1, 1)))
        self.r = nn.Parameter(torch.randn(size=(1, 1)))

        ''' drop out '''
        self.dropout = nn.Dropout(p=0.05)
        # self.dropout = nn.Dropout()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, x, supports):
        """"""

        t = self.temporal1(x)
        # lfs shape = [batch_size, N, time_len, channels]
        lfs = torch.einsum("ijk,iklm->ijlm", [supports, t])
        # 加一层残差
        t2 = F.relu(torch.matmul(lfs, self.Theta1) + t)
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))

        '''在空间卷积后使用一个通道注意力'''
        ca = self.channel_attention(t2.permute(0, 3, 1, 2))
        ca = ca.permute(0, 2, 3, 1)
        # t2 = torch.mul(ca, t2)
        t2 = self.r_c * torch.mul(ca, t2) + self.r * t2

        t3 = self.temporal2(t2)
        out = self.batch_norm(t3)
        # 加一层 dropout
        # return self.dropout(out)
        return out

"""
ADGCN中的st-block
params:
    c_in:输入维度
    c_out:输出维度
    num_nodes:图节点
    tem_size:训练序列长度
    K:空间卷积阶数
    Kt:时序卷积核大小
"""
class ST_BLOCK_7(nn.Module):

    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_7, self).__init__()
        spatial_channels = 32

        self.temporal1 = TimeBlock(in_channels=c_in,
                                   out_channels=spatial_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(spatial_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=c_out)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

        self.channel_attention = ChannelAttention(in_planes=spatial_channels, ratio=16)

        '''经过注意力加权后的权重'''
        self.r_c = nn.Parameter(torch.randn(size=(1, 1)))
        self.r = nn.Parameter(torch.randn(size=(1, 1)))

        ''' drop out '''
        self.dropout = nn.Dropout(p=0.05)
        # self.dropout = nn.Dropout()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, x, supports):
        """"""

        t = self.temporal1(x)
        # lfs shape = [batch_size, N, time_len, channels]
        lfs = torch.einsum("ijk,iklm->ijlm", [supports, t])
        # 加一层残差
        t2 = F.relu(torch.matmul(lfs, self.Theta1) + t)
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))

        '''在空间卷积后使用一个通道注意力'''
        ca = self.channel_attention(t2.permute(0, 3, 1, 2))
        ca = ca.permute(0, 2, 3, 1)
        '''这样的效果比直接ca*t2的结果好一些, 相当于残差了'''
        t2 = self.r_c * torch.mul(ca, t2) + self.r * t2

        t3 = self.temporal2(t2)
        out = self.batch_norm(t3)
        # 加一层 dropout
        # return self.dropout(out)
        return out


###OTSGGCN(ITSM)
class cheby_conv1(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv1,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

###TGCN
class TGCNGraphConvolution(nn.Module):
    def __init__(self, device, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self.device = device
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        # self.register_buffer(
        #     "laplacian", self.calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )
        self.laplacian = self.calculate_laplacian_with_self_loop(adj)
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def calculate_laplacian_with_self_loop(self, matrix):
        matrix = matrix + torch.eye(matrix.size(0)).to(self.device)
        # matrix = matrix + torch.eye(matrix.size(0))
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(self.device)
        # d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

###
class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int, device):
        super(TGCNCell, self).__init__()
        self.device = device
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        # self.register_buffer("adj", torch.FloatTensor(adj))
        self.adj = adj
        self.graph_conv1 = TGCNGraphConvolution(
            self.device, self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.device, self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

class ST_BLOCK_6(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_6,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out,2*c_out,K,1)
        
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)   
        x2=self.gcn(x1,supports)
        filter,gate=torch.split(x2,[self.c_out,self.c_out],1)
        x=(filter+x_input1)*torch.sigmoid(gate)
        return x    
    
    
##gwnet
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class multi_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(multi_gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    


##cluster    

    
class nconv_batch(nn.Module):
    def __init__(self):
        super(nconv_batch,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        #try:
       #     x = torch.einsum('ncvl,vw->ncwl',(x,A))
        #except:
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()
    
class linear_time(nn.Module):
    def __init__(self,c_in,c_out,Kt):
        super(linear_time,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class multi_gcn_time(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_time,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SATT_pool(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_pool,self).__init__()
        self.conv1=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = F.softmax(logits,2)
        logits = torch.mean(logits,-1)
        return logits

class SATT_h_gcn(nn.Module):
    def __init__(self,c_in,tem_size):
        super(SATT_h_gcn,self).__init__()
        self.conv1=Conv2d(c_in, c_in//8, kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in//8, kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=False)
        self.c_in=c_in
    def forward(self,seq,a):
        shape = seq.shape
        f1 = self.conv1(seq).squeeze().permute(0,2,1).contiguous()
        f2 = self.conv2(seq).squeeze().contiguous()
        
        logits = torch.matmul(f1,f2)
        
        logits=F.softmax(logits,-1)
        
        return logits

class multi_gcn_batch(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_batch,self).__init__()
        self.nconv = nconv_batch()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:            
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gate(nn.Module):
    def __init__(self,c_in):
        super(gate,self).__init__()
        self.conv1=Conv2d(c_in, c_in//2, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        
        
    def forward(self,seq,seq_cluster):
        
        #x=torch.cat((seq_cluster,seq),1)     
        #gate=torch.sigmoid(self.conv1(x)) 
        out=torch.cat((seq,(seq_cluster)),1)    
        
        return out
           
    
class Transmit(nn.Module):
    def __init__(self,c_in,tem_size,transmit,num_nodes,cluster_nodes):
        super(Transmit,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(tem_size,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(num_nodes,cluster_nodes), requires_grad=True)
        self.c_in=c_in
        self.transmit=transmit
        
    def forward(self,seq,seq_cluster):
        
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq_cluster.permute(0,3,1,2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
        logits=torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        a = torch.mean(logits, 1, True)
        logits = logits - a
        logits = torch.sigmoid(logits)
        
        coefs = (logits)*self.transmit
        return coefs    

class T_cheby_conv_ds_1(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv_ds_1,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 
    
class dynamic_adj(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(dynamic_adj,self).__init__()
        
        self.SATT=SATT_pool(c_in,num_nodes)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
    def forward(self,x):
        S_coef=self.SATT(x)        
        shape=S_coef.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) #2020/3/28/22:17,试验下效果
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        
        return adj_out
    
    
class GCNPool_dynamic(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_dynamic,self).__init__()
        self.dropout=dropout
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        self.multigcn1=multi_gcn_batch(c_out,2*c_out,Kt,dropout,support_len,order)
        self.dynamic_gcn=T_cheby_conv_ds_1(c_out,2*c_out,order+1,Kt)
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.gate=gate1(c_out)
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.SATT=SATT_pool(c_out,num_nodes)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
        
    
    def forward(self,x,support):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        
        
        x=self.multigcn(x,support) 
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        
          
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)       
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #out=torch.sigmoid(x)
        return out



class GCNPool_h(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_h,self).__init__()
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        self.multigcn1=multi_gcn_batch(c_out,2*c_out,Kt,dropout,support_len,order)
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        self.dynamic_gcn=T_cheby_conv_ds_1(c_out,2*c_out,order+1,Kt)
        # 作者的错误???
        # self.gate=gate1(2*c_out)
    
    def forward(self,x,support,A):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        #print(x.shape)
        #dynamic_adj=self.SATT(x)
        new_support=[]
        new_support.append(support[0]+A)
        new_support.append(support[1]+A)
        new_support.append(support[2]+A)
        x=self.multigcn1(x,new_support)        
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2))
        
        
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)  
              
        out=self.bn(x+residual[:, :, :, -x.size(3):])       
        return out
   
           
class GCNPool(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool,self).__init__()
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

    def forward(self,x,support):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        
        
        x=self.multigcn(x,support)        
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        #x=F.dropout(x,0.3,self.training)
        
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        return out

# 快速版的空间注意力, 使用了修改后的raod-wise embedding
class SpatialAttention_fast(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_fast, self).__init__()

        self.time_len = 12
        self.N = 716

        # 输入通道为2, 输出通道为1, 卷积核大小为7, padding大小
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding='SAME', bias=False)#concat完channel维度为2
        self.sigmoid = nn.Sigmoid()

        '''嵌入使用的矩阵为road-wise'''
        in_channels = 10
        self.emb_list = list()
        # 用于嵌入矩阵补全
        self.padding = list()

        # 遍历adj_list, 声明参数
        self.embedding = nn.Parameter(torch.randn(size=(self.N, in_channels, in_channels)))

        self.embedding_bias = nn.Parameter(torch.randn(size=(self.N, 1, in_channels)))

        # FIXME 这里可能出现特别大或者特别小的数
        # self.W1 = nn.Parameter(torch.FloatTensor(time_len))
        # self.W2 = nn.Parameter(torch.FloatTensor(in_channels, time_len))
        # self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        # self.bs = nn.Parameter(torch.FloatTensor(1, N, N))
        # self.Vs = nn.Parameter(torch.FloatTensor(N, N))

        # 用正态分布初始化
        self.W1 = nn.Parameter(torch.randn(self.time_len))
        self.W2 = nn.Parameter(torch.randn(in_channels, self.time_len))
        self.W3 = nn.Parameter(torch.randn(in_channels))
        self.bs = nn.Parameter(torch.randn(1, self.N, self.N))
        self.Vs = nn.Parameter(torch.randn(self.N, self.N))

    def forward(self, input_tpl):

        # 输入的tpl [batch_size, time_len], 通过mlp映射
        batch_size, time_len = input_tpl.shape[0], input_tpl.shape[1]

        # embedding 矩阵补齐
        # embedding_hat = [torch.cat(self.emb_list[i], self.padding[i]) for i in range(len(self.emb_list))]
        # embedding_hat = torch.cat(embedding_hat)

        # 先对x补齐
        # 在预处理就可以完成
        # x_hat = list()
        # for i in range(batch_size):
        #     for j in range(time_len):
        #         for k, d in input_tpl[i, j].items():
        #             tmp = [v for v in d.values()]
        #             tmp += [0 for i in range(10 - len(tmp))]
        #             x_hat.append(torch.tensor(tmp, dtype=float))
        # # reshape
        # x_hat = torch.cat([v for v in x_hat]).float().cuda()
        x_hat = torch.tensor(input_tpl).float().cuda()
        x_hat = x_hat.reshape(shape=(batch_size, time_len, self.N, 10))
        # x_hat.shape = (batch_size, time_len, N, 10)
        # x_hat.shape: (batch_size, time_len, N, 10) -> (batch_size, time_len, N, 1, 10)
        x_hat = x_hat.unsqueeze(3)
        # embedding_hat.shape = (N, 10, 10)
        # x_hat.shape = (batch_size, time_len, N, 1, 10)
        x_hat = torch.matmul(x_hat, self.embedding)

        '''x_hat后加上bias, 再加上一个激活函数, 获得非线性能力'''
        x_hat += self.embedding_bias
        x_hat = F.relu(x_hat)

        # remove demension whose size is 1
        x_hat = x_hat.squeeze()
        x_hat = x_hat.permute(0, 2, 3, 1)

        # lhs = x * w1 * w2, [b, N, F, T] * [T] * [F, T] = [b, N, T]
        lhs = torch.matmul(torch.matmul(x_hat, self.W1), self.W2)

        # rhs = (W3 * x).transpose(-1, -2), ([F] * [b, N, F, T]).transpose(-1, -2) = [b, T, N]
        rhs = torch.matmul(self.W3, x_hat).transpose(-1, -2)

        # product = [b, N, T] * [b, T, N] = [B, N, N]
        product = torch.matmul(lhs, rhs)

        # S = Vs * sigmoid(product + bs), [N, N] * [b, N, N] = [b, N, N]
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        # TODO 这里可以考虑使用 gamble softmax
        S_normalized = F.softmax(S, dim=1)

        return S_normalized

# TPL的空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, adj_list=None):
        super(SpatialAttention, self).__init__()
        # 输入通道为2, 输出通道为1, 卷积核大小为7, padding大小
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding='SAME', bias=False)#concat完channel维度为2
        self.sigmoid = nn.Sigmoid()

        '''为每个节点做一个mlp, 从邻居映射到k(in_channels)'''
        in_channels = 10
        self.emb_dict = dict()
        # 遍历adj_list
        for k, d in adj_list.items():
            # d 可能为空, 空的情况不能用mlp
            # TODO 这里加上了自边, 因此不会出现空
            self.emb_dict[k] = Linear(len(d), in_channels)

        time_len = 12
        N = 716
        # FIXME 这里可能出现特别大或者特别小的数
        # self.W1 = nn.Parameter(torch.FloatTensor(time_len))
        # self.W2 = nn.Parameter(torch.FloatTensor(in_channels, time_len))
        # self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        # self.bs = nn.Parameter(torch.FloatTensor(1, N, N))
        # self.Vs = nn.Parameter(torch.FloatTensor(N, N))

        # 用正态分布初始化
        self.W1 = nn.Parameter(torch.randn(time_len))
        self.W2 = nn.Parameter(torch.randn(in_channels, time_len))
        self.W3 = nn.Parameter(torch.randn(in_channels))
        self.bs = nn.Parameter(torch.randn(1, N, N))
        self.Vs = nn.Parameter(torch.randn(N, N))

    def forward(self, input_tpl):

        # 输入的tpl [batch_size, time_len], 通过mlp映射
        batch_size, time_len = input_tpl.shape[0], input_tpl.shape[1]
        x = list()
        for i in range(batch_size):
            for j in range(time_len):
                for k, d in input_tpl[i, j].items():
                    tmp = self.emb_dict[k](torch.tensor([v for v in d.values()]).float())
                    x.append(tmp)
        # x reshape
        '''性能检测发现这一步非常耗时'''
        x = torch.cat([v for v in x])
        x = x.reshape(shape=(batch_size, time_len, len(self.emb_dict), 10))
        # x.shape = [batch_size, time_len, N, 10]
        # x -> [batch_size, N, 10, time_len]
        x = x.permute(0, 2, 3, 1).cuda()

        # lhs = x * w1 * w2, [b, N, F, T] * [T] * [F, T] = [b, N, T]
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)

        # rhs = (W3 * x).transpose(-1, -2), ([F] * [b, N, F, T]).transpose(-1, -2) = [b, T, N]
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)

        # product = [b, N, T] * [b, T, N] = [B, N, N]
        product = torch.matmul(lhs, rhs)

        # S = Vs * sigmoid(product + bs), [N, N] * [b, N, N] = [b, N, N]
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        # TODO 这里可以考虑使用 gamble softmax
        S_normalized = F.softmax(S, dim=1)

        return S_normalized