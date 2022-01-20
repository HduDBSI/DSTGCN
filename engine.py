import torch.optim as optim
from model import *
import util
class trainer1():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, decay):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse
    
    
class trainer2():
    def __init__(self, in_dim, seq_length, target_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, decay, x_stats):
        self.model = ASTGCN_Recent(device, num_nodes, dropout, supports=supports, in_dim=in_dim, seq_length=seq_length, out_dim=target_len,
                                   residual_channels=nhid, dilation_channels=nhid,
                                   skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        # 改为pems数据集
        # self.loss = util.masked_mae
        self.loss = util.pems_mae
        # 加上x_stats
        self.x_stats = x_stats
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        #output = [batch_size,12,num_nodes,1]
        output = output.squeeze(-1)
        output = output.transpose(1, 2)
        #
        predict = output[:, :, :]

        # loss = self.loss(predict, real,0.0)
        loss = self.loss(predict, real_val, self.x_stats)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mae = util.masked_mae(predict, real, 0.0).item()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        mape = util.pems_mape(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.squeeze(-1)
        output = output.transpose(1, 2)

        predict = output
        # loss = self.loss(predict, real,0.0)
        loss = self.loss(predict, real_val, self.x_stats)

        # mae = util.masked_mae(predict, real, 0.0).item()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        mape = util.pems_mape(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return loss.item(), mae, mape, rmse

    '''
    case study评价过程, loss 与 metrics 都修改过
    '''
    def ex_eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0)), 相当于 net(input), 正向计算结果
        output, _, _ = self.model(input)
        output = output.squeeze(-1)
        output = output.transpose(1, 2)
        predict = output

        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return mae, rmse, predict
    
class trainer3():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = GRCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse     

"""
Gated_STGCN, x_stats 为训练数据的统计信息
"""
class trainer4():
    def __init__(self, in_dim, seq_length, target_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, decay, x_stats):
        self.model = Gated_STGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, seq_length=seq_length, out_dim=target_len,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        # 将loss从 masked_mae 改为 pems_mae
        # self.loss = util.masked_mae
        self.loss = util.pems_mae
        self.x_stats = x_stats
        
        self.clip = 5

    '''
    训练过程, loss 与 metrics 都修改过
    '''
    def train(self, input, real_val):
        # print("train4")
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        '''STGCN的输出只有一个时间步'''
        output, _, _ = self.model(input)
        predict = output

        loss = self.loss(predict, real_val, self.x_stats)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        mape = util.pems_mape(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return loss.item(), mae, mape, rmse

    '''
    评价过程, loss 与 metrics 都修改过
    '''
    def eval(self, input, real_val):
        # 相当于 net.eval
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0)), 相当于 net(input), 正向计算结果
        output, _, _ = self.model(input)
        predict = output

        loss = self.loss(predict, real_val, self.x_stats)
        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        mape = util.pems_mape(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return loss.item(), mae, mape, rmse

    '''
    case study评价过程, loss 与 metrics 都修改过
    '''
    def ex_eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0)), 相当于 net(input), 正向计算结果
        output, _, _ = self.model(input)
        predict = output

        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return mae, rmse, predict

class trainer5():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = H_GCN_wh(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse       

class trainer6():
    def __init__(self, in_dim,in_dim_cluster, seq_length, num_nodes, cluster_nodes, nhid , dropout, lrate, wdecay, device, supports,supports_cluster,transmit,decay):
        self.model = H_GCN_wdf(device, num_nodes,cluster_nodes, dropout, supports=supports, supports_cluster=supports_cluster,
                           in_dim=in_dim,in_dim_cluster=in_dim_cluster, out_dim=seq_length, transmit=transmit,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.supports=supports
        self.num_nodes=num_nodes

    def train(self, input, input_cluster, real_val,real_val_cluster):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,output_cluster,tran2 = self.model(input,input_cluster)
        output = output.transpose(1,3)
        #output_cluster = output_cluster.transpose(1,3)
        #output = [batch_size,1,num_nodes,12]
        real = torch.unsqueeze(real_val,dim=1)
        #real_cluster = real_val_cluster[:,1,:,:]
        #real_cluster = torch.unsqueeze(real_cluster,dim=1)
        predict = output
        
        loss = self.loss(predict, real,0.0)#+energy
        #loss1 =self.loss(output_cluster, real_cluster,0.0)
        #print(loss)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, input_cluster, real_val,real_val_cluster):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input,input_cluster)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse 
    
    
class trainer7():
    def __init__(self, in_dim,in_dim_cluster, seq_length, num_nodes, cluster_nodes, nhid , dropout, lrate, wdecay, device, supports,supports_cluster,transmit,decay):
        self.model = H_GCN(device, num_nodes,cluster_nodes, dropout, supports=supports, supports_cluster=supports_cluster,
                           in_dim=in_dim,in_dim_cluster=in_dim_cluster, out_dim=seq_length, transmit=transmit,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.supports=supports
        self.num_nodes=num_nodes

    def train(self, input, input_cluster, real_val,real_val_cluster):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,output_cluster,tran2 = self.model(input,input_cluster)
        output = output.transpose(1,3)
        #output_cluster = output_cluster.transpose(1,3)
        #output = [batch_size,1,num_nodes,12]
        real = torch.unsqueeze(real_val,dim=1)
        #real_cluster = real_val_cluster[:,1,:,:]
        #real_cluster = torch.unsqueeze(real_cluster,dim=1)
        predict = output
        
        loss = self.loss(predict, real,0.0)#+energy
        #loss1 =self.loss(output_cluster, real_cluster,0.0)
        #print(loss)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, input_cluster, real_val,real_val_cluster):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input,input_cluster)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse     
    
class trainer8():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = OGCRNN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse    
    
class trainer9():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = OTSGGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse   
    
    
class trainer10():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = LSTM(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,1,num_nodes,12]
       
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse 
    
    
class trainer11():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = GRU(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        print("train11")
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

### TGCN
class trainer12():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay, x_stats):
        self.model = TGCN(device, num_nodes, dropout, supports=supports,
                         in_dim=in_dim, out_dim=seq_length,
                         residual_channels=nhid, dilation_channels=nhid,
                         skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)

        # 将loss从 masked_mae 改为 pems_mae
        # self.loss = util.masked_mae
        self.loss = util.pems_mae
        self.x_stats = x_stats

        self.clip = 5

    def train(self, input, real_val):
        # print("train12")
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        predict = output.permute(0, 2, 1)
        # output = [batch_size,12,num_nodes,1]
        # 只取历史序列的后一位作为计算loss
        real = torch.unsqueeze(real_val, dim=1)
        real = real[:, :, :, 0]

        loss = self.loss(predict, real, self.x_stats)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # 需要映射回原空间
        mae = util.pems_mae(predict,real,self.x_stats).item()
        mape = util.pems_mape(predict,real,self.x_stats).item()
        rmse = util.pems_rmse(predict,real,self.x_stats).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        predict = output.permute(0, 2, 1)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        real = real[:, :, :, 0]

        loss = self.loss(predict, real, self.x_stats)
        mae = util.pems_mae(predict,real,self.x_stats).item()
        mape = util.pems_mape(predict,real,self.x_stats).item()
        rmse = util.pems_rmse(predict,real,self.x_stats).item()
        return loss.item(), mae, mape, rmse

### ADGCN
class trainer13():
    def __init__(self, in_dim, seq_length, target_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay, x_stats, adj_list):
        self.model = ADGCN(device, num_nodes, dropout, supports=supports,
                         in_dim=in_dim, out_dim=target_length, seq_length=seq_length,
                         residual_channels=nhid, dilation_channels=nhid,
                         skip_channels=nhid * 8, end_channels=nhid * 16, adj_list=adj_list)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        # self.loss = util.masked_mae
        # 将loss从 masked_mae 改为 pems_mae
        # self.loss = util.masked_mae
        self.loss = util.pems_mae
        self.x_stats = x_stats

        self.clip = 5

    '''tpl是流量转移'''
    def train(self, input, real_val, tpl):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input, tpl)
        predict = output

        # loss = self.loss(predict, real,0.0)
        loss = self.loss(predict, real_val, self.x_stats)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        mape = util.pems_mape(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val, tpl):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input, tpl)
        # output = [batch_size,12,num_nodes,1]

        predict = output
        # loss = self.loss(predict, real,0.0)
        loss = self.loss(predict, real_val, self.x_stats)

        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        mape = util.pems_mape(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return loss.item(), mae, mape, rmse

    '''
    case study评价过程, loss 与 metrics 都修改过
    '''
    def ex_eval(self, input, real_val, tpl):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0)), 相当于 net(input), 正向计算结果
        output, _, _ = self.model(input, tpl)
        predict = output

        mae = util.pems_mae(predict, real_val, self.x_stats).item()
        rmse = util.pems_rmse(predict, real_val, self.x_stats).item()
        return mae, rmse, predict