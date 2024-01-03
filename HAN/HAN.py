import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
import pickle
import re
import pandas as pd
## wandb
import os
os.environ["WANDB_API_KEY"] = '50be8987754b9e64edf8896536fd4b59ef67f2f5' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"

class InputDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, npyFile, dataloaderFile, embedded_size, max_news_cnt):
        # read in the dataloader file and store it in class
        # make sure some rows that have no news at all are discarded
        # 读取数据加载器文件并将其存储在类中
        # 确保丢弃一些完全没有消息的行
        self.npy = np.load(npyFile)
        self.df  = pd.read_csv(dataloaderFile)
        self.embedded_size = embedded_size#嵌入维度大小
        self.max_news_cnt  = max_news_cnt #最大新闻数量
        self.dataPath = dataPath # prefix of all the files
     
    def __getitem__(self, index):
        # read in one line of data
        # return x, y
        # [['今天', '股市', '又', '降', '了'], sentence2, ..., sentence10], label
        # [vector of corpus1, vector of corpus2, ..., vector of corpus10], label
          
        corpus = self.npy[index, :, :, :]#index值的所有数据维度进行切片
        #语料库，多维
        label = self.df['label'][index]
        if label == 'UP':
            label = 0
        elif label == 'DOWN':
            label = 1
        elif label == 'PRESERVE':
            label = 2
           #标签映射成01 002
        return corpus, label
    def __len__(self):
        return self.df.shape[0]

############################
# Model Building
############################
class HAN(nn.Module):
    def __init__(self, embedded_size, max_news, batch_size, seq_len, hidden_size, num_layers, num_classes, dropout):
        super(HAN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedded_size = embedded_size
        self.batch_size = batch_size #输入几个样本
        self.seq_len = seq_len #序列长度
        self.hidden_size = hidden_size
        self.max_news = max_news#最大新闻数
        self.num_layers = num_layers #GRU层数量
        self.dropout=dropout#减小费时和过拟合

        ################
        # News-level attention
        ################全连接层输入大小为embedding_size，输出大小为1
        self.Wn = nn.Linear(in_features = self.embedded_size, out_features = 1, bias = True) # 1*300
        torch.nn.init.xavier_uniform_(self.Wn.weight)  #权重初始化
        ################
        
        ################
        # GRU门控循环单元
        ################
        self.gru = nn.GRU(input_size=embedded_size,  # The number of expected features in the input x, which is embedded size 
                          hidden_size=hidden_size, # The number of features in the hidden state h, which is the output dim
                          num_layers=num_layers,  # Number of recurrent layers
                          batch_first=True, # (batch, seq, feature)
                          bidirectional=True, # bidirectional GRU, concatenate two directions
                          dropout=self.dropout) # dropout
        ################
        #inputsize预期有多少特征，hiddensize隐层特征，numlayers是GRU层数，输入维度为(batch, seq, feature)，双向，丢弃率
        ################
        # Temporal attention
        ################
        self.Wh = nn.Linear(in_features = 2 * self.hidden_size, out_features = 1, bias = True)
        #时间注意力层  为啥二倍，得到的应该是日期层参数β
        torch.nn.init.xavier_uniform_(self.Wh.weight)
        ################
        # β应该在这层要加权乘
        ################
        # Discriminative Network
        ################
        self.fc = nn.Linear(in_features = 2 * self.hidden_size, out_features = num_classes, bias = True)
        ################

    def forward(self, X):
        # X.shape: [5, 10, 4, 300]
        # batch_size, seq_len, max_news, embedded_size
        
        ################
        # News-level attention
        ################
        Ut = nn.LeakyReLU()(self.Wn(X.float())) # Wn(X)/Ut is [5, 10, 4, 1]
        Ut = torch.squeeze(Ut).unsqueeze(2) # [5, 10, 4, 1] -> [5, 10, 4] -> [5, 10, 1, 4]#去除张量中维度为1的轴，再在第三位添加维度为1的轴
        at = nn.Softmax(dim = 3)(Ut) # [5, 10, 1, 4]#沿着第三个维度归一化
        # [5, 10, 1, 4] * [5, 10, 4, 300] = [5, 10, 1, 300] 
        dt = torch.matmul(at, X) 
        dt = torch.squeeze(dt) # [5, 10, 1, 300] -> [5, 10, 300]
        ################
        
        ################
        # GRU
        ################
        # input:
        #   x of shape (batch, seq_len, input_size)
        #   h0 of shape (num_layers * num_directions, batch, hidden_size)
        # output:
        #   all ht of shape (batch, seq_len, num_directions * hidden_size)
        #   h_n (the last ht) of shape (batch, num_layers * num_directions,  hidden_size #是不是写错了，第二维应该是seq_len
        h0 = torch.zeros(self.num_layers*2, X.size(0), self.hidden_size).to(device)
        #这个RNN的初始隐藏状态（rnn层数，输入数据多少个，隐层特征数）
        nn.init.orthogonal_(h0) #确保初始权重矩阵正交
        h, _ = self.gru(dt,h0)
        #h是输出的隐藏状态序列，用于预测，_产生的额外的记忆相关输出
        ################
        
        ################
        # Temporal attention
        ################
        o_i = nn.LeakyReLU()(self.Wh(h)) # [5, 10, 1]
        beta_i = nn.Softmax(dim = 1)(o_i) # [5, 10, 1]
        V = torch.matmul(beta_i.unsqueeze(3), h.unsqueeze(2)) # [5, 10, 1, 2 * self.hidden_size]
        #应该是5,10,1,1 * 5，10,1,2 * self.hidden_size
        V = torch.sum(torch.squeeze(V), 1) # [5, 2 * self.hidden_size]
        ################
        
        ################
        # Discriminative Network
        ################
        output = self.fc(V)
        output = nn.Softmax(dim=1)(output)
        ################
        
        return output


# Notes:  
# 1. matmul automaticlly do batch matrix multiplication: (*, n, m) * (*, m, p) = (*, n, p)  
# 2. Linear layers will keep the dimension in the front and only transform the last dimension: (*, n) -> (*, m)  
# 3. sigmoid is element-wise  
# 4. softmax need to decide which dimension to sum over 
# 5. when specifying shape in forward, try to set -1 for batch_size. Then we don't have to worry about drop_last in dataloader. 
# 6. remember to apply .to(device) on both model and data 
############################

############################
# Model Training 
############################
# Initiation
dataPath = "D:\stockdata_and_code\HAN" # change this
dataloaderPath_train = dataPath + '/dataloader/train_data_full_1.csv'
dataloaderPath_valid = dataPath + '/dataloader/cv_data_full_1.csv'
dataloaderPath_test = dataPath + '/dataloader/test_data_full_1.csv'

npyFile_train = dataPath + '/dataloader/train_bert_np_full_1.npy'
npyFile_valid = dataPath + '/dataloader/cv_bert_np_full_1.npy'
npyFile_test = dataPath + '/dataloader/test_bert_np_full_1.npy'
num_epochs = 30 # training epoch
learning_rate = 0.5 #控制模型参数更新的步长
batch_size = 512 #每次训练模型时用来更新参数的样本数目
embedded_size = 300 #嵌入向量的维度大小
max_news = 4 # 每个股票的数据中包含的最大新闻数目，超过这个数目则进行填充或截断
hidden_size = 384 # 在 GRU中隐藏状态的维度大小
seq_len = 10 # 模型每次考虑的时间序列长度
num_layers = 2 # GRU 模型中的层数
num_classes = 3 # 预测的类别数目
dropout = 0.5
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loader
train_data = InputDataset(dataPath,
                          npyFile_train,
                          dataloaderPath_train,
                          embedded_size = embedded_size,
                          max_news_cnt = max_news)

valid_data = InputDataset(dataPath,
                          npyFile_valid,
                          dataloaderPath_valid,
                          embedded_size = embedded_size,
                          max_news_cnt = max_news)

test_data  = InputDataset(dataPath,
                          npyFile_test,
                          dataloaderPath_test,
                          embedded_size = embedded_size,
                          max_news_cnt = max_news)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size, 
                                           shuffle=True)#打乱
valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size, 
                                           shuffle=False)
# model
model = HAN(embedded_size = embedded_size,
            max_news = max_news,
            hidden_size = hidden_size,
            batch_size = batch_size,
            seq_len = seq_len,
            num_layers = num_layers,
            num_classes = num_classes,
            dropout = dropout)
model = model.to(device)
# Loss and optimizer 损失函数和优化器
criterion = nn.CrossEntropyLoss() #交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 1e-4) #随机梯度下降SGD
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #另一个优化器备选项
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=3, verbose = True)

import wandb
config = dict(
    epochs=30,
    classes=3,
    batch_size=512,
    learning_rate=0.5)
wandb.init(config=config)
wandb.watch(model,log = 'all')

# Train the model
import time
total_time = 0
elapsed_time = 0
total_step = len(train_loader)
train_loss = []
valid_loss = []
train_acc  = []
valid_acc  = []

for epoch in range(num_epochs):
    time1 = time.time()
    model.train() # train mode
    average_loss = 0
    print("Start training...")
    for i, (X, labels) in enumerate(train_loader):
        X = X.float().to(device) #转换格式放到设备上加速运算
        labels = labels.to(device)
        
        # Forward pass
        y = model(X) # 0.005s
        loss = criterion(y, labels) #实际输出和label之间的损失
        # Backward and optimize
        optimizer.zero_grad() #清理梯度
        loss.backward() #反向传播
        optimizer.step() #更新参数
        elapsed_time += time.time()-time1 #累加训练时间
        average_loss += loss.item() #计算该周期平均损失
        # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Elapsed Time: {}'
        #       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), time.strftime(str(elapsed_time))))
        # 每十轮显示当前是第几轮，用的第几个mini-batch，损失值，已经经过的时间
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Elapsed Time: {}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), time.strftime(str(elapsed_time))))
            #每十轮显示当前是第几轮，用的第几个mini-batch，损失值，已经经过的时间
    average_loss/=len(train_loader)#更新整个epoch的平均loss并添加到列表
    train_loss.append(average_loss)
    
    
    
    time2 = time.time()
    epoch_time = time2-time1 #本轮训练花的时间
    total_time+=epoch_time   #累加到总的训练时间
    print(f'Epoch {epoch} completed, cost time { time.strftime(str(epoch_time))}.')
    torch.save(model.state_dict(), dataPath + '/models/HAN_1210.torch') 
    print("Runing validation")
    
    model.eval() # evaluation mode 
    with torch.no_grad(): #反向传播时候不自动求导
        correct = 0
        total = 0
        for i, (X, labels) in enumerate(train_loader):
            X = X.float().to(device)
            labels = labels.to(device)
            y = model(X)
            _, predicted = torch.max(y.data, 1) #找到预测输出中的最大值和索引，类别
            total += labels.size(0) #累加当前batch的样本数量
            correct += (predicted == labels).sum().item() #计算模型预测正确的样本数，布尔张量.sum.item
        print('Train Accuracy: {} %, Loss: {}'.format(100 * round(correct / total, 4), round(average_loss,3)))
    train_acc.append(correct / total)   
    
    average_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (X, labels) in enumerate(valid_loader):
            X = X.float().to(device)
            labels = labels.to(device)
            y = model(X)
            _, predicted = torch.max(y.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            average_loss +=  criterion(y, labels).item()
        average_loss/=len(valid_loader)
        print('Valid Accuracy: {} %, Loss: {}'.format(100 * round(correct / total, 4), round(average_loss,3))) 
        valid_loss.append(average_loss)
        valid_acc.append(correct / total) 

    
    scheduler.step(average_loss)


    wandb.log({'train_loss':train_loss[-1],
               'valid_loss':valid_loss[-1],
               'train_acc':train_acc[-1],
               'valid_acc':valid_acc[-1]})
############################
