import multiprocessing as mp
import pandas as pd
import os
import numpy as np
import time 
def createDir(id2name, outputPath):
    # create the directories 
    # outputPath: "stock_news/"
    
    #os.mkdir(outputPath)
    #创建名为stock_news的目录
    
    for stockid in id2name.keys():
        os.makedirs(outputPath + "/{}".format(stockid))#循环遍历键，也就是股票的ID，然后以股票id命名目录
    pass

def assign(id_, newsFiles, id2name, stockNewsCount):#把新闻写到txt里面去
    # write corresponding news to corresponding txt file
    stockNewsDateCount = {}
    for csvFile in newsFiles: # 600 days of news file
        # csvFile: sina-2020-10-10.csv
        if csvFile[-3:] != 'csv':
            continue            #只要后缀.csv的文件
        date = csvFile[-14:-4] # 2020-10-10

        # open file
        newsFile = pd.read_csv(csvFile, sep = '\t')
        for tup in newsFile.itertuples(): #循环遍历每一行并返回行元组
            title = tup[2]
            content = tup[3]
            cls_id = tup[4]
            if pd.isna(content):#检查content是否为空，如果为空给成空字符串
                content = ' '
            if pd.isna(title):
                title = ' '
            content = content.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ').replace('\n', ' ')
            #把各类特殊的空白换成正常的空格
            if id_ in title or id_[2:] in title or id2name[id_] in title or \
               id_ in content or id_[2:] in content or id2name[id_] in content or\
               id_ in cls_id or id_[2:] in content or id2name[id_] in cls_id:
                #如果股票ID出现在title或者content中了
                f = open(outputPath + f'/{id_}/{date}.txt', 'a+', encoding='utf-8')
                f.write(title + ' ' + content + '\n')
                f.close()
                #这种情况新建个文件，把内容写里头，位置应该是stockNews/ + f'/{id_}/{date}.txt

                stockNewsDateCount[date] = stockNewsDateCount.get(date, 0) + 1 # maintain the count of news for each stock
                #如果已经有了就加一，没有就先初始化为0之后加一

    stockNewsCount[id_] = stockNewsDateCount #把每个id对应的计数存储到stockNewsCount字典中，id_为键
    pass


def process(idList, stockNewsCount):
    for id_ in idList:
        assign(id_, newsFiles, id2name, stockNewsCount)
    pass

###########################
# Defining variables
# change this path to fit yours directory
dataPath = 'D:\stockdata_and_code\HAN'
outputPath = dataPath + '/stock_news'
mappingPath = dataPath + '/stockid2name.csv'

id2name= pd.read_csv(mappingPath, sep = ',')#csv
id2name = {tup[1]:tup[2] for tup in id2name.itertuples()}#第0位是index，换存成字典类型，键为第一列值为第二列

newsFiles = [dataPath + '/news_from_cls/' + csvFile for csvFile in os.listdir(dataPath + '/news_from_cls')]
#一个csv文件名的列表

# start 
# creating directories
#createDir(id2name, outputPath)
#创建该目录
# initializing variables for parallel processing
if __name__ == '__main__':
    mp.freeze_support()
    id_list = list(id2name.keys())#股票id转换成列表，赋值到id_list
    nb_process = int(mp.cpu_count()) - 1#并行处理初始化的变量，目的是创建合适数量的进程来处理数据
    #nb_process = 7
    l = list(np.array_split(id_list, nb_process))#把股票ID均匀地分配给各个进程进行处理
    l = [x.tolist() for x in l]

    stockNewsCount = mp.Manager().dict() # 创建一个可在多个进程之间共享的字典对象count how many news on a certain date for a certain stock

    process_list = [mp.Process(target=process, args = (idList,stockNewsCount)) for idList in l]
    #每个进程干什么传入什么的列表
    time1=time.time()
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    time2=time.time()
    print('Cost time: ' + str(time2 - time1) + 's')
    # Cost time: 1346.320482969284s
    # for 300 stocks and 4000 news files
    ######################
    # some analysis of the assign result

    stockNewsCount = dict(stockNewsCount)#共享类型转换为普通类型
    cnt = 0
    for key in stockNewsCount.keys():
        cnt += np.sum(list(stockNewsCount[key].values()))#求所有股票新闻总数
    print("Altogether {} news for 300 stocks".format(cnt))
    # Altogether 163251 news for 300 stocks

    print("Averagely {} news for each stock".format(cnt/300))
    # Averagely 544.17 news for each stock

    #count	300.000000
    #mean	544.170000
    #std	1145.446034
    #min	48.000000
    #25%	144.750000
    #50%	260.500000
    #75%	542.750000
    #max	15046.000000
