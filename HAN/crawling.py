# !pip install tushare
# !pip install tqdm

import datetime
import tushare as ts
import numpy as np

'''
def get_nday_list(n):
    before_n_days = []
    for i in range(1, n + 1)[::-1]: #逆序迭代
#        before_n_days.append(str(datetime.date.today() - datetime.timedelta(days=i)))
        before_n_days.append(str(datetime.datetime.strptime('2022-07-04', '%Y-%m-%d') - datetime.timedelta(days=i)))
    return before_n_days

#date_list = get_nday_list(652)
date_list = get_nday_list(733)
print(date_list)
'''
import pandas as pd

from tqdm import tqdm
# # This token is deprecated, we bought it from others for 3 days to crawl the data
# # 此令牌已弃用，我们从其他人那里购买了3天来爬网数据
# #ts.set_token('1cca4b0343f513cd592e6b2e0e2a3cf31a06cb4c0263fb3ff4e6d4ae')
pro = ts.pro_api('20231121155429-5c10eac5-f2de-4911-9733-06d69f36a33f') # 创建一个API用于访问Tushare数据
pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'

#
# for i in tqdm(range(len(date_list) - 1)):
#     df = pro.news(src='sina', start_date=date_list[i], end_date=date_list[i+1], fields = 'datetime,content,title,channels')
#     #接口接入新浪财经，起始日期，得到的csv里包括datetime,content,title,channels
#     df['channels'] = df.loc[:,'channels'].apply(lambda x: ' '.join([d['name'] for d in x]))
#     # channel这一列得到的应该是字典，比如‘id’：‘1’，‘name’：疫情
#     df.to_csv('D:\stockdata_and_code\HAN\StockdatafromTushare\sina-{}.csv'.format(date_list[i]),sep = '\t',index = False)
#     #将处理后的数据框dataframe保存成csv文件，不包含行索引，放到指定的相对路径里
#     df = pro.news(src='wallstreetcn', start_date=date_list[i], end_date=date_list[i+1], fields = 'datetime,content,title,channels')
#     df['channels'] = df.loc[:,'channels'].apply(lambda x: ' '.join(x))
#     df.to_csv('D:\stockdata_and_code\HAN\StockdatafromTushare\ws-{}.csv'.format(date_list[i]),sep = '\t',index = False)
#     #对华尔街见闻网、同花顺金融网、东方财富网、云财经进行相同的爬取操作
#     df = pro.news(src='10jqka', start_date=date_list[i], end_date=date_list[i+1], fields = 'datetime,content,title,channels')
#     # df['channels'] = df.loc[:,'channels'].apply(lambda x: ' '.join([d['name'] for d in x]))
#     df.to_csv('D:/stockdata_and_code/HAN/StockdatafromTushare/ths-{}.csv'.format(date_list[i]),sep = '\t',index = False)
#
#     df = pro.news(src='eastmoney', start_date=date_list[i], end_date=date_list[i+1], fields = 'datetime,content,title,channels')
#     # df['channels'] = df.loc[:,'channels'].apply(lambda x: ' '.join([d['name'] for d in x]))
#     df.to_csv('D:\stockdata_and_code\HAN\StockdatafromTushare\eastmoney-{}.csv'.format(date_list[i]),sep = '\t',index = False)
#
#     df = pro.news(src='yuncaijing', start_date=date_list[i], end_date=date_list[i+1], fields = 'datetime,content,title,channels')
#     df['channels'] = df.loc[:,'channels'].apply(lambda x: ' '.join(x))
#     df.to_csv('D:\stockdata_and_code\HAN\StockdatafromTushare\ycj-{}.csv'.format(date_list[i]),sep = '\t',index = False)

# for i in tqdm(range(len(date_list) - 1)):
#     df = pro.major_news(src='', start_date=date_list[i]+' 00:00:00', end_date=date_list[i+1]+' 00:00:00', fields='title,content,pub_time')
#
#     df.to_csv('D:\stockdata_and_code\HAN\StockdatafromTushare\general-{}.csv'.format(date_list[i]),sep = '\t',index = False)
# #接口接入所有来源的新闻，确定起止时间，选择想要的字段，存在df里

#用买的token就用这个
# pro = ts.pro_api('20231121155429-5c10eac5-f2de-4911-9733-06d69f36a33f') # 创建一个API用于访问Tushare数据
# pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
#用自己的token就用下面这个
# ts.set_token('1c8faad280406de85f6883fbd3cd01cb55f2bfda86bb683558e95f2f')
# pro = ts.pro_api()


# 读取 CSV 文件
file_path = 'D:\stockdata_and_code\HAN\stockid2name.csv'
df = pd.read_csv(file_path)

# 提取股票代码列
stock_codes = df['transformed_id'].tolist()

# 将股票代码列表转换为逗号分隔的字符串
stock_codes_str = ','.join(stock_codes)

# 使用 tushare 查询股票信息
data = pro.stock_basic(ts_code=stock_codes_str, list_status='L', fields='ts_code')
#data = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
#使用Tushare接口获取股票基本信息数据。`exchange`参数为空表示获取所有交易所的股票数据，'L'表示获取上市状态的股票，指定获取股票代码
for ts_code in list(data['ts_code']):
    print("Crawling {}".format(ts_code.lower()[-2:]+ts_code[:-3]))
    #打印正在爬取的股票代码
    df = pro.daily_basic(ts_code=ts_code, start_date = '20200701', end_date='20220704', fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv')
    #获取指定股票代码每天的基本数据
    df.to_csv('D:\stockdata_and_code\HAN\StockdatafromTushare\daily_basic\{}.csv'.format(ts_code.lower()[-2:]+ts_code[:-3]), sep = '\t',index = False)
    #保存成csv文件，文件路径和名称根据股票代码动态生成在daily_basic文件夹中
for ts_code in list(data['ts_code']):
    print("Crawling {}".format(ts_code.lower()[-2:]+ts_code[:-3]))
    # 打印正在爬取的股票代码
    df = pro.daily(ts_code=ts_code, start_date = '20200701', end_date='20220704', fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount')
    df.to_csv('D:\stockdata_and_code\HAN\StockdatafromTushare\daily\{}.csv'.format(ts_code.lower()[-2:]+ts_code[:-3]), sep = '\t',index = False)
    #指定股票代码的另一些数据

