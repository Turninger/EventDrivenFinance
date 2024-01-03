#import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.functional as F
# import pickle
# import re
# import pandas as pd
import os

import pandas as pd
import datetime

start_time = datetime.datetime.now()  # 获取程序开始时间
path = r'H:\事件驱动选股金融数据集\导出为csv\xk_news_data_20231220.csv'
result_path_dir = r'H:\事件驱动选股金融数据集\拆分之后\新闻数据'  #输出文件的路径
df = pd.read_csv(path,encoding='utf-8',dtype = str)
df['new_date_column'] = pd.to_datetime(df['new_date_column'], format='%Y%m%d')
#l = df.new_date_column.unique()
# 创建一个包含唯一日期字符串的列表。它使用unique()函数从日期列中提取唯一的日期值，并将其存储在列表l中。
#data = [df.loc[df['new_date_column']==i] for i in l]
# 创建了一个名为data的列表。对于每个唯一的日期字符串i，它选择原始DataFrame中日期列中与i匹配的行，并将这些行的子集作为一个新的DataFrame存储在data列表中。
# 假设 data 是包含分好日期的 DataFrame 列表
#for i, df_date in enumerate(data):
for date, group in df.groupby('new_date_column'):
    # 创建文件名，以日期命名，例如 '盘中宝_20180902.csv'
    filename = os.path.join(result_path_dir, f'新闻_{date.strftime("%Y%m%d")}.csv')

    # 将 DataFrame 保存为 CSV 文件
    group.to_csv(filename, index=False, sep='\t')

    print(f'{os.path.basename(filename)} 已保存成功')

end_time = datetime.datetime.now()  # 获取程序结束时间
print(start_time)
print(end_time)


