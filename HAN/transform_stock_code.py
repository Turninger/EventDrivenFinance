import pandas as pd


def transform_stock_code(file_path, id_column_name):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 对id列进行转换
    def transform_id(code):
        # 获取后六个数字和前两个字母
        digits = code[-6:]
        letters = code[:2]

        # 转换格式并拼接新的股票代码
        new_code = f"{digits}.{letters.upper()}"

        return new_code

    # 应用函数进行转换
    df['transformed_id'] = df[id_column_name].apply(transform_id)

    # 将结果写回原文件
    df.to_csv(file_path, index=False)


# 示例用法
file_path = 'D:\stockdata_and_code\HAN\stockid2name.csv'  # 替换为你的文件路径
id_column_name = 'id'  # 替换为你的id列的列名

transform_stock_code(file_path, id_column_name)
