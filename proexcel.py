import os
import pandas as pd
from datetime import datetime

# Python 标准库中 os 模块

file_dir = './excel_in'
output_file_dir = './sub_excel'

for file_name in os.listdir(file_dir):
    # os.listdir(指定目录) 返回一个列表：包含指定目录中的文件和目录名称
    if file_name.endswith('.csv'):
        # .endswith 字符串方法，用于检查字符串是否以指定的后缀结束
        file_path = os.path.join(file_dir, file_name)
        # os.path.join 将一个或多个部分路径名组合成一个完整的路径名：在 UNIX 和 Linux 中是正斜杠/，在 Windows 中是反斜杠
        data = pd.read_csv(file_path)
        # pd.read_csv 用于从 CSV 文件（逗号分隔值文件）读取数据并将其转换为 DataFrame 对象
        # col_1_name = 'Timeline'
        col_1_name = 'Timeline' if 'Timeline' in data.columns else 'timeline'
        col_2_name = 'x'

        data_col_num = data.shape[0]
        sub_file_col_len = 1200
        sub_file_last_col_ptr = 0

        for idx in range(data_col_num // sub_file_col_len):
            sub_col_1 = data.loc[sub_file_last_col_ptr:sub_file_last_col_ptr + sub_file_col_len - 1, col_1_name]
            sub_col_2 = data.loc[sub_file_last_col_ptr:sub_file_last_col_ptr + sub_file_col_len - 1, col_2_name]

            # # 将时间字符串解析为datetime对象
            # time_objects = [datetime.strptime(time_str, "%Y/%m/%d %H:%M") for time_str in sub_col_1]
            # # 将datetime对象转换为数值（秒数）
            # sub_col_1 = [(time - time_objects[0]).total_seconds() for time in time_objects]
            sub_data = {'Time': sub_col_1, 'FHR': sub_col_2}

            sub_file = pd.DataFrame(sub_data)
            # 转换成 DataFrame 对象
            sub_file_name = os.path.join(output_file_dir, file_name[:-4] + '_' + str(idx) + file_name[-4:])
            # file_name[-4] =.csv前面的部分 ; file_name[-4:] = .csv
            # str(idx)转换为字符串

            sub_file.to_csv(sub_file_name, index=False)
            # 将 DataFrame sub_file 保存为 CSV 文件的
            # 默认情况下，Panda在保存到CSV时将DataFrame的索引（即每行的标签）作为第一列写入；设置index = False表示在输出的CSV文件中不包括索引列
            # 通常用于避免在CSV文件中出现不需要的额外索引列
            sub_file_last_col_ptr += sub_file_col_len
            # += 一个赋值运算符，用于将右侧操作数的值加到左侧操作数的值上，并将结果赋值给左侧操作数
            # 简化的方式来进行加法和赋值的组合操作
