import pandas as pd

# 读取Excel文件
file_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/policy_data.xlsx'

try:
    # 读取Excel文件的前10行数据
    df = pd.read_excel(file_path, nrows=10)
    
    # 显示数据基本信息
    print("数据形状:", df.shape)
    print("\n列名:")
    print(df.columns.tolist())
    
    print("\n前10行数据:")
    print(df)
    
    # 显示数据类型
    print("\n数据类型:")
    print(df.dtypes)
    
    # 显示基本统计信息
    print("\n基本统计信息:")
    print(df.describe())
    
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"读取文件时出错: {e}")