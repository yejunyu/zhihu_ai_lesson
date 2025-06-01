import pandas as pd

# 读取Excel文件
df = pd.read_excel(
    "/Users/xiejunyu/Documents/ai_Lesson/3-Cursor编程-从入门到精通/【完成参考】dashboard_epidemic/香港各区疫情数据_20250322.xlsx"
)

# 显示前20行
print(df.head(20))

# 显示基本信息
print(f"\n数据形状: {df.shape}")
print(f"\n列名: {list(df.columns)}")
