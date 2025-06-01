import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置SimHei字体（或你系统中存在的其他中文字体）
font_path = fm.findfont("PingFang", fallback_to_default=False)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False
# 读取CSV文件
df = pd.read_csv("train.csv")

# 打印前5行数据
print("--- 前5行数据 ---")
print(df.head(5))
print("\n" + "-" * 30 + "\n")

# 假设年龄列为 'age'，欺诈列为 'fraud_reported'
age_column = "age"
fraud_column = "fraud_reported"

# 确保欺诈列存在，并将 'Y'/'N' 转换为 1/0 用于计算
if fraud_column in df.columns and age_column in df.columns:
    if df[fraud_column].dtype == "object":
        df["is_fraud"] = df[fraud_column].apply(
            lambda x: 1 if x == "Y" else (0 if x == "N" else pd.NA)
        )
        df.dropna(subset=["is_fraud"], inplace=True)
        df["is_fraud"] = df["is_fraud"].astype(int)
    elif pd.api.types.is_numeric_dtype(df[fraud_column]):
        df["is_fraud"] = df[fraud_column]
    else:
        print(f"欺诈列 '{fraud_column}' 的数据类型无法直接处理，请检查数据。")
        exit()

    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]

    df["age_group"] = pd.cut(df[age_column], bins=bins, labels=labels, right=False)

    fraud_by_age_group = df.groupby("age_group")["is_fraud"].mean().reset_index()
    fraud_by_age_group.rename(columns={"is_fraud": "fraud_rate"}, inplace=True)

    print("--- 不同年龄层的欺诈比例 ---")
    print(fraud_by_age_group)

    # --- 绘制条形图 ---
    plt.figure(figsize=(10, 6))  # 设置图表大小
    bars = plt.bar(
        fraud_by_age_group["age_group"],
        fraud_by_age_group["fraud_rate"],
        color="skyblue",
    )

    # 在每个条形上方显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.2%}",
            va="bottom",
            ha="center",
            fontproperties=prop,
        )

    plt.xlabel("年龄层", fontproperties=prop)
    plt.ylabel("欺诈比例", fontproperties=prop)
    plt.title("不同年龄层的欺诈比例", fontproperties=prop)
    plt.xticks(rotation=45, ha="right", fontproperties=prop)
    plt.tight_layout()
    plt.show()

else:
    missing_cols = []
    if age_column not in df.columns:
        missing_cols.append(age_column)
    if fraud_column not in df.columns:
        missing_cols.append(fraud_column)
    print(f"错误：在CSV文件中找不到以下必需的列：{', '.join(missing_cols)}。")
    print(f"可用的列有：{df.columns.tolist()}")
