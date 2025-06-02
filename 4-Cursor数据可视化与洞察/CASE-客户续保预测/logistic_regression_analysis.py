import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib  # 添加这一行
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path):
    """
    加载和预处理数据
    """
    print("正在加载数据...")
    df = pd.read_excel(file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("\n数据基本信息:")
    print(df.info())
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(df.head())
    
    return df

def encode_categorical_features(df):
    """
    对分类特征进行编码
    """
    print("\n正在处理分类特征...")
    
    # 创建数据副本
    df_encoded = df.copy()
    
    # 识别分类列（排除目标变量）
    categorical_columns = []
    for col in df.columns:
        if col != 'renewal' and (df[col].dtype == 'object' or df[col].dtype == 'category'):
            categorical_columns.append(col)
    
    print(f"分类特征: {categorical_columns}")
    
    # 对分类特征进行标签编码
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # 处理目标变量
    if 'renewal' in df.columns:
        if df['renewal'].dtype == 'object':
            le_target = LabelEncoder()
            df_encoded['renewal'] = le_target.fit_transform(df['renewal'])
            print(f"目标变量 renewal: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
            label_encoders['renewal'] = le_target
    
    return df_encoded, label_encoders

def train_logistic_regression(X, y):
    """
    训练逻辑回归模型
    """
    print("\n正在训练逻辑回归模型...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练逻辑回归模型
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    return lr_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_pred_proba

def print_model_coefficients(model, feature_names):
    """
    打印模型系数
    """
    print("\n=== 逻辑回归模型系数 ===")
    print(f"截距项 (Intercept): {model.intercept_[0]:.4f}")
    print("\n特征系数:")
    
    # 创建系数DataFrame
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': model.coef_[0],
        '绝对值': np.abs(model.coef_[0])
    })
    
    # 按绝对值排序
    coef_df = coef_df.sort_values('绝对值', ascending=False)
    
    for idx, row in coef_df.iterrows():
        print(f"{row['特征']:20s}: {row['系数']:8.4f} (|{row['绝对值']:.4f}|)")
    
    return coef_df

def visualize_coefficients(coef_df, save_path=None):
    """
    可视化逻辑回归系数
    """
    print("\n正在生成系数可视化图表...")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 图1: 系数条形图（按绝对值排序）
    colors = ['red' if x < 0 else 'blue' for x in coef_df['系数']]
    bars1 = ax1.barh(range(len(coef_df)), coef_df['系数'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(coef_df)))
    ax1.set_yticklabels(coef_df['特征'])
    ax1.set_xlabel('系数值')
    ax1.set_title('逻辑回归系数 (按重要性排序)')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, coef) in enumerate(zip(bars1, coef_df['系数'])):
        ax1.text(coef + (0.01 if coef >= 0 else -0.01), i, f'{coef:.3f}', 
                va='center', ha='left' if coef >= 0 else 'right', fontsize=9)
    
    # 图2: 系数重要性（绝对值）
    bars2 = ax2.barh(range(len(coef_df)), coef_df['绝对值'], color='green', alpha=0.7)
    ax2.set_yticks(range(len(coef_df)))
    ax2.set_yticklabels(coef_df['特征'])
    ax2.set_xlabel('系数绝对值')
    ax2.set_title('特征重要性 (系数绝对值)')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, abs_coef) in enumerate(zip(bars2, coef_df['绝对值'])):
        ax2.text(abs_coef + 0.01, i, f'{abs_coef:.3f}', va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"系数可视化图已保存至: {save_path}")
    
    plt.show()

def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    评估模型性能
    """
    print("\n=== 模型性能评估 ===")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # AUC分数
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC分数: {auc_score:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(cm)
    
    return auc_score, cm

def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    绘制ROC曲线
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存至: {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    """
    绘制混淆矩阵热力图
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['不续保', '续保'], 
                yticklabels=['不续保', '续保'])
    plt.title('混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.show()

def save_model_and_preprocessors(model, scaler, label_encoders, feature_columns, save_dir):
    """
    保存训练好的模型和预处理器
    """
    import os
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'logistic_regression_model.pkl')
    joblib.dump(model, model_path)
    print(f"模型已保存至: {model_path}")
    
    # 保存标准化器
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存至: {scaler_path}")
    
    # 保存标签编码器
    encoders_path = os.path.join(save_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"标签编码器已保存至: {encoders_path}")
    
    # 保存特征列名
    features_path = os.path.join(save_dir, 'feature_columns.pkl')
    joblib.dump(feature_columns, features_path)
    print(f"特征列名已保存至: {features_path}")
    
    return model_path, scaler_path, encoders_path, features_path

def main():
    """
    主函数
    """
    # 文件路径
    file_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/policy_data.xlsx'
    
    try:
        # 1. 加载和预处理数据
        df = load_and_preprocess_data(file_path)
        
        # 2. 编码分类特征
        df_encoded, label_encoders = encode_categorical_features(df)
        
        # 3. 准备特征和目标变量
        if 'renewal' not in df_encoded.columns:
            print("错误: 未找到目标变量 'renewal'")
            return
        
        # 排除非特征列
        exclude_columns = ['policy_id', 'renewal']
        if 'policy_start_date' in df_encoded.columns:
            exclude_columns.append('policy_start_date')
        if 'policy_end_date' in df_encoded.columns:
            exclude_columns.append('policy_end_date')
        
        feature_columns = [col for col in df_encoded.columns if col not in exclude_columns]
        
        X = df_encoded[feature_columns]
        y = df_encoded['renewal']
        
        print(f"\n特征列: {feature_columns}")
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量分布:\n{y.value_counts()}")
        
        # 4. 训练逻辑回归模型
        lr_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_pred_proba = train_logistic_regression(X, y)
        
        # 5. 打印模型系数
        coef_df = print_model_coefficients(lr_model, feature_columns)
        
        # 6. 可视化系数
        coef_viz_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/lr_coefficients.png'
        visualize_coefficients(coef_df, coef_viz_path)
        
        # 7. 评估模型
        auc_score, cm = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # 8. 绘制ROC曲线
        roc_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/lr_roc_curve.png'
        plot_roc_curve(y_test, y_pred_proba, roc_path)
        
        # 9. 绘制混淆矩阵
        cm_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/lr_confusion_matrix.png'
        plot_confusion_matrix(cm, cm_path)
        
        # 10. 保存系数到CSV文件
        coef_csv_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/lr_coefficients.csv'
        coef_df.to_csv(coef_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n系数数据已保存至: {coef_csv_path}")
        
        # 11. 保存模型和预处理器
        save_dir = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/saved_model'
        model_path, scaler_path, encoders_path, features_path = save_model_and_preprocessors(
            lr_model, scaler, label_encoders, feature_columns, save_dir
        )
        
        print("\n=== 分析完成 ===")
        print(f"模型AUC分数: {auc_score:.4f}")
        print("所有结果文件已保存到项目目录中。")
        print("\n=== 模型文件已保存 ===")
        print(f"模型文件: {model_path}")
        print(f"标准化器: {scaler_path}")
        print(f"标签编码器: {encoders_path}")
        print(f"特征列名: {features_path}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确认文件路径是否正确。")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()