import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
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
    
    # 显示数据基本信息
    print("\n数据基本信息:")
    print(df.info())
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 显示目标变量分布
    print("\n续保字段分布:")
    print(df['renewal'].value_counts())
    
    return df

def encode_categorical_features(df):
    """
    对分类特征进行编码
    """
    print("\n正在对分类特征进行编码...")
    
    # 创建数据副本
    df_encoded = df.copy()
    
    # 存储标签编码器
    label_encoders = {}
    
    # 对分类特征进行编码
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != 'renewal']  # 排除目标变量
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"已编码特征: {col}")
    
    # 对目标变量进行编码
    target_encoder = LabelEncoder()
    df_encoded['renewal'] = target_encoder.fit_transform(df_encoded['renewal'])
    label_encoders['renewal'] = target_encoder
    
    print(f"编码后的续保字段: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
    
    return df_encoded, label_encoders

def train_decision_tree(X_train, y_train, max_depth=5):
    """
    训练决策树模型
    """
    print(f"\n正在训练决策树模型（最大深度={max_depth}）...")
    
    # 创建决策树分类器
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion='gini'
    )
    
    # 训练模型
    dt_classifier.fit(X_train, y_train)
    
    print("决策树模型训练完成！")
    return dt_classifier

def evaluate_model(model, X_test, y_test, label_encoders):
    """
    评估模型性能
    """
    print("\n正在评估模型性能...")
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 分类报告
    target_names = label_encoders['renewal'].classes_
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    return y_pred, y_pred_proba, accuracy

def visualize_decision_tree(model, feature_names, class_names, save_path=None):
    """
    可视化决策树
    """
    print("\n正在生成决策树可视化...")
    
    plt.figure(figsize=(20, 12))
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
    
    plt.title('决策树可视化 (深度=5)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"决策树可视化已保存至: {save_path}")
    
    plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    绘制特征重要性
    """
    print("\n正在生成特征重要性图...")
    
    # 获取特征重要性
    importance = model.feature_importances_
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 绘制特征重要性
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
    plt.title('决策树特征重要性 (Top 10)', fontsize=14, fontweight='bold')
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {save_path}")
    
    plt.show()
    
    return feature_importance_df

def plot_confusion_matrix(y_test, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    """
    print("\n正在生成混淆矩阵图...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('决策树混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存至: {save_path}")
    
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    绘制ROC曲线
    """
    print("\n正在生成ROC曲线...")
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('决策树ROC曲线', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存至: {save_path}")
    
    plt.show()
    
    return roc_auc

def save_decision_tree_text(model, feature_names, save_path=None):
    """
    保存决策树的文本表示
    """
    print("\n正在生成决策树文本表示...")
    
    tree_text = export_text(model, feature_names=feature_names)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(tree_text)
        print(f"决策树文本表示已保存至: {save_path}")
    
    print("\n决策树文本表示（前50行）:")
    print('\n'.join(tree_text.split('\n')[:50]))
    
    return tree_text

def main():
    """
    主函数
    """
    # 文件路径
    data_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/policy_data.xlsx'
    
    try:
        # 1. 加载和预处理数据
        df = load_and_preprocess_data(data_path)
        
        # 2. 编码分类特征
        df_encoded, label_encoders = encode_categorical_features(df)
        
        # 3. 准备特征和目标变量
        X = df_encoded.drop('renewal', axis=1)
        
        # 排除日期时间列
        exclude_columns = []
        if 'policy_start_date' in X.columns:
            exclude_columns.append('policy_start_date')
        if 'policy_end_date' in X.columns:
            exclude_columns.append('policy_end_date')
        
        # 排除日期时间类型的列
        datetime_columns = X.select_dtypes(include=['datetime64']).columns.tolist()
        exclude_columns.extend(datetime_columns)
        
        if exclude_columns:
            print(f"\n排除的列: {exclude_columns}")
            X = X.drop(columns=exclude_columns)
        
        y = df_encoded['renewal']
        
        feature_names = X.columns.tolist()
        class_names = label_encoders['renewal'].classes_
        
        print(f"\n特征数量: {len(feature_names)}")
        print(f"特征名称: {feature_names}")
        print(f"类别名称: {class_names}")
        
        # 4. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        # 5. 训练决策树模型
        dt_model = train_decision_tree(X_train, y_train, max_depth=5)
        
        # 6. 评估模型
        y_pred, y_pred_proba, accuracy = evaluate_model(dt_model, X_test, y_test, label_encoders)
        
        # 7. 可视化结果
        print("\n=== 开始生成可视化图表 ===")
        
        # 决策树可视化
        visualize_decision_tree(
            dt_model, feature_names, class_names,
            save_path='dt_visualization.png'
        )
        
        # 特征重要性
        feature_importance_df = plot_feature_importance(
            dt_model, feature_names,
            save_path='dt_feature_importance.png'
        )
        
        # 混淆矩阵
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            save_path='dt_confusion_matrix.png'
        )
        
        # ROC曲线
        roc_auc = plot_roc_curve(
            y_test, y_pred_proba,
            save_path='dt_roc_curve.png'
        )
        
        # 保存决策树文本表示
        save_decision_tree_text(
            dt_model, feature_names,
            save_path='decision_tree_text.txt'
        )
        
        # 8. 保存特征重要性到CSV
        feature_importance_df.to_csv('dt_feature_importance.csv', index=False)
        print(f"\n特征重要性已保存至: dt_feature_importance.csv")
        
        # 9. 总结
        print("\n=== 决策树分析总结 ===")
        print(f"模型准确率: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"最重要的特征: {feature_importance_df.iloc[0]['feature']}")
        print(f"决策树深度: 5")
        print(f"叶子节点数量: {dt_model.get_n_leaves()}")
        
        print("\n=== 分析完成 ===")
        print("生成的文件:")
        print("- dt_visualization.png: 决策树可视化")
        print("- dt_feature_importance.png: 特征重要性图")
        print("- dt_confusion_matrix.png: 混淆矩阵")
        print("- dt_roc_curve.png: ROC曲线")
        print("- decision_tree_text.txt: 决策树文本表示")
        print("- dt_feature_importance.csv: 特征重要性数据")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
        print("请确认文件路径是否正确。")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()