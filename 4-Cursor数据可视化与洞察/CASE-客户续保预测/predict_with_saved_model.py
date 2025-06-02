import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_saved_model(save_dir):
    """
    加载保存的模型和预处理器
    """
    print("正在加载保存的模型和预处理器...")
    
    # 加载模型
    model_path = os.path.join(save_dir, 'logistic_regression_model.pkl')
    model = joblib.load(model_path)
    print(f"模型已加载: {model_path}")
    
    # 加载标准化器
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    print(f"标准化器已加载: {scaler_path}")
    
    # 加载标签编码器
    encoders_path = os.path.join(save_dir, 'label_encoders.pkl')
    label_encoders = joblib.load(encoders_path)
    print(f"标签编码器已加载: {encoders_path}")
    
    # 加载特征列名
    features_path = os.path.join(save_dir, 'feature_columns.pkl')
    feature_columns = joblib.load(features_path)
    print(f"特征列名已加载: {features_path}")
    
    return model, scaler, label_encoders, feature_columns

def preprocess_test_data(df, label_encoders, feature_columns):
    """
    预处理测试数据
    """
    print("\n正在预处理测试数据...")
    
    # 创建数据副本
    df_processed = df.copy()
    
    # 对分类特征进行编码
    for col in df_processed.columns:
        if col in label_encoders and col != 'renewal':
            le = label_encoders[col]
            # 处理未见过的类别
            unique_values = df_processed[col].unique()
            for val in unique_values:
                if str(val) not in le.classes_:
                    print(f"警告: 特征 '{col}' 中发现未见过的值 '{val}'，将使用最频繁的类别替代")
                    # 使用训练集中最频繁的类别替代
                    most_frequent = le.classes_[0]  # 假设第一个是最频繁的
                    df_processed.loc[df_processed[col] == val, col] = most_frequent
            
            df_processed[col] = le.transform(df_processed[col].astype(str))
    
    # 确保所有需要的特征列都存在
    missing_features = set(feature_columns) - set(df_processed.columns)
    if missing_features:
        print(f"警告: 测试数据中缺少以下特征: {missing_features}")
        # 为缺失的特征添加默认值（0）
        for feature in missing_features:
            df_processed[feature] = 0
    
    # 选择特征列
    X_test = df_processed[feature_columns]
    
    print(f"预处理后的测试数据形状: {X_test.shape}")
    return X_test

def make_predictions(model, scaler, X_test):
    """
    使用模型进行预测
    """
    print("\n正在进行预测...")
    
    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    predictions = model.predict(X_test_scaled)
    prediction_probabilities = model.predict_proba(X_test_scaled)
    
    return predictions, prediction_probabilities

def save_predictions(df_original, predictions, prediction_probabilities, output_path):
    """
    保存预测结果
    """
    print("\n正在保存预测结果...")
    
    # 创建结果DataFrame
    results_df = df_original.copy()
    results_df['predicted_renewal'] = predictions
    results_df['renewal_probability'] = prediction_probabilities[:, 1]  # 续保概率
    results_df['no_renewal_probability'] = prediction_probabilities[:, 0]  # 不续保概率
    
    # 添加预测标签
    results_df['predicted_renewal_label'] = results_df['predicted_renewal'].map({0: 'No', 1: 'Yes'})
    
    # 保存到Excel文件
    results_df.to_excel(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")
    
    return results_df

def analyze_predictions(results_df):
    """
    分析预测结果
    """
    print("\n=== 预测结果分析 ===")
    
    # 预测分布
    pred_counts = results_df['predicted_renewal_label'].value_counts()
    print(f"\n预测结果分布:")
    for label, count in pred_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    # 概率分布统计
    print(f"\n续保概率统计:")
    print(f"平均续保概率: {results_df['renewal_probability'].mean():.4f}")
    print(f"续保概率中位数: {results_df['renewal_probability'].median():.4f}")
    print(f"续保概率标准差: {results_df['renewal_probability'].std():.4f}")
    
    # 高风险客户（续保概率低于0.3）
    high_risk = results_df[results_df['renewal_probability'] < 0.3]
    print(f"\n高风险客户（续保概率<30%）: {len(high_risk)} 人 ({len(high_risk)/len(results_df)*100:.1f}%)")
    
    # 高价值客户（续保概率高于0.7）
    high_value = results_df[results_df['renewal_probability'] > 0.7]
    print(f"高价值客户（续保概率>70%）: {len(high_value)} 人 ({len(high_value)/len(results_df)*100:.1f}%)")

def main():
    """
    主函数
    """
    # 文件路径
    test_data_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/policy_test.xlsx'
    model_save_dir = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/saved_model'
    output_path = '/Users/xiejunyu/Documents/ai_Lesson/4-Cursor数据可视化与洞察/CASE-客户续保预测/prediction_results.xlsx'
    
    try:
        # 1. 检查模型文件是否存在
        if not os.path.exists(model_save_dir):
            print(f"错误: 模型保存目录不存在: {model_save_dir}")
            print("请先运行训练脚本保存模型。")
            return
        
        # 2. 加载保存的模型
        model, scaler, label_encoders, feature_columns = load_saved_model(model_save_dir)
        
        # 3. 加载测试数据
        print(f"\n正在加载测试数据: {test_data_path}")
        test_df = pd.read_excel(test_data_path)
        print(f"测试数据形状: {test_df.shape}")
        print(f"测试数据列名: {test_df.columns.tolist()}")
        
        # 4. 预处理测试数据
        X_test = preprocess_test_data(test_df, label_encoders, feature_columns)
        
        # 5. 进行预测
        predictions, prediction_probabilities = make_predictions(model, scaler, X_test)
        
        # 6. 保存预测结果
        results_df = save_predictions(test_df, predictions, prediction_probabilities, output_path)
        
        # 7. 分析预测结果
        analyze_predictions(results_df)
        
        print("\n=== 预测完成 ===")
        print(f"预测结果已保存至: {output_path}")
        
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {str(e)}")
        print("请确认文件路径是否正确。")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()