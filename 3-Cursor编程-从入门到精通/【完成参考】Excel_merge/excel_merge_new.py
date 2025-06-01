import pandas as pd
import os

def merge_employee_data():
    """
    合并员工基本信息表和员工绩效表
    将全年绩效评分作为最后一列添加到员工基本信息中
    """
    try:
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 使用绝对路径
        basic_info_file = os.path.join(script_dir, '员工基本信息表.xlsx')
        performance_file = os.path.join(script_dir, '员工绩效表.xlsx')
        
        if not os.path.exists(basic_info_file):
            print(f"错误：找不到文件 {basic_info_file}")
            return
        
        if not os.path.exists(performance_file):
            print(f"错误：找不到文件 {performance_file}")
            return
        
        # 读取Excel文件
        print("正在读取员工基本信息表...")
        basic_info_df = pd.read_excel(basic_info_file)
        
        print("正在读取员工绩效表...")
        performance_df = pd.read_excel(performance_file)
        
        # 显示数据基本信息
        print(f"\n员工基本信息表包含 {len(basic_info_df)} 条记录")
        print("字段列表：", list(basic_info_df.columns))
        print("\n前3行数据：")
        print(basic_info_df.head(3))
        
        print(f"\n员工绩效表包含 {len(performance_df)} 条记录")
        print("字段列表：", list(performance_df.columns))
        print("\n前3行数据：")
        print(performance_df.head(3))
        
        # 检查是否有员工ID字段用于合并
        if '员工ID' not in basic_info_df.columns:
            print("错误：员工基本信息表中没有找到'员工ID'字段")
            return
        
        if '员工ID' not in performance_df.columns:
            print("错误：员工绩效表中没有找到'员工ID'字段")
            return
        
        # 处理全年绩效数据
        if '年度' in performance_df.columns and '季度' in performance_df.columns:
            # 获取最新年度的所有季度数据
            latest_year = performance_df['年度'].max()
            print(f"\n统计 {latest_year} 年全年绩效数据")
            
            # 筛选最新年度的所有数据
            yearly_performance = performance_df[performance_df['年度'] == latest_year]
            
            # 计算每个员工的年度平均绩效
            if '绩效评分' in yearly_performance.columns:
                performance_summary = yearly_performance.groupby('员工ID').agg({
                    '绩效评分': 'mean'  # 计算平均绩效
                }).reset_index()
                performance_summary['绩效评分'] = performance_summary['绩效评分'].round(2)
                performance_summary = performance_summary.rename(columns={'绩效评分': f'{latest_year}年度平均绩效'})
                print(f"\n计算 {latest_year} 年度平均绩效完成")
            else:
                print("错误：绩效表中没有找到'绩效评分'字段")
                return
        else:
            # 如果没有时间字段，使用所有绩效数据
            print("\n未发现年度和季度字段，使用所有绩效数据")
            performance_summary = performance_df[['员工ID', '绩效评分']].copy()
        
        print(f"\n将要合并的绩效字段：{list(performance_summary.columns)[1:]}")
        
        # 执行左连接合并
        merged_df = pd.merge(
            basic_info_df, 
            performance_summary, 
            on='员工ID', 
            how='left'
        )
        
        # 显示合并结果统计
        print(f"\n=== 数据合并完成 ===")
        print(f"合并后数据包含 {len(merged_df)} 条记录")
        print(f"成功匹配绩效数据的员工：{merged_df[list(performance_summary.columns)[1]].notna().sum()} 人")
        print(f"未匹配到绩效数据的员工：{merged_df[list(performance_summary.columns)[1]].isna().sum()} 人")
        
        # 显示绩效统计信息
        perf_col = list(performance_summary.columns)[1]
        if merged_df[perf_col].notna().sum() > 0:
            print(f"\n=== 绩效统计信息 ===")
            print(f"- 平均绩效评分：{merged_df[perf_col].mean():.2f}")
            print(f"- 最高绩效评分：{merged_df[perf_col].max():.2f}")
            print(f"- 最低绩效评分：{merged_df[perf_col].min():.2f}")
        
        # 保存合并后的数据
        output_file = os.path.join(script_dir, '员工信息与绩效合并表.xlsx')
        merged_df.to_excel(output_file, index=False)
        print(f"\n✅ 合并完成！结果已保存到：{output_file}")
        
        # 显示最终结果的前几行
        print("\n=== 合并结果预览 ===")
        print(merged_df.head())
        
        return merged_df
        
    except Exception as e:
        print(f"\n❌ 数据合并失败：{str(e)}")
        print("请检查文件格式和数据完整性。")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print(" 📋 员工信息与绩效数据合并工具 ")
    print("=" * 50)
    
    result = merge_employee_data()
    
    if result is not None:
        print("\n🎉 程序执行成功！")
    else:
        print("\n💥 程序执行失败，请检查错误信息。")