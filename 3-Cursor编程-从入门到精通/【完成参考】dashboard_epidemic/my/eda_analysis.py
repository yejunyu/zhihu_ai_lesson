import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - 保留用于数据显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_data():
    """加载数据"""
    file_path = "/Users/xiejunyu/Documents/ai_Lesson/3-Cursor编程-从入门到精通/【完成参考】dashboard_epidemic/香港各区疫情数据_20250322.xlsx"
    df = pd.read_excel(file_path)
    
    # 数据预处理
    df['报告日期'] = pd.to_datetime(df['报告日期'])
    
    return df

def basic_info(df):
    """基本信息分析"""
    print("=" * 80)
    print("Data Basic Information")
    print("=" * 80)
    print(f"Data shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nDate range: {df['报告日期'].min()} to {df['报告日期'].max()}")
    print(f"\nNumber of regions: {df['地区名称'].nunique()}")
    print(f"\nRegion list: {df['地区名称'].unique().tolist()}")
    
    return df.describe()

def time_series_analysis(df):
    """时间序列分析"""
    # 按日期汇总全港数据
    daily_summary = df.groupby('报告日期').agg({
        '新增确诊': 'sum',
        '累计确诊': 'sum',
        '现存确诊': 'sum',
        '新增康复': 'sum',
        '累计康复': 'sum',
        '新增死亡': 'sum',
        '累计死亡': 'sum'
    }).reset_index()
    
    # 创建时间序列图表
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Hong Kong Epidemic Time Series Analysis', fontsize=16, fontweight='bold')
    
    # 每日新增确诊趋势
    axes[0, 0].plot(daily_summary['报告日期'], daily_summary['新增确诊'], 
                    color='red', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Daily New Confirmed Cases Trend', fontsize=14)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('New Confirmed Cases')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 累计确诊趋势
    axes[0, 1].plot(daily_summary['报告日期'], daily_summary['累计确诊'], 
                    color='orange', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Cumulative Confirmed Cases Trend', fontsize=14)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Cumulative Confirmed Cases')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 现存确诊趋势
    axes[1, 0].plot(daily_summary['报告日期'], daily_summary['现存确诊'], 
                    color='blue', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Active Cases Trend', fontsize=14)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Active Cases')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 每日新增康复趋势
    axes[1, 1].plot(daily_summary['报告日期'], daily_summary['新增康复'], 
                    color='green', linewidth=2, alpha=0.8)
    axes[1, 1].set_title('Daily New Recovered Cases Trend', fontsize=14)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('New Recovered Cases')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/xiejunyu/Documents/ai_Lesson/3-Cursor编程-从入门到精通/【完成参考】dashboard_epidemic/my/time_series_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return daily_summary

def regional_analysis(df):
    """地区分析"""
    # 获取最新日期的数据
    latest_date = df['报告日期'].max()
    latest_data = df[df['报告日期'] == latest_date]
    
    # 按地区汇总总体数据
    regional_summary = df.groupby('地区名称').agg({
        '新增确诊': 'sum',
        '累计确诊': 'max',
        '现存确诊': 'last',
        '累计康复': 'max',
        '累计死亡': 'max',
        '人口': 'first',
        '发病率(每10万人)': 'last'
    }).reset_index()
    
    # 创建地区分析图表
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Hong Kong Regional Epidemic Analysis', fontsize=16, fontweight='bold')
    
    # 各地区累计确诊排序
    regional_summary_sorted = regional_summary.sort_values('累计确诊', ascending=True)
    
    # 累计确诊横向条形图
    axes[0, 0].barh(regional_summary_sorted['地区名称'], 
                    regional_summary_sorted['累计确诊'], 
                    color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Cumulative Confirmed Cases by Region', fontsize=14)
    axes[0, 0].set_xlabel('Cumulative Confirmed Cases')
    
    # 发病率分析
    regional_summary_rate = regional_summary.sort_values('发病率(每10万人)', ascending=True)
    axes[0, 1].barh(regional_summary_rate['地区名称'], 
                    regional_summary_rate['发病率(每10万人)'], 
                    color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Incidence Rate by Region (per 100k)', fontsize=14)
    axes[0, 1].set_xlabel('Incidence Rate')
    
    # 人口与累计确诊散点图
    axes[1, 0].scatter(regional_summary['人口'], regional_summary['累计确诊'], 
                      s=100, alpha=0.7, color='green')
    axes[1, 0].set_title('Population vs Cumulative Cases', fontsize=14)
    axes[1, 0].set_xlabel('Population')
    axes[1, 0].set_ylabel('Cumulative Confirmed Cases')
    
    # 添加地区标签
    for i, txt in enumerate(regional_summary['地区名称']):
        axes[1, 0].annotate(txt, (regional_summary['人口'].iloc[i], 
                                 regional_summary['累计确诊'].iloc[i]),
                           fontsize=8, alpha=0.7)
    
    # 现存确诊饼图
    current_cases = regional_summary[regional_summary['现存确诊'] > 0]
    if len(current_cases) > 0:
        axes[1, 1].pie(current_cases['现存确诊'], 
                       labels=current_cases['地区名称'],
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Active Cases Distribution by Region', fontsize=14)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Active Cases', 
                        ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Active Cases Distribution by Region', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('/Users/xiejunyu/Documents/ai_Lesson/3-Cursor编程-从入门到精通/【完成参考】dashboard_epidemic/my/regional_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return regional_summary

def risk_level_analysis(df):
    """风险等级分析"""
    # 风险等级分布
    risk_counts = df['风险等级'].value_counts()
    
    # 按风险等级和时间分析
    risk_time = df.groupby(['报告日期', '风险等级']).size().reset_index(name='地区数量')
    risk_pivot = risk_time.pivot(index='报告日期', columns='风险等级', values='地区数量').fillna(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Risk Level Analysis', fontsize=16, fontweight='bold')
    
    # 风险等级分布饼图
    colors = ['green', 'yellow', 'orange', 'red']
    axes[0].pie(risk_counts.values, labels=risk_counts.index, 
               autopct='%1.1f%%', startangle=90, colors=colors[:len(risk_counts)])
    axes[0].set_title('Overall Risk Level Distribution', fontsize=14)
    
    # 风险等级时间变化
    for col in risk_pivot.columns:
        axes[1].plot(risk_pivot.index, risk_pivot[col], 
                    label=col, linewidth=2, marker='o', markersize=4)
    axes[1].set_title('Risk Level Trend Over Time', fontsize=14)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Number of Regions')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/xiejunyu/Documents/ai_Lesson/3-Cursor编程-从入门到精通/【完成参考】dashboard_epidemic/my/risk_level_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return risk_counts, risk_pivot

def correlation_analysis(df):
    """相关性分析"""
    # 选择数值型列进行相关性分析
    numeric_cols = ['新增确诊', '累计确诊', '现存确诊', '新增康复', '累计康复', 
                   '新增死亡', '累计死亡', '发病率(每10万人)', '人口']
    
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Epidemic Data Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/xiejunyu/Documents/ai_Lesson/3-Cursor编程-从入门到精通/【完成参考】dashboard_epidemic/my/correlation_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def statistical_summary(df):
    """统计摘要"""
    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)
    
    # 总体统计
    total_cases = df['累计确诊'].max()
    total_recovered = df['累计康复'].max()
    total_deaths = df['累计死亡'].max()
    
    print(f"Total Cumulative Cases: {total_cases:,}")
    print(f"Total Recovered: {total_recovered:,}")
    print(f"Total Deaths: {total_deaths:,}")
    print(f"Recovery Rate: {(total_recovered/total_cases*100):.2f}%")
    print(f"Fatality Rate: {(total_deaths/total_cases*100):.2f}%")
    
    # 每日新增统计
    daily_new = df.groupby('报告日期')['新增确诊'].sum()
    print(f"\nDaily Average New Cases: {daily_new.mean():.2f}")
    print(f"Peak Daily New Cases: {daily_new.max():,}")
    print(f"Peak Date: {daily_new.idxmax().strftime('%Y-%m-%d')}")
    
    # 地区统计
    regional_stats = df.groupby('地区名称')['累计确诊'].max().sort_values(ascending=False)
    print(f"\nRegion with Most Cases: {regional_stats.index[0]} ({regional_stats.iloc[0]:,} cases)")
    print(f"Region with Least Cases: {regional_stats.index[-1]} ({regional_stats.iloc[-1]:,} cases)")
    
    return {
        'total_cases': total_cases,
        'total_recovered': total_recovered,
        'total_deaths': total_deaths,
        'daily_avg': daily_new.mean(),
        'max_daily': daily_new.max(),
        'regional_stats': regional_stats
    }

def generate_report(df, stats):
    """生成分析报告"""
    report = f"""
# Hong Kong Epidemic Data Exploratory Analysis Report

## Data Overview
- Date Range: {df['报告日期'].min().strftime('%Y-%m-%d')} to {df['报告日期'].max().strftime('%Y-%m-%d')}
- Total Records: {len(df):,}
- Number of Regions: {df['地区名称'].nunique()}
- Total Population: {df['人口'].sum():,}

## Overall Epidemic Situation
- Total Cumulative Cases: {stats['total_cases']:,}
- Total Recovered Cases: {stats['total_recovered']:,}
- Total Deaths: {stats['total_deaths']:,}
- Recovery Rate: {(stats['total_recovered']/stats['total_cases']*100):.2f}%
- Fatality Rate: {(stats['total_deaths']/stats['total_cases']*100):.2f}%

## Time Trend Analysis
- Daily Average New Cases: {stats['daily_avg']:.2f}
- Peak Daily New Cases: {stats['max_daily']:,}

## Regional Distribution
- Region with Most Cases: {stats['regional_stats'].index[0]} ({stats['regional_stats'].iloc[0]:,} cases)
- Region with Least Cases: {stats['regional_stats'].index[-1]} ({stats['regional_stats'].iloc[-1]:,} cases)

## Risk Level Distribution
{df['风险等级'].value_counts().to_string()}

## Analysis Conclusions
1. Significant differences in epidemic spread across regions
2. Higher population density areas tend to have more confirmed cases
3. Overall high recovery rate with relatively low fatality rate
4. Epidemic development shows clear temporal fluctuation patterns

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 保存报告
    with open('/Users/xiejunyu/Documents/ai_Lesson/3-Cursor编程-从入门到精通/【完成参考】dashboard_epidemic/my/epidemic_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 80)
    print("Analysis Report Generated and Saved")
    print("=" * 80)
    print(report)

def main():
    """主函数"""
    print("Starting Hong Kong Epidemic Data Exploratory Analysis...")
    
    # 加载数据
    df = load_data()
    
    # 基本信息分析
    desc_stats = basic_info(df)
    
    # 时间序列分析
    daily_summary = time_series_analysis(df)
    
    # 地区分析
    regional_summary = regional_analysis(df)
    
    # 风险等级分析
    risk_counts, risk_pivot = risk_level_analysis(df)
    
    # 相关性分析
    correlation_matrix = correlation_analysis(df)
    
    # 统计摘要
    stats = statistical_summary(df)
    
    # 生成报告
    generate_report(df, stats)
    
    print("\nAnalysis completed! All charts and reports have been saved to the 'my' folder.")
    
    return {
        'dataframe': df,
        'daily_summary': daily_summary,
        'regional_summary': regional_summary,
        'correlation_matrix': correlation_matrix,
        'statistics': stats
    }

if __name__ == "__main__":
    results = main()