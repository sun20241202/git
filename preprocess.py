"""
数据预处理脚本
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def main():
    # 1. 加载数据
    print("正在加载数据...")
    df = pd.read_csv('data/robot_data.csv')
    print(f"原始数据形状: {df.shape}")
    
    # 2. 检查缺失值
    print(f"缺失值数量: {df.isnull().sum().sum()}")
    
    # 3. 标签编码
    le = LabelEncoder()
    df['状态编码'] = le.fit_transform(df['健康状态'])
    print("标签编码映射:", dict(zip(le.classes_, le.transform(le.classes_))))
    
    # 4. 特征标准化
    scaler = StandardScaler()
    feature_cols = ['振动值(mm/s)', '温度(℃)']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 5. 保存预处理后的数据
    df.to_csv('data/robot_data_processed.csv', index=False)
    print("预处理完成，已保存至: data/robot_data_processed.csv")
    
    # 6. 显示各类别数量
    print("\n各类别样本数量:")
    print(df['健康状态'].value_counts())

if __name__ == "__main__":
    main()