import os
import numpy as np
import pandas as pd

# 固定随机种子，保证结果可复现
np.random.seed(42)

# 噪声参数
vibration_noise_std = 0.2    # 振动噪声标准差
temperature_noise_std = 1.5  # 温度噪声标准差
label_noise_ratio = 0.05     # 标签噪声比例（5%）


def get_health_status(vibration, temperature):
    """
    根据理想规则返回健康状态。
    如果不符合任一规则，返回 None，由调用方重新生成。
    """
    if vibration < 2.0 and temperature < 50:
        return "平稳"
    elif 2.0 <= vibration <= 5.0 and 50 <= temperature <= 70:
        return "轻微抖动"
    elif vibration > 5.0 and temperature > 70:
        return "严重抖动"
    return None


def main():
    n_samples = 1000
    rows = []

    while len(rows) < n_samples:
        # 1. 生成原始特征（理想值）
        vibration_true = np.random.uniform(0.1, 10.0)
        temperature_true = np.random.uniform(20, 90)

        # 2. 根据原始值确定标签
        label = get_health_status(vibration_true, temperature_true)

        # 如果原始组合不在任何规则内，则重新生成
        if label is None:
            continue

        # 3. 添加传感器噪声
        vibration = vibration_true + np.random.normal(0, vibration_noise_std)
        temperature = temperature_true + np.random.normal(0, temperature_noise_std)

        # 限制范围，避免超出物理边界
        vibration = np.clip(vibration, 0.1, 10.0)
        temperature = np.clip(temperature, 20, 90)

        # 4. 加入少量标签噪声
        if np.random.random() < label_noise_ratio:
            # 随机替换为任意状态
            label = np.random.choice(["平稳", "轻微抖动", "严重抖动"])

        # 其他字段
        time = len(rows) + 1
        angle = np.random.uniform(-180, 180)
        current = np.random.uniform(1, 20)

        rows.append({
            "时间(s)": time,
            "关节角度(°)": round(angle, 4),
            "电机电流(A)": round(current, 4),
            "振动值(mm/s)": round(vibration, 4),
            "温度(℃)": round(temperature, 4),
            "健康状态": label
        })

    df = pd.DataFrame(rows)

    # 创建 data 目录并保存数据
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/robot_data.csv", index=False, encoding="utf-8-sig")

    print("数据已生成：data/robot_data.csv")
    print("各类样本数量：")
    print(df["健康状态"].value_counts().to_string())
    print(f"噪声参数：振动噪声σ={vibration_noise_std}, 温度噪声σ={temperature_noise_std}, 标签噪声比例={label_noise_ratio*100:.1f}%")


if __name__ == "__main__":
    main()