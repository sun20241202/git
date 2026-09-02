import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    # 读取数据
    df = pd.read_csv("data/robot_data.csv")

    # 输入特征与预测目标
    X = df[["振动值(mm/s)", "温度(℃)"]]
    y = df["健康状态"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 保持各类别在训练集和测试集中的比例
    )

    # 创建随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 保存模型
    joblib.dump(model, "model.pkl")
    print("模型已保存为 model.pkl")

    # 打印准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率 Accuracy: {accuracy:.4f}")

    # 打印分类报告
    print("\n分类报告：")
    print(classification_report(
        y_test,
        y_pred,
        labels=["平稳", "轻微抖动", "严重抖动"]
    ))


if __name__ == "__main__":
    main()