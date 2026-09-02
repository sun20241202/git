import os
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# 创建 Flask 应用
app = Flask(__name__)
# 解决跨域问题
CORS(app)

# 让 JSON 接口返回中文，不再转义为 \uXXXX
app.json.ensure_ascii = False

# ---------- 1. 加载模型 ----------
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在，请先运行 train_model.py")

model = joblib.load(MODEL_PATH)
print("模型加载成功")

# ---------- 2. 初始化 SQLite 数据库 ----------
DB_PATH = "robot_history.db"

def init_db():
    """创建预测记录表（如果不存在）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vibration REAL NOT NULL,
            temperature REAL NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()
print("数据库初始化完成")

# ---------- 3. 健康指数与剩余寿命估算函数 ----------
def estimate_health_and_life(vibration, temperature, status):
    """
    根据振动、温度和状态计算健康指数和剩余寿命估算值。
    返回：(health_score, remaining_life_hours, remaining_life_percent)
    """
    # 初始健康分100
    health_score = 100.0

    # 振动惩罚：振动从0.1到10，线性扣分（最多扣40分）
    health_score -= (vibration / 10.0) * 40.0

    # 温度惩罚：温度从20到90，线性扣分（最多扣40分）
    health_score -= ((temperature - 20.0) / 70.0) * 40.0

    # 状态额外惩罚：根据严重度额外扣分
    if status == "轻微抖动":
        health_score -= 10.0
    elif status == "严重抖动":
        health_score -= 30.0

    # 限制范围0~100
    health_score = max(0.0, min(100.0, health_score))

    # 假设满寿命为1000小时，剩余寿命按健康分数比例计算
    remaining_life_hours = (health_score / 100.0) * 1000.0
    remaining_life_percent = health_score

    return round(health_score, 2), round(remaining_life_hours, 2), round(remaining_life_percent, 2)

# ---------- 4. 根路由 ----------
@app.route('/')
def index():
    return '''
    <h3>工业机器人关节健康状态诊断系统</h3>
    <p>接口列表：</p>
    <ul>
        <li>POST /predict - 预测接口，body: {"vibration": 数值, "temperature": 数值}</li>
        <li>GET /history - 获取最近20条预测记录</li>
    </ul>
    '''

# ---------- 5. 预测接口 ----------
@app.route('/predict', methods=['POST'])
def predict():
    """
    接收 JSON：{"vibration": 数值, "temperature": 数值}
    返回 JSON：包含状态、健康指数、剩余寿命估算值等
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "请求体必须为 JSON 格式"}), 400

    try:
        vibration = float(data.get("vibration"))
        temperature = float(data.get("temperature"))
    except (TypeError, ValueError):
        return jsonify({"error": "振动值和温度必须为数值"}), 400

    # 使用模型预测（特征顺序与训练时一致：振动值、温度）
    input_features = [[vibration, temperature]]
    predicted_status = model.predict(input_features)[0]

    # 计算健康指数和剩余寿命
    health_score, remaining_life_hours, remaining_life_percent = estimate_health_and_life(
        vibration, temperature, predicted_status
    )

    # 将本次预测记录存入数据库
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (timestamp, vibration, temperature, status) VALUES (?, ?, ?, ?)",
        (timestamp, vibration, temperature, predicted_status)
    )
    conn.commit()
    conn.close()

    # 返回完整结果
    return jsonify({
        "status": predicted_status,
        "health_score": health_score,
        "remaining_life_hours": remaining_life_hours,
        "remaining_life_percent": remaining_life_percent
    })

# ---------- 6. 历史记录接口 ----------
@app.route('/history', methods=['GET'])
def history():
    """
    返回最近 20 条预测记录（按时间倒序）
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, vibration, temperature, status FROM predictions ORDER BY id DESC LIMIT 20"
    )
    rows = cursor.fetchall()
    conn.close()

    history = [dict(row) for row in rows]
    return jsonify(history)

# ---------- 7. 启动应用 ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)