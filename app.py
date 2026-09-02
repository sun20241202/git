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

# ---------- 3. 根路由 ----------
@app.route('/')
def index():
    return '''
    <h3>工业机器人关节健康状态诊断系统</h3>
    <p>接口列表：</p>
    <ul>
        <li>POST /predict - 预测接口，body: {"vibration": 数值, "temperature": 数值}</li>
        <li>GET /history - 获取最近20条预测记录</li>
        <li>POST /history/delete - 清空所有历史记录</li>
    </ul>
    '''

# ---------- 4. 预测接口 ----------
@app.route('/predict', methods=['POST'])
def predict():
    """
    接收 JSON：{"vibration": 数值, "temperature": 数值}
    返回 JSON：{"status": "平稳/轻微抖动/严重抖动"}
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

    # 保存记录
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (timestamp, vibration, temperature, status) VALUES (?, ?, ?, ?)",
        (timestamp, vibration, temperature, predicted_status)
    )
    conn.commit()
    conn.close()

    return jsonify({"status": predicted_status})

# ---------- 5. 历史记录接口 ----------
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

# ---------- 6. 删除历史记录接口（清空全部） ----------
@app.route('/history/delete', methods=['POST', 'DELETE'])
def delete_history():
    """
    清空所有历史记录
    返回 JSON：{"message": "历史记录已清空", "deleted_count": 删除条数}
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 查询当前记录数
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    # 删除所有记录
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return jsonify({"message": "历史记录已清空", "deleted_count": count})

# ---------- 7. 启动应用 ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)