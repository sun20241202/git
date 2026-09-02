"""
自动化测试脚本：test_app.py
用于测试 Flask 后端接口：
- GET /history 返回 200
- POST /predict 正常输入返回 200 且包含 status 字段
- POST /predict 缺少参数返回 400
"""
import unittest
from app import app  # 导入 Flask 应用实例


class TestApp(unittest.TestCase):
    """Flask 接口测试类"""

    def setUp(self):
        """每个测试用例执行前创建测试客户端"""
        self.client = app.test_client()
        # 开启测试模式，便于定位错误
        app.config.update(TESTING=True)

    def test_history_endpoint_returns_200(self):
        """测试 GET /history 是否返回 200"""
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)

    def test_predict_success(self):
        """测试 POST /predict 正常输入"""
        payload = {
            "vibration": 3.5,
            "temperature": 55
        }
        # 使用 json= 参数自动设置 Content-Type: application/json
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)

        # 检查返回 JSON 中包含 status 字段
        data = response.get_json()
        self.assertIn('status', data)
        # 可选：检查状态值是否在预期集合中
        self.assertIn(data['status'], ['平稳', '轻微抖动', '严重抖动'])

    def test_predict_missing_parameters(self):
        """测试 POST /predict 缺少参数时返回 400"""
        # 只提供 vibration，缺少 temperature
        payload = {"vibration": 3.5}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

        # 也可以测试只提供 temperature，缺少 vibration
        payload2 = {"temperature": 55}
        response2 = self.client.post('/predict', json=payload2)
        self.assertEqual(response2.status_code, 400)

        # 测试完全空的 JSON 对象
        response3 = self.client.post('/predict', json={})
        self.assertEqual(response3.status_code, 400)


if __name__ == '__main__':
    unittest.main()