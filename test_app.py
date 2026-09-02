import unittest
from app import app


class TestApp(unittest.TestCase):
    """Flask 接口测试类"""

    def setUp(self):
        """每个测试用例执行前创建测试客户端"""
        self.client = app.test_client()
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
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn('status', data)
        self.assertIn(data['status'], ['平稳', '轻微抖动', '严重抖动'])

    def test_predict_missing_parameters(self):
        """测试 POST /predict 缺少参数时返回 400"""
        # 只提供 vibration，缺少 temperature
        response = self.client.post('/predict', json={"vibration": 3.5})
        self.assertEqual(response.status_code, 400)

        # 只提供 temperature，缺少 vibration
        response2 = self.client.post('/predict', json={"temperature": 55})
        self.assertEqual(response2.status_code, 400)

        # 完全空的 JSON 对象
        response3 = self.client.post('/predict', json={})
        self.assertEqual(response3.status_code, 400)

    def test_delete_history_endpoint(self):
        """测试删除历史记录接口"""
        # 先插入一条预测记录
        self.client.post('/predict', json={"vibration": 3.5, "temperature": 55})
        # 调用删除接口
        response = self.client.post('/history/delete')
        self.assertEqual(response.status_code, 200)
        # 检查返回消息
        data = response.get_json()
        self.assertIn('message', data)
        # 再查询历史，应为空列表
        history_response = self.client.get('/history')
        self.assertEqual(history_response.status_code, 200)
        history_data = history_response.get_json()
        self.assertEqual(history_data, [])


if __name__ == '__main__':
    unittest.main()