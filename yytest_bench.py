import requests
import time
import concurrent.futures
import numpy as np
import pandas as pd
from typing import List, Dict

class EmbeddingBenchmark:
    def __init__(self, endpoint: str, payload: Dict, headers: Dict = None):
        """
        初始化性能测试工具
        :param endpoint: Embedding接口URL
        :param payload: 请求体模板
        :param headers: 请求头(可选)
        """
        self.endpoint = endpoint
        self.payload = payload
        self.headers = headers or {'Content-Type': 'application/json'}
        self.latencies = []
        self.errors = 0

    def _send_request(self, text: str) -> float:
        """
        发送单个请求并返回延迟(秒)
        """
        start_time = time.perf_counter()
        try:
            response = requests.post(
                self.endpoint,
                json={**self.payload, "input": text},
                headers=self.headers
            )
            if response.status_code != 200:
                self.errors += 1
                return None
            return time.perf_counter() - start_time
        except Exception as e:
            print(f"Request failed: {str(e)}")
            self.errors += 1
            return None

    def run_test(self, 
                concurrency: int = 10,
                total_requests: int = 100,
                sample_texts: List[str] = None):
        """
        执行性能测试
        :param concurrency: 并发线程数
        :param total_requests: 总请求数
        :param sample_texts: 测试文本样本集
        """
        self.latencies = []
        self.errors = 0
        
        # 生成测试文本数据
        sample_texts = sample_texts or [
            "This is a test sentence for embedding",
            "机器学习模型性能评估方法",
            "自然语言处理技术的最新进展"
        ] * (total_requests // 3 + 1)

        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(total_requests):
                text = sample_texts[i % len(sample_texts)]
                futures.append(executor.submit(self._send_request, text))
            
            for future in concurrent.futures.as_completed(futures):
                latency = future.result()
                if latency is not None:
                    self.latencies.append(latency)

    def analyze_results(self):
        """
        分析并输出性能报告
        """
        if not self.latencies:
            print("No successful requests recorded")
            return

        # 计算统计指标
        stats = {
            "Total Requests": len(self.latencies) + self.errors,
            "Success Rate": f"{len(self.latencies)/(len(self.latencies)+self.errors):.1%}",
            "Average Latency (s)": np.mean(self.latencies),
            "P50 Latency (s)": np.percentile(self.latencies, 50),
            "P95 Latency (s)": np.percentile(self.latencies, 95),
            "Throughput (req/s)": len(self.latencies)/sum(self.latencies),
            "Min Latency (s)": np.min(self.latencies),
            "Max Latency (s)": np.max(self.latencies)
        }

        # 生成DataFrame报告
        report = pd.DataFrame([stats], index=["Metrics"]).T
        print("\nPerformance Report:")
        print(report.to_string(header=False))

        # 延迟分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(self.latencies, bins=20, alpha=0.7)
        plt.title('Latency Distribution')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Request Count')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # 配置测试参数（根据实际接口调整）
    config = {
        "endpoint": "http://10.68.23.173:35000/v1/embeddings",
        "payload": {
            "model": "inspur-bge-large-zh-v1.5",
            "encoding_format": "float"
        },
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"
        }
    }

    # 初始化测试工具
    benchmark = EmbeddingBenchmark(**config)
    
    # 执行测试（参数可根据需要调整）
    benchmark.run_test(
        concurrency=2,        # 并发线程数
        total_requests=10,   # 总请求数
        sample_texts=None      # 可传入自定义测试文本
    )
    
    # 生成分析报告
    benchmark.analyze_results()
