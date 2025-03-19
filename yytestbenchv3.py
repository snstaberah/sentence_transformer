import argparse
import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

# 配置参数
RERANK_API_URL = "http://10.68.23.33:35000/v1/rerank"  # 根据网页2部署地址
API_KEY = os.getenv("API_KEY")  # 从环境变量读取密钥
MODEL_NAME = "inspur-bge-reranker-large"  # 网页2推荐的rerank模型
CONCURRENCY = 1  # 并发线程数
REQUEST_COUNT = 500  # 总请求量
TEST_DOCS = 5  # 每个请求包含的文档数
TEST_QUERY = "医疗问答系统特征"
TEST_TEXTS = [
    "医疗客服系统需要处理症状描述和慢性病特征分析",
    "智能客服系统支持多语言交互和工单管理",
    "糖尿病患者的血糖监测与饮食管理方案",
    "基于深度学习的医学影像分析系统架构",
    "医院预约挂号系统的微服务架构设计"
] * 3  # 生成15个测试文档，每个请求随机抽取

# 性能统计容器
latencies = []
total_docs = 0
success_count = 0

def send_request():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": TEST_QUERY,
        "documents": np.random.choice(TEST_TEXTS, TEST_DOCS, replace=False).tolist(),
        "model": "inspur-bge-reranker-large",
         "top_n": 2
    }
    start_time = time.perf_counter()
    try:
        print("sendrequest", start_time)
        response = requests.post(
            RERANK_API_URL,
            headers=headers,
            json=payload,
            timeout=300
        )
        elapsed = time.perf_counter() - start_time
        print("elapsed", elapsed)
        if response.status_code == 200:
            return (elapsed, len(payload["documents"]), True)
        print("response.status_code",  response.status_code, response.content)
        return (elapsed, 0, False)
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return (0, 0, False)

def run_test():
    global total_docs, success_count, total_process_time
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        start = time.perf_counter()
        futures = [executor.submit(send_request) for _ in range(REQUEST_COUNT)]
        print("end send cost", time.perf_counter()-start)
        start = time.perf_counter()
        for future in futures:
            latency, doc_count, success = future.result()
            latencies.append(latency)
            if success:
                total_docs += doc_count
                success_count += 1
        total_process_time = time.perf_counter()-start
def generate_report():
    sorted_latencies = np.sort(latencies)
    total_time = np.sum(latencies)
    
    report = f"""Rerank接口性能测试报告
================================
测试配置：
- 模型名称：{MODEL_NAME}
- 并发数：{CONCURRENCY}
- 总请求量：{REQUEST_COUNT}
- 总耗时：{total_process_time}
- 单请求文档数：{TEST_DOCS}
- 测试文档库：{len(TEST_TEXTS)}个样例文档

性能指标：
1. 成功率：{success_count/REQUEST_COUNT:.1%}
2. 处理文档总数：{total_docs}
3. 文档处理速率：{total_docs/total_time:.1f} doc/s
4. 请求平均延迟：{np.mean(latencies)*1000:.1f}ms
5. 延迟分布：
   - P50：{np.percentile(sorted_latencies, 50)*1000:.1f}ms
   - P95：{np.percentile(sorted_latencies, 95)*1000:.1f}ms 
   - P99：{np.percentile(sorted_latencies, 99)*1000:.1f}ms
6. 系统吞吐量：{success_count/total_time:.1f} req/s
================================
注：测试结果受服务器资源配置影响"""
    return report

if __name__ == "__main__":
    print("Starting rerank performance test...")
    run_test()
    
    with open("rerank_performance_report.txt", "w") as f:
        f.write(generate_report())
    print("Test completed. Report saved.")
