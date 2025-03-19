import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

# 配置参数
API_URL = "http://10.68.23.33:35000/v1/embeddings"
API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量读取API KEY
MODEL = "inspur-bge-large-zh-v1.5"
CONCURRENCY = 1  # 并发线程数
REQUEST_COUNT = 500  # 总请求量
TEST_TEXT = "医疗客服问答系统需要处理复杂的症状描述，包括发热、咳嗽、呼吸困难等常见症状，以及类似肠胃炎、糖尿病等慢性病况的详细特征。" * 4  # 约200字测试文本

# 性能统计容器
latencies = []
total_tokens = 0
success_count = 0

def send_request():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": TEST_TEXT,
        "model": MODEL
    }
    approx_tokens = int(len(TEST_TEXT) * 1.3)
    start_time = time.perf_counter()
    try:
        print("sendrequest", start_time)
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
        elapsed = time.perf_counter() - start_time
        print("elapsed", elapsed)
        if response.status_code == 200:
            return (elapsed, approx_tokens, True)
        print("response.status_code",  response.status_code)
        return (elapsed, 0, False)
    except Exception as e:
        print("send request error",  e)
        return (0, 0, False)

def run_test():
    global total_tokens, success_count, total_process_time
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        start = time.perf_counter()
        # print("Starting send",  time.perf_counter())
        futures = [executor.submit(send_request) for _ in range(REQUEST_COUNT)]
        print("end send cost", time.perf_counter()-start)
        start = time.perf_counter()
        for future in futures:
            latency, tokens, success = future.result()
            latencies.append(latency)
            if success:
                total_tokens += tokens
                success_count += 1
        total_process_time = time.perf_counter()-start
def generate_report():
    # 基础计算
    total_time = np.sum(latencies)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    sorted_latencies = np.sort(latencies)
    
    # 统计指标
    metrics = {
        "Total Requests": REQUEST_COUNT,
        "Total Time": total_process_time,
        "Success Rate": f"{success_count/REQUEST_COUNT:.1%}",
        "Tokens Processed": total_tokens,
        "Tokens/s": round(tokens_per_second, 1),
        "First Token Latency(avg)": f"{np.mean(latencies)*1000:.1f}ms",
        "Latency Distribution": {
            "p50": f"{np.percentile(latencies, 50)*1000:.1f}ms",
            "p95": f"{np.percentile(latencies, 95)*1000:.1f}ms",
            "p99": f"{np.percentile(latencies, 99)*1000:.1f}ms"
        },
        "Throughput": f"{success_count/total_time if total_time >0 else 0:.1f} req/s"
    }
    
    # 生成报告
    report = f"""Embedding接口性能测试报告
================================
基础配置：
- 测试模型：{MODEL}
- 并发数：{CONCURRENCY}
- 总请求量：{REQUEST_COUNT}
- 总耗时：{metrics['Total Time']}
- 文本长度：{len(TEST_TEXT)}字符

性能指标：
1. 成功率：{metrics['Success Rate']}
2. 总处理token数：{metrics['Tokens Processed']}
3. 令牌速率：{metrics['Tokens/s']} tokens/秒
4. 首Token延迟(平均)：{metrics['First Token Latency(avg)']}
5. 延迟分布：
   - P50：{metrics['Latency Distribution']['p50']}
   - P95：{metrics['Latency Distribution']['p95']}
   - P99：{metrics['Latency Distribution']['p99']}
6. 系统吞吐量：{metrics['Throughput']}
================================
注：测试结果受网络环境和API速率限制影响"""
    return report

if __name__ == "__main__":
    print("Starting performance test...")
    run_test()
    
    with open("performance_report.txt", "w") as f:
        report = generate_report()
        f.write(report)
    
    print("Test completed. Report saved to performance_report.txt")
