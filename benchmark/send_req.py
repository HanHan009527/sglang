import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# 配置参数
URL = "http://127.0.0.1:30300/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
PAYLOAD = {
    "model": "DeepSeek-R1",
    "messages": [{"role": "user", "content": "你好"}]
}
CONCURRENCY = 5     # 并发数
TOTAL_REQUESTS = 20 # 总请求数
OUTPUT_FILE = f"request_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def send_request(request_id):
    """发送单个请求并返回详细结果"""
    start_time = time.time()
    
    try:
        # 发送请求
        response = requests.post(
            URL,
            headers=HEADERS,
            json=PAYLOAD,
            timeout=30  # 设置超时时间，避免长时间等待
        )
        
        # 确保正确处理编码
        response.encoding = response.apparent_encoding
        
        # 构建结果对象
        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "status_code": response.status_code,
            "response_time": time.time() - start_time,
            "success": response.status_code == 200,
            "headers": dict(response.headers),
            "body": response.text,  # 完整响应内容
            "error": None
        }
        
    except Exception as e:
        # 异常处理
        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "status_code": None,
            "response_time": time.time() - start_time,
            "success": False,
            "headers": {},
            "body": "",
            "error": str(e)
        }
    
    return result

def run_concurrent_test():
    """执行并发测试并输出结果"""
    print(f"开始并发测试: {URL}")
    print(f"配置: 并发数={CONCURRENCY}, 总请求数={TOTAL_REQUESTS}")
    
    start_time = time.time()
    
    # 使用线程池执行并发请求
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        results = list(executor.map(send_request, range(1, TOTAL_REQUESTS + 1)))
    
    total_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    failed_count = TOTAL_REQUESTS - success_count
    avg_time = sum(r["response_time"] for r in results) / TOTAL_REQUESTS
    max_time = max(r["response_time"] for r in results)
    min_time = min(r["response_time"] for r in results)
    
    # 输出统计摘要
    print("\n=== 测试结果摘要 ===")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"请求总数: {TOTAL_REQUESTS}")
    print(f"成功请求: {success_count}")
    print(f"失败请求: {failed_count}")
    print(f"平均响应时间: {avg_time:.2f}秒")
    print(f"最大响应时间: {max_time:.2f}秒")
    print(f"最小响应时间: {min_time:.2f}秒")
    
    # 保存详细结果到JSON文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {OUTPUT_FILE}")
    
    # 输出失败请求详情
    if failed_count > 0:
        print("\n=== 失败请求详情 ===")
        for r in results:
            if not r["success"]:
                print(f"请求 #{r['request_id']}: {r['error'] or f'状态码 {r['status_code']}'}")

if __name__ == "__main__":
    run_concurrent_test()