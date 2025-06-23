import requests
import time
import json

# API基础URL
BASE_URL = "http://localhost:8088"

def test_single_generation():
    """测试单个图像生成"""
    print("🎨 Testing single image generation...")
    
    # 提交任务
    request_data = {
        "prompt": "A beautiful sunset over a mountain landscape, highly detailed, 8k",
        "resolution": "1024x1024",
        "num_inference_steps": 30,
        "num_images_per_prompt": 2,
        "seed": 42
    }
    
    response = requests.post(f"{BASE_URL}/submit", json=request_data)
    if response.status_code == 200:
        result = response.json()
        task_id = result["requestId"]
        print(f"✅ Task submitted: {task_id}")
        
        # 轮询状态
        while True:
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"📊 Status: {status['status']}, Progress: {status['progress']}%")
                
                if status["status"] == "completed":
                    print(f"🎉 Task completed!")
                    print(f"📸 Generated images: {status['result_urls']}")
                    break
                elif status["status"] == "failed":
                    print(f"❌ Task failed: {status['error']}")
                    break
                    
            time.sleep(2)
    else:
        print(f"❌ Failed to submit task: {response.status_code}")

def test_batch_generation():
    """测试批量图像生成"""
    print("📦 Testing batch image generation...")
    
    batch_data = {
        "requests": [
            {
                "prompt": "A cat sitting on a windowsill",
                "resolution": "1024x1024",
                "num_inference_steps": 25,
                "num_images_per_prompt": 1
            },
            {
                "prompt": "A dog running in a park",
                "resolution": "1024x1024", 
                "num_inference_steps": 25,
                "num_images_per_prompt": 1
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/batch_submit", json=batch_data)
    if response.status_code == 200:
        results = response.json()
        print(f"✅ Batch submitted: {len(results)} tasks")
        
        for result in results:
            print(f"  - Task {result['requestId']}: {result['status']}")

def test_health():
    """测试健康检查"""
    print("🔍 Testing health check...")
    
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✅ Service healthy: {json.dumps(health, indent=2)}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

if __name__ == "__main__":
    print("🚀 HiDream-I1 API Test Suite")
    print("=" * 50)
    
    test_health()
    print()
    
    test_single_generation()
    print()
    
    test_batch_generation()