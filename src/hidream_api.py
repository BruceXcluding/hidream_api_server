import os
import uuid
import asyncio
import threading
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from models import *
from pipelines.hidream_pipeline_factory import create_pipeline
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = FastAPI(title="HiDream-I1 Image Generation API", version="1.0.0")
    
    # 🔥 全局状态管理
    status_dict: Dict[str, dict] = {}
    task_queue = deque()
    
    # 🔥 创建输出目录
    os.makedirs("generated_images", exist_ok=True)
    app.mount("/images", StaticFiles(directory="generated_images"), name="images")
    
    # 🔥 初始化模型管道
    pipeline = None
    
    def init_pipeline():
        nonlocal pipeline
        try:
            model_path = os.getenv("HIDREAM_MODEL_PATH", "/data/HiDream-I1-Full")
            extra_model_path = os.getenv("HIDREAM_EXTRA_MODEL_PATH", "/data/Llama-3.1-8B-Instruct")
            device_type = os.getenv("DEVICE_TYPE", "npu")
            device_count = int(os.getenv("DEVICE_COUNT", "1"))
            
            logger.info(f"🔧 Initializing pipeline with:")
            logger.info(f"  - Model path: {model_path}")
            logger.info(f"  - Extra model path: {extra_model_path}")
            logger.info(f"  - Device: {device_type} ({device_count} devices)")
            
            pipeline = create_pipeline(
                model_path=model_path,
                extra_model_path=extra_model_path,
                device_type=device_type,
                device_count=device_count
            )
            logger.info("🚀 HiDream-I1 pipeline initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise e
    
    def process_task(task_id: str, request: ImageGenerationRequest):
        """处理单个图像生成任务"""
        try:
            logger.info(f"🎨 Starting image generation for task {task_id}")
            logger.info(f"📝 Prompt: {request.prompt[:100]}...")

            # 更新状态
            status_dict[task_id]["status"] = TaskStatus.PROCESSING
            status_dict[task_id]["progress"] = 10

            # 调用模型生成
            start_time = datetime.now()

            result_paths = pipeline.generate_images(
                task_id=task_id,
                request=request,
                progress_callback=lambda p: update_progress(task_id, p)
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # 生成访问URL
            base_url = os.getenv("API_BASE_URL", "http://localhost:8088")
            result_urls = [f"{base_url}/images/{os.path.basename(path)}" 
                          for path in result_paths]

            # 更新完成状态
            status_dict[task_id].update({
                "status": TaskStatus.COMPLETED,
                "progress": 100,
                "result_urls": result_urls,
                "completed_at": end_time.isoformat(),
                "processing_time": processing_time
            })

            logger.info(f"✅ Task {task_id} completed in {processing_time:.2f}s")
            logger.info(f"📸 Generated {len(result_paths)} images")

        except KeyboardInterrupt:
            logger.info(f"🛑 Task {task_id} interrupted by user")
            status_dict[task_id].update({
                "status": TaskStatus.FAILED,
                "error": "Task interrupted",
                "completed_at": datetime.now().isoformat()
            })
            raise
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed: {str(e)}")
            status_dict[task_id].update({
                "status": TaskStatus.FAILED,
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })

    def update_progress(task_id: str, progress: int):
        """更新任务进度"""
        if task_id in status_dict:
            status_dict[task_id]["progress"] = min(progress, 95)  # 生成完成前保持95%
            logger.debug(f"📊 Task {task_id} progress: {progress}%")
    
    def task_worker():
        """后台任务处理器"""
        logger.info("🔄 Task worker started")
        while True:
            try:
                if task_queue and pipeline is not None:
                    task_id, request = task_queue.popleft()
                    logger.info(f"🎯 Processing task {task_id} from queue")
                    process_task(task_id, request)
                else:
                    threading.Event().wait(1)  # 等待1秒
            except Exception as e:
                logger.error(f"❌ Task worker error: {e}")
                threading.Event().wait(5)  # 错误后等待5秒
    
    # 🔥 启动后台任务处理器
    threading.Thread(target=task_worker, daemon=True).start()
    
    @app.on_event("startup")
    async def startup_event():
        """应用启动时初始化"""
        logger.info("🚀 Starting HiDream-I1 API Server...")
        try:
            init_pipeline()
            logger.info("✅ HiDream-I1 API Server ready!")
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            raise e
    
    @app.get("/")
    async def root():
        return {
            "message": "HiDream-I1 Image Generation API", 
            "status": "running",
            "version": "1.0.0"
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "pipeline_ready": pipeline is not None,
            "queue_length": len(task_queue),
            "active_tasks": len([t for t in status_dict.values() 
                               if t["status"] == TaskStatus.PROCESSING]),
            "total_tasks": len(status_dict)
        }
    
    @app.post("/submit", response_model=ImageSubmitResponse)
    async def submit(request: ImageGenerationRequest):
        """提交图像生成任务"""
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        
        task_id = f"img_{uuid.uuid4().hex[:16]}"
        
        # 初始化任务状态
        status_dict[task_id] = {
            "status": TaskStatus.PENDING,
            "progress": 0,
            "result_urls": [],
            "error": "",
            "created_at": datetime.now().isoformat(),
            "request": request.model_dump() if hasattr(request, 'model_dump') else request.dict()
        }
        
        # 添加到队列
        task_queue.append((task_id, request))
        
        logger.info(f"📝 Task submitted: {task_id}")
        logger.info(f"📋 Queue length: {len(task_queue)}")
        
        # 估算处理时间
        estimated_time = estimate_processing_time(request)
        
        return ImageSubmitResponse(
            requestId=task_id,
            status=TaskStatus.PENDING,
            message="任务已提交，正在队列中等待处理",
            estimated_time=estimated_time
        )
    
    @app.post("/batch_submit", response_model=List[ImageSubmitResponse])
    async def batch_submit(batch_request: BatchImageRequest):
        """批量提交图像生成任务"""
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        
        responses = []
        for req in batch_request.requests:
            task_id = f"img_{uuid.uuid4().hex[:16]}"
            
            status_dict[task_id] = {
                "status": TaskStatus.PENDING,
                "progress": 0,
                "result_urls": [],
                "error": "",
                "created_at": datetime.now().isoformat(),
                "request": req.model_dump() if hasattr(req, 'model_dump') else req.dict()
            }
            
            task_queue.append((task_id, req))
            estimated_time = estimate_processing_time(req)
            
            responses.append(ImageSubmitResponse(
                requestId=task_id,
                status=TaskStatus.PENDING,
                message="批量任务已提交",
                estimated_time=estimated_time
            ))
        
        logger.info(f"📦 Batch submitted: {len(responses)} tasks")
        return responses
        
    @app.get("/status/{task_id}", response_model=ImageStatusResponse)
    async def get_status(task_id: str):
        """查询任务状态"""
        if task_id not in status_dict:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_info = status_dict[task_id]
        return ImageStatusResponse(**task_info)
    
    @app.get("/download/{task_id}")
    async def download_result(task_id: str, image_index: int = 0):
        """下载生成的图片"""
        if task_id not in status_dict:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_info = status_dict[task_id]
        if task_info["status"] != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Task not completed")
        
        if not task_info["result_urls"] or image_index >= len(task_info["result_urls"]):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # 从URL提取文件路径
        image_url = task_info["result_urls"][image_index]
        filename = image_url.split("/")[-1]
        file_path = f"generated_images/{filename}"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        return FileResponse(
            file_path,
            media_type="image/png",
            filename=f"{task_id}_image_{image_index}.png"
        )
    
    def estimate_processing_time(request: ImageGenerationRequest) -> int:
        """估算处理时间"""
        base_time = 15  # 基础时间15秒
        
        # 根据参数调整
        step_factor = request.num_inference_steps / 50  # 步数因子
        image_factor = request.num_images_per_prompt      # 图片数量因子
        resolution_factor = 1.0
        
        if "2048" in request.resolution:
            resolution_factor = 2.0
        elif "1536" in request.resolution:
            resolution_factor = 1.5
        
        estimated = int(base_time * step_factor * image_factor * resolution_factor)
        return max(estimated, 5)  # 最少5秒
    
    return app

# 🔥 直接运行模式
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8088,
        log_level="info"
    )