from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    resolution: str = "1024x1024"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    batch_size: int = 1
    infer_type: str = "Accuracy"  # Accuracy, Performance, Profiling

class ImageSubmitResponse(BaseModel):
    requestId: str
    status: TaskStatus
    message: str
    estimated_time: int

class ImageStatusResponse(BaseModel):
    requestId: str
    status: TaskStatus
    progress: int
    result_urls: List[str] = []
    error: str = ""
    created_at: str
    completed_at: Optional[str] = None
    processing_time: Optional[float] = None

class BatchImageRequest(BaseModel):
    requests: List[ImageGenerationRequest]