import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def create_pipeline(
    model_path: str,
    extra_model_path: str,
    device_type: str = "npu",
    device_count: int = 1,
    **kwargs
):
    """创建HiDream-I1管道"""
    
    if device_type.lower() == "npu":
        from .npu_pipeline import NPUHiDreamPipeline
        return NPUHiDreamPipeline(
            model_path=model_path,
            extra_model_path=extra_model_path,
            device_count=device_count,
            **kwargs
        )
    elif device_type.lower() == "cuda":
        from .cuda_pipeline import CUDAHiDreamPipeline
        return CUDAHiDreamPipeline(
            model_path=model_path,
            extra_model_path=extra_model_path,
            device_count=device_count,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported device type: {device_type}")