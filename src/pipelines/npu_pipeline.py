import os
import sys
import subprocess
import tempfile
import json
import logging
from pathlib import Path
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)

class NPUHiDreamPipeline:
    """NPU版本的HiDream-I1管道"""
    
    def __init__(self, model_path: str, extra_model_path: str, device_count: int = 1):
        self.model_path = model_path
        self.extra_model_path = extra_model_path
        self.device_count = device_count
        
        # 智能检测项目路径
        self.project_path = self._detect_project_path()
        
        # 计算分布式配置
        self.tp_size, self.ep_size = self._calculate_distributed_config()
        
        # 创建输出目录
        self.output_dir = Path("generated_images")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🎨 NPU HiDream Pipeline initialized:")
        logger.info(f"  - Project path: {self.project_path}")
        logger.info(f"  - Device count: {device_count}")
        logger.info(f"  - TP size: {self.tp_size}, EP size: {self.ep_size}")
    
    def _detect_project_path(self) -> str:
        """智能检测HiDream-I1项目路径"""
        if "HIDREAM_PROJECT_PATH" in os.environ:
            return os.environ["HIDREAM_PROJECT_PATH"]
        
        possible_paths = [
            "/workspace/HiDream-I1",
            "/data/HiDream-I1",
            Path.home() / "HiDream-I1",
            Path.cwd() / "HiDream-I1",
            Path.cwd().parent / "HiDream-I1",
            Path.cwd()
        ]
        
        for path in possible_paths:
            if Path(path).exists() and (Path(path) / "inference.py").exists():
                logger.info(f"🔍 Auto-detected HiDream-I1 project: {path}")
                return str(path)
        
        raise FileNotFoundError("HiDream-I1 project not found")
    
    def _calculate_distributed_config(self):
        """计算最优分布式配置"""
        if self.device_count == 1:
            return 1, 1
        elif self.device_count == 2:
            return 2, 1
        elif self.device_count == 4:
            return 4, 1  # 或者 2, 2
        elif self.device_count == 8:
            return 4, 2
        else:
            # 默认策略：优先TP并行
            tp_size = min(4, self.device_count)
            ep_size = self.device_count // tp_size
            return tp_size, ep_size
    
    def generate_images(
        self,
        task_id: str,
        request,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[str]:
        """生成图像"""
        try:
            logger.info(f"🚀 Starting image generation for task {task_id}")
            logger.info(f"📝 Prompt: {request.prompt[:100]}...")
            
            if progress_callback:
                progress_callback(10)
            
            # 准备输出目录
            task_output_dir = self.output_dir / task_id
            task_output_dir.mkdir(exist_ok=True)
            
            # 创建临时prompt文件
            prompt_file = task_output_dir / f"{task_id}_prompt.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(request.prompt)
            
            if progress_callback:
                progress_callback(20)
            
            # 执行推理
            if self.device_count > 1:
                image_paths = self._run_distributed_inference(
                    task_id, prompt_file, task_output_dir, request, progress_callback
                )
            else:
                image_paths = self._run_single_inference(
                    task_id, prompt_file, task_output_dir, request, progress_callback
                )
            
            if progress_callback:
                progress_callback(100)
            
            logger.info(f"✅ Generated {len(image_paths)} images for task {task_id}")
            return image_paths
            
        except Exception as e:
            logger.error(f"❌ Image generation failed for task {task_id}: {e}")
            raise e
    
    def _run_distributed_inference(
        self, task_id: str, prompt_file: Path, output_dir: Path, 
        request, progress_callback: Optional[Callable[[int], None]]
    ) -> List[str]:
        """运行分布式推理 - 使用torchrun"""
        
        logger.info(f"🔄 Starting multi-NPU inference with torchrun ({self.device_count} devices)")
        
        env = os.environ.copy()
        device_list = env.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
        
        # 设置环境变量
        env.update({
            "HCCL_TIMEOUT": "7200",
            "HCCL_BUFFSIZE": "2048",
            "HCCL_SINGLE_NODE": "1",
            "HCCL_LOCAL_RANK_NUM": str(self.device_count),
            "HCCL_WHITELIST_DISABLE": "1",
            "HCCL_SECURITY_ENABLE": "0",
            "ACL_STREAM_TIMEOUT": "7200",
            "PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:512",
            "TOKENIZERS_PARALLELISM": "false"
        })
        
        # 构建torchrun命令
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.device_count}",
            "--standalone",
            "--nnodes=1",
            "inference.py",
            "--enable_parallelism", "True",
            "--tp_size", str(self.tp_size),
            "--ep_size", str(self.ep_size),
            "--model_path", self.model_path,
            "--model_path_extra", self.extra_model_path,
            "--prompt_file", str(prompt_file),
            "--prompt_file_type", "plain",
            "--info_file_save_path", str(output_dir / f"{task_id}_info.json"),
            "--save_dir", str(output_dir),
            "--num_images_per_prompt", str(request.num_images_per_prompt),
            "--resolution", request.resolution,
            "--num_inference_steps", str(request.num_inference_steps),
            "--batch_size", "1",
            "--infer_type", "Accuracy"
        ]
        
        return self._execute_inference(cmd, env, progress_callback, output_dir, request.num_images_per_prompt)
    
    def _run_single_inference(
        self, task_id: str, prompt_file: Path, output_dir: Path,
        request, progress_callback: Optional[Callable[[int], None]]
    ) -> List[str]:
        """运行单卡推理"""
        
        logger.info("🔧 Starting single NPU inference")
        
        env = os.environ.copy()
        env.update({
            "ACL_STREAM_TIMEOUT": "7200",
            "PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:512",
            "TOKENIZERS_PARALLELISM": "false",
            "HCCL_DISABLE": "1"
        })
        
        # 构建单卡命令
        cmd = [
            "python3", "inference.py",
            "--model_path", self.model_path,
            "--model_path_extra", self.extra_model_path,
            "--prompt_file", str(prompt_file),
            "--prompt_file_type", "plain",
            "--info_file_save_path", str(output_dir / f"{task_id}_info.json"),
            "--save_dir", str(output_dir),
            "--num_images_per_prompt", str(request.num_images_per_prompt),
            "--resolution", request.resolution,
            "--num_inference_steps", str(request.num_inference_steps),
            "--batch_size", "1",
            "--infer_type", "Accuracy",
            "--device_id", "0"
        ]
        
        return self._execute_inference(cmd, env, progress_callback, output_dir, request.num_images_per_prompt)
    
    def _execute_inference(
        self, cmd: List[str], env: dict, progress_callback: Optional[Callable[[int], None]],
        output_dir: Path, num_images: int
    ) -> List[str]:
        """执行推理命令"""
        
        logger.info(f"🔧 Executing: {' '.join(cmd[:8])}...")
        logger.info(f"🔧 Working in: {self.project_path}")
        logger.info(f"🔧 NPU devices: {env.get('ASCEND_RT_VISIBLE_DEVICES')}")
        
        original_cwd = os.getcwd()
        try:
            # 切换到HiDream-I1项目目录
            os.chdir(self.project_path)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            progress = 30
            output_lines = []
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_line = output.strip()
                    output_lines.append(output_line)
                    logger.debug(f"Inference: {output_line}")
                    
                    # 更新进度
                    if any(keyword in output_line.lower() for keyword in ["step", "sampling", "inference"]):
                        progress = min(progress + 2, 85)
                        if progress_callback:
                            progress_callback(progress)
            
            return_code = process.poll()
            
            if return_code != 0:
                # 如果是多卡失败，尝试单卡
                if self.device_count > 1 and "torchrun" in cmd[0]:
                    error_output = '\n'.join(output_lines[-10:])
                    logger.warning(f"⚠️ Multi-NPU failed, trying single NPU...")
                    logger.warning(f"⚠️ Error: {error_output}")
                    
                    # 强制单卡重试
                    original_device_count = self.device_count
                    self.device_count = 1
                    try:
                        return self._run_single_inference(
                            output_dir.name, output_dir / f"{output_dir.name}_prompt.txt", 
                            output_dir, type('Request', (), {
                                'num_images_per_prompt': num_images,
                                'resolution': '1024x1024',
                                'num_inference_steps': 50
                            })(), progress_callback
                        )
                    finally:
                        self.device_count = original_device_count
                else:
                    error_output = '\n'.join(output_lines[-15:])
                    logger.error(f"❌ Inference failed with code {return_code}")
                    logger.error(f"❌ Error: {error_output}")
                    raise RuntimeError(f"Inference failed with code {return_code}")
            
            # 收集生成的图片
            return self._collect_images(output_dir, num_images)
            
        finally:
            os.chdir(original_cwd)
    
    def _collect_images(self, output_dir: Path, num_images: int) -> List[str]:
        """收集生成的图片"""
        image_paths = []
        
        # 查找图片文件
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            image_files = list(output_dir.glob(ext))
            image_paths.extend([str(p) for p in image_files])
        
        # 按修改时间排序，返回最新的图片
        if image_paths:
            image_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return image_paths[:num_images]
        
        return []
    
    def get_device_info(self) -> dict:
        """获取设备信息"""
        return {
            "device_type": "NPU",
            "device_count": self.device_count,
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
            "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES", ""),
            "project_path": self.project_path
        }
    
    def cleanup_task(self, task_id: str):
        """清理任务资源"""
        try:
            task_dir = self.output_dir / task_id
            if task_dir.exists():
                # 只清理临时文件，保留图片
                temp_files = list(task_dir.glob("*_prompt.txt"))
                for temp_file in temp_files:
                    temp_file.unlink()
                    logger.debug(f"🧹 Cleaned up: {temp_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup task {task_id}: {e}")