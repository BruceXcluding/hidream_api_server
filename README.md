# HiDream-I1 API Server

🎨 **HiDream-I1 图像生成 API 服务** - 基于NPU的分布式推理服务化部署

## 📋 项目概述

HiDream-I1 API Server 是一个基于FastAPI的图像生成服务，支持单卡和多卡NPU分布式推理。通过RESTful API接口，用户可以轻松地提交图像生成任务并获取结果。

### ✨ 主要特性

- 🚀 **分布式推理** - 支持多NPU分布式推理加速
- 🔄 **异步处理** - 异步任务队列，支持批量处理
- 📊 **实时监控** - 任务状态追踪和进度更新
- 🎯 **自动降级** - 分布式失败时自动降级单卡推理
- 🛠️ **智能配置** - 自动检测环境和设备配置
- 📦 **容器化部署** - 支持Docker和裸机部署

## 🏗️ 项目结构

```
hidream_api_server/
├── examples/
│   └── test_api.py              # API测试示例
├── scripts/
│   └── start_hidream_service.sh # 服务启动脚本
└── src/
    ├── hidream_api.py           # 主API服务
    ├── models.py                # 数据模型定义
    └── pipelines/
        ├── hidream_pipeline_factory.py  # 管道工厂
        └── npu_pipeline.py              # NPU推理管道
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch (支持NPU)
- torch_npu
- FastAPI
- Uvicorn
- HiDream-I1 项目

### 安装依赖

```bash
pip install fastapi uvicorn torch torch_npu pathlib
```

### 启动服务

#### 1. 单卡模式启动

```bash
# 使用默认配置
bash scripts/start_hidream_service.sh

# 指定单个NPU设备
bash scripts/start_hidream_service.sh -d 0
```

#### 2. 多卡分布式启动

```bash
# 4卡分布式推理
bash scripts/start_hidream_service.sh -d 0,1,2,3

# 8卡分布式推理
bash scripts/start_hidream_service.sh -d 0,1,2,3,4,5,6,7
```

#### 3. 完整参数启动

```bash
bash scripts/start_hidream_service.sh \
    -m /data/HiDream-I1-Full \
    -e /data/Llama-3.1-8B-Instruct \
    -p /workspace/HiDream-I1 \
    -d 0,1,2,3 \
    --port 8088
```

### 启动参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model-path` | HiDream-I1-Full模型路径 | `/data/HiDream-I1-Full` |
| `-e, --extra-model-path` | Llama-3.1-8B-Instruct模型路径 | `/data/Llama-3.1-8B-Instruct` |
| `-p, --project-path` | HiDream-I1项目路径 | 自动检测 |
| `-d, --devices` | NPU设备列表(逗号分隔) | `0` |
| `-c, --device-count` | 设备数量 | 从devices计算 |
| `-t, --device-type` | 设备类型(npu/cuda) | `npu` |
| `--port` | 服务端口 | `8088` |
| `--host` | 服务地址 | `0.0.0.0` |

## 📖 API 使用指南

### 基础接口

- **服务状态**: `GET /health`
- **API文档**: `GET /docs`
- **服务信息**: `GET /`

### 图像生成

#### 单张图像生成

```bash
curl -X POST "http://localhost:8088/submit" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A beautiful landscape with mountains and lakes at sunset",
       "resolution": "1024x1024",
       "num_inference_steps": 50,
       "num_images_per_prompt": 1
     }'
```

**响应示例**:
```json
{
  "task_id": "12345678-1234-1234-1234-123456789012",
  "status": "queued",
  "estimated_time": 30,
  "message": "Task submitted successfully"
}
```

#### 批量图像生成

```bash
curl -X POST "http://localhost:8088/batch_submit" \
     -H "Content-Type: application/json" \
     -d '{
       "requests": [
         {
           "prompt": "A beautiful sunset",
           "resolution": "1024x1024",
           "num_inference_steps": 50,
           "num_images_per_prompt": 1
         },
         {
           "prompt": "A mountain landscape",
           "resolution": "1024x1024", 
           "num_inference_steps": 50,
           "num_images_per_prompt": 1
         }
       ]
     }'
```

### 任务状态查询

```bash
curl "http://localhost:8088/status/{task_id}"
```

**响应示例**:
```json
{
  "task_id": "12345678-1234-1234-1234-123456789012",
  "status": "completed",
  "progress": 100,
  "result_urls": [
    "http://localhost:8088/images/image_1.png"
  ],
  "processing_time": 25.6,
  "created_at": "2024-01-01T12:00:00",
  "completed_at": "2024-01-01T12:00:25"
}
```

### 下载结果

```bash
curl "http://localhost:8088/download/{task_id}?image_index=0" -o result.png
```

## 🔧 请求参数详解

### ImageGenerationRequest

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | ✅ | - | 图像生成提示词 |
| `resolution` | string | ❌ | "1024x1024" | 图像分辨率 |
| `num_inference_steps` | int | ❌ | 50 | 推理步数 |
| `num_images_per_prompt` | int | ❌ | 1 | 每个提示词生成图片数量 |

### 分辨率支持

- `1024x1024` (推荐)
- `1024x768`  
- `768x1024`
- `512x512`

## 🎯 分布式推理说明

### 分布式配置策略

| 设备数量 | TP Size | EP Size | 说明 |
|----------|---------|---------|------|  
| 1 | 1 | 1 | 单卡推理 |
| 2 | 2 | 1 | 2卡张量并行 |
| 4 | 4 | 1 | 4卡张量并行 |
| 8 | 4 | 2 | 4卡张量并行 + 2卡专家并行 |

### 环境变量

服务启动时会自动设置以下环境变量：

```bash
# NPU设备配置
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
export DEVICE_COUNT="4"

# 分布式通信配置  
export HCCL_TIMEOUT="7200"
export HCCL_BUFFSIZE="2048"
export HCCL_SINGLE_NODE="1"

# 性能优化配置
export ACL_STREAM_TIMEOUT="7200"
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM="false"
```

### 分布式执行流程

1. **API接收请求** → 添加到任务队列
2. **后台任务处理** → 调用NPU Pipeline
3. **Pipeline选择模式** → 单卡 or 分布式
4. **分布式推理** → torchrun启动多进程推理
5. **结果收集** → 返回生成的图像路径

## 📊 监控和日志

### 健康检查

```bash
curl "http://localhost:8088/health"
```

**响应示例**:
```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "queue_length": 0,
  "active_tasks": 1,
  "total_tasks": 10
}
```

### 日志查看

```bash
# 查看实时日志
tail -f logs/hidream_api_YYYYMMDD_HHMMSS.log

# 查看启动日志
tail -f logs/startup.log
```

### 日志级别

- **INFO**: 一般信息和任务状态
- **DEBUG**: 详细的推理过程
- **WARNING**: 警告信息(如分布式降级)
- **ERROR**: 错误信息

## 🔄 故障处理

### 常见问题

#### 1. torchrun不可用

```bash
# 错误信息
❌ torchrun not found, required for multi-NPU

# 解决方案
pip install torch
# 或者使用单卡模式
bash start_hidream_service.sh -d 0 -c 1
```

#### 2. 分布式推理失败

系统会自动降级到单卡推理：

```
⚠️ Distributed inference failed, trying single NPU...
🔧 Single NPU inference
✅ Single NPU inference completed
```

#### 3. 项目路径检测失败

```bash
# 手动指定项目路径
bash start_hidream_service.sh -p /path/to/HiDream-I1
```

#### 4. 端口被占用

```bash
# 使用其他端口
bash start_hidream_service.sh --port 8089
```

### 重启服务

```bash
# 停止服务
pkill -f "hidream_api"

# 重新启动
bash scripts/start_hidream_service.sh -d 0,1,2,3
```

## 🧪 测试

### 运行测试脚本

```bash
python examples/test_api.py
```

### 性能测试

```bash
# 单个请求测试
time curl -X POST "http://localhost:8088/submit" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "test", "num_inference_steps": 10}'

# 并发测试
for i in {1..5}; do
  curl -X POST "http://localhost:8088/submit" \
       -H "Content-Type: application/json" \
       -d '{"prompt": "test '$i'", "num_inference_steps": 10}' &
done
```

## 🚀 部署建议

### 生产环境配置

```bash
# 推荐配置
bash start_hidream_service.sh \
    -m /data/models/HiDream-I1-Full \
    -e /data/models/Llama-3.1-8B-Instruct \
    -p /workspace/HiDream-I1 \
    -d 0,1,2,3,4,5,6,7 \
    --port 8088 \
    --host 0.0.0.0
```

### 性能优化

1. **模型预加载**: 确保模型在本地SSD存储
2. **内存优化**: 根据设备内存调整batch_size
3. **并发控制**: 监控任务队列长度，避免过载
4. **日志管理**: 定期清理日志文件

### 安全考虑

1. **访问控制**: 生产环境建议配置防火墙和访问限制
2. **HTTPS**: 使用反向代理(如Nginx)配置HTTPS
3. **资源限制**: 设置合理的任务超时和队列长度限制

## 📝 更新日志

### v1.0.0
- ✅ 支持单卡和多卡NPU分布式推理
- ✅ 异步任务队列和状态管理
- ✅ 自动环境检测和配置
- ✅ 完整的API接口和文档

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

请查看项目许可证文件。

---

🎨 **HiDream-I1 API Server** - 让AI图像生成变得简单高效！
