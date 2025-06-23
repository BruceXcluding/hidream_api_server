#!/bin/bash

set -e

# 🎨 HiDream-I1 API 服务启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 默认配置
DEVICE_TYPE="npu"
DEVICE_COUNT=""
PORT=8088
HOST="0.0.0.0"
HIDREAM_MODEL_PATH="/data/HiDream-I1-Full"
HIDREAM_EXTRA_MODEL_PATH="/data/Llama-3.1-8B-Instruct"
HIDREAM_PROJECT_PATH=""

# 显示帮助信息
show_help() {
    echo "🎨 HiDream-I1 API Service Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model-path PATH          HiDream-I1-Full模型路径 (默认: /data/HiDream-I1-Full)"
    echo "  -e, --extra-model-path PATH    Llama-3.1-8B-Instruct模型路径 (默认: /data/Llama-3.1-8B-Instruct)"
    echo "  -p, --project-path PATH        HiDream-I1项目路径 (自动检测或手动指定)"
    echo "  -d, --devices DEVICES          NPU设备列表，逗号分隔 (如: 4,5,6,7)"
    echo "  -c, --device-count COUNT       设备数量 (自动从devices计算)"
    echo "  -t, --device-type TYPE         设备类型: npu, cuda (默认: npu)"
    echo "  --port PORT                    服务端口 (默认: 8088)"
    echo "  --host HOST                    服务地址 (默认: 0.0.0.0)"
    echo "  -h, --help                     显示此帮助信息"
    echo ""
    echo "Examples:"
    echo "  # 使用默认配置启动"
    echo "  $0"
    echo ""
    echo "  # 指定NPU设备启动"
    echo "  $0 -d 4,5,6,7"
    echo ""
    echo "  # 完整配置启动"
    echo "  $0 -m /data/HiDream-I1-Full -e /data/Llama-3.1-8B-Instruct -p /workspace/HiDream-I1 -d 4,5,6,7"
    echo ""
    echo "  # 单卡模式"
    echo "  $0 -d 4 -c 1"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            HIDREAM_MODEL_PATH="$2"
            shift 2
            ;;
        -e|--extra-model-path)
            HIDREAM_EXTRA_MODEL_PATH="$2"
            shift 2
            ;;
        -p|--project-path)
            HIDREAM_PROJECT_PATH="$2"
            shift 2
            ;;
        -d|--devices)
            export ASCEND_RT_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        -c|--device-count)
            DEVICE_COUNT="$2"
            shift 2
            ;;
        -t|--device-type)
            DEVICE_TYPE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🎨 HiDream-I1 API Service Startup${NC}"

# 🔥 智能设备计数
if [ -n "$ASCEND_RT_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra DEVICE_ARRAY <<< "$ASCEND_RT_VISIBLE_DEVICES"
    DETECTED_DEVICE_COUNT=${#DEVICE_ARRAY[@]}
    DEVICE_COUNT=${DEVICE_COUNT:-$DETECTED_DEVICE_COUNT}
else
    DEVICE_COUNT=${DEVICE_COUNT:-1}
    export ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((DEVICE_COUNT-1)))
fi

# 🔥 智能项目路径检测
if [ -z "$HIDREAM_PROJECT_PATH" ]; then
    echo "🔍 Auto-detecting HiDream-I1 project path..."
    POSSIBLE_PATHS=(
        "/workspace/HiDream-I1"
        "/data/HiDream-I1" 
        "$HOME/HiDream-I1"
        "$(pwd)/HiDream-I1"
        "$(dirname $(pwd))/HiDream-I1"
        "$(pwd)"
    )
    
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -f "$path/inference.py" ]; then
            HIDREAM_PROJECT_PATH="$path"
            echo "  ✅ Found: $path"
            break
        fi
    done
    
    if [ -z "$HIDREAM_PROJECT_PATH" ]; then
        echo -e "${RED}❌ HiDream-I1 project not found in common paths${NC}"
        echo "Please specify with: -p /path/to/HiDream-I1"
        exit 1
    fi
fi

echo -e "${BLUE}📋 Configuration:${NC}"
echo "  - Device: $DEVICE_TYPE ($DEVICE_COUNT devices)"
echo "  - NPU Devices: $ASCEND_RT_VISIBLE_DEVICES"
echo "  - Server: $HOST:$PORT"
echo "  - Model: $HIDREAM_MODEL_PATH"
echo "  - Extra Model: $HIDREAM_EXTRA_MODEL_PATH"
echo "  - Project Path: $HIDREAM_PROJECT_PATH"

# 验证路径
echo "🔍 Validating paths..."
if [ ! -d "$HIDREAM_MODEL_PATH" ]; then
    echo -e "${RED}❌ Model path not found: $HIDREAM_MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$HIDREAM_EXTRA_MODEL_PATH" ]; then
    echo -e "${RED}❌ Extra model path not found: $HIDREAM_EXTRA_MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$HIDREAM_PROJECT_PATH" ]; then
    echo -e "${RED}❌ HiDream-I1 project path not found: $HIDREAM_PROJECT_PATH${NC}"
    exit 1
fi

if [ ! -f "$HIDREAM_PROJECT_PATH/inference.py" ]; then
    echo -e "${RED}❌ inference.py not found in: $HIDREAM_PROJECT_PATH${NC}"
    exit 1
fi

echo "  ✅ All paths validated"

# 创建必要目录
mkdir -p logs
mkdir -p generated_images

# 环境变量设置
export HIDREAM_MODEL_PATH="$HIDREAM_MODEL_PATH"
export HIDREAM_EXTRA_MODEL_PATH="$HIDREAM_EXTRA_MODEL_PATH"
export HIDREAM_PROJECT_PATH="$HIDREAM_PROJECT_PATH"
export DEVICE_TYPE="$DEVICE_TYPE"
export DEVICE_COUNT="$DEVICE_COUNT"

# 🔥 NPU环境设置
if [ "$DEVICE_TYPE" = "npu" ]; then
    echo -e "${BLUE}📱 NPU Environment Setup...${NC}"
    
    # 检查torchrun可用性
    if [ "$DEVICE_COUNT" -gt 1 ]; then
        echo "🔧 Checking torchrun for distributed inference..."
        if command -v torchrun >/dev/null 2>&1; then
            echo "  ✅ torchrun found: $(which torchrun)"
        else
            echo "  ❌ torchrun not found, required for multi-NPU"
            echo "  💡 Install with: pip install torch"
            exit 1
        fi
    fi
    
    # NPU优化配置
    export ACL_STREAM_TIMEOUT="7200"
    export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"
    export TOKENIZERS_PARALLELISM="false"
    
    if [ "$DEVICE_COUNT" -gt 1 ]; then
        # 多卡配置
        export HCCL_TIMEOUT="7200"
        export HCCL_BUFFSIZE="2048"
        export HCCL_SINGLE_NODE="1"
        export HCCL_LOCAL_RANK_NUM="$DEVICE_COUNT"
        export HCCL_WHITELIST_DISABLE="1"
        export HCCL_SECURITY_ENABLE="0"
        echo "  - Multi-NPU mode: $DEVICE_COUNT devices"
        echo "  - Torchrun ready for distributed inference"
    else
        # 单卡配置
        export HCCL_DISABLE="1"
        echo "  - Single-NPU mode"
    fi
    
    echo "  - NPU Devices: $ASCEND_RT_VISIBLE_DEVICES"
    echo "  - Memory Config: max_split_size_mb=512"
fi

# 清理旧进程
echo -e "${YELLOW}🧹 Cleaning up old processes...${NC}"
pkill -f "hidream_api.py" 2>/dev/null || true
pkill -f "uvicorn.*hidream" 2>/dev/null || true
sleep 2

# 启动服务
echo -e "${GREEN}🚀 Starting HiDream-I1 API Service...${NC}"

LOG_FILE="logs/hidream_api_$(date +%Y%m%d_%H%M%S).log"

# 切换到项目目录并启动
cd "$(dirname "$0")/../src"

python3 -m uvicorn hidream_api:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level info \
    --access-log \
    2>&1 | tee "../$LOG_FILE" &

SERVER_PID=$!

# 等待服务启动
echo -e "${BLUE}⏳ Waiting for service to start...${NC}"
sleep 5

# 检查服务状态
if kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${GREEN}✅ HiDream-I1 API Service started successfully!${NC}"
    echo -e "${BLUE}🔗 Service URL: http://$HOST:$PORT${NC}"
    echo -e "${BLUE}📖 API Docs: http://$HOST:$PORT/docs${NC}"
    echo -e "${BLUE}🔍 Health Check: http://$HOST:$PORT/health${NC}"
    echo -e "${BLUE}📊 Logs: tail -f $LOG_FILE${NC}"
    
    echo -e "${BLUE}📋 Final Configuration:${NC}"
    echo "  - NPU Devices: $ASCEND_RT_VISIBLE_DEVICES"
    echo "  - Device Count: $DEVICE_COUNT"
    echo "  - Process ID: $SERVER_PID"
    echo "  - Project Path: $HIDREAM_PROJECT_PATH"
    
    # 等待用户中断
    trap "echo -e '${YELLOW}🛑 Shutting down...${NC}'; kill $SERVER_PID; exit 0" INT
    wait $SERVER_PID
else
    echo -e "${RED}❌ Failed to start HiDream-I1 API Service${NC}"
    echo -e "${RED}💡 Check the log file: $LOG_FILE${NC}"
    exit 1
fi