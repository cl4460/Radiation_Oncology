#!/bin/bash
################################################################################
# Exp A1: Parallel 5-Fold Training Script
# 
# Purpose: Run 5 folds in parallel using GPUs 0-4
# Author: Chengze Li
# Date: 2025-11-06
################################################################################

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXP_DIR="/home/lichengze/Research/DeepFeature/myresearch/Exp1"
PYTHON_SCRIPT="$EXP_DIR/exp1.py"
LOG_DIR="$EXP_DIR/logs"
PID_DIR="$EXP_DIR/pids"

# GPU配置（使用GPU 0-4）
GPUS=(0 1 2 3 4)
N_FOLDS=5

# ============================================================
# Pre-flight Checks
# ============================================================
echo "========================================================================"
echo "Exp A1: 5-Fold Parallel Training"
echo "========================================================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ ERROR: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi
echo "✓ Python script found: $PYTHON_SCRIPT"

# 检查GPU可用性
echo ""
echo "Checking GPU availability..."
for gpu in "${GPUS[@]}"; do
    if nvidia-smi -i $gpu &> /dev/null; then
        mem_free=$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits)
        echo "  GPU $gpu: Available (${mem_free}MB free)"
    else
        echo "❌ ERROR: GPU $gpu not accessible"
        exit 1
    fi
done

# 创建日志和PID目录
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"
echo ""
echo "✓ Log directory: $LOG_DIR"
echo "✓ PID directory: $PID_DIR"

# 清理旧的PID文件
rm -f "$PID_DIR"/*.pid
echo "✓ Cleaned old PID files"

# ============================================================
# Function: Launch Single Fold
# ============================================================
launch_fold() {
    local fold=$1
    local gpu=$2
    local log_file="$LOG_DIR/fold${fold}.log"
    local pid_file="$PID_DIR/fold${fold}.pid"
    
    echo ""
    echo "=========================================="
    echo "Launching Fold $fold on GPU $gpu"
    echo "=========================================="
    echo "  Log file: $log_file"
    echo "  PID file: $pid_file"
    
    # 启动训练进程（后台运行）
    (
        # 子shell环境
        export CUDA_VISIBLE_DEVICES=$gpu
        
        # 记录启动信息
        {
            echo "========================================================================"
            echo "Fold $fold Training - GPU $gpu"
            echo "========================================================================"
            echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "CUDA_VISIBLE_DEVICES: $gpu"
            echo "Working directory: $EXP_DIR"
            echo ""
            
            # 运行Python脚本
            cd "$EXP_DIR"
            python "$PYTHON_SCRIPT" 2>&1
            
            # 记录结束信息
            exit_code=$?
            echo ""
            echo "========================================================================"
            echo "Fold $fold Completed"
            echo "========================================================================"
            echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Exit code: $exit_code"
            
            exit $exit_code
            
        } > "$log_file" 2>&1
        
    ) &
    
    # 保存进程ID
    local pid=$!
    echo $pid > "$pid_file"
    echo "  Process ID: $pid"
    echo "  Status: Running..."
}

# ============================================================
# Function: Monitor Progress
# ============================================================
monitor_progress() {
    echo ""
    echo "========================================================================"
    echo "Monitoring Training Progress"
    echo "========================================================================"
    
    while true; do
        sleep 30  # 每30秒检查一次
        
        local all_done=true
        local status_msg=""
        
        for fold in $(seq 0 $((N_FOLDS-1))); do
            local pid_file="$PID_DIR/fold${fold}.pid"
            local log_file="$LOG_DIR/fold${fold}.log"
            
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file")
                
                if ps -p $pid > /dev/null 2>&1; then
                    # 进程仍在运行
                    all_done=false
                    
                    # 提取最新的epoch信息
                    if [ -f "$log_file" ]; then
                        local last_line=$(tail -5 "$log_file" | grep -E "Epoch|STEP" | tail -1 || echo "Initializing...")
                        status_msg="${status_msg}  Fold $fold (GPU ${GPUS[$fold]}): Running - $last_line\n"
                    else
                        status_msg="${status_msg}  Fold $fold (GPU ${GPUS[$fold]}): Running - Starting up...\n"
                    fi
                else
                    # 进程已结束
                    if grep -q "All corrections verified and applied!" "$log_file" 2>/dev/null; then
                        status_msg="${status_msg}  Fold $fold (GPU ${GPUS[$fold]}): ✓ Completed successfully\n"
                    else
                        status_msg="${status_msg}  Fold $fold (GPU ${GPUS[$fold]}): ✗ Failed or incomplete\n"
                    fi
                fi
            fi
        done
        
        # 打印状态更新
        clear
        echo "========================================================================"
        echo "Training Progress Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================================"
        echo -e "$status_msg"
        
        if [ "$all_done" = true ]; then
            break
        fi
        
        echo ""
        echo "Press Ctrl+C to stop monitoring (processes will continue running)"
        echo "Next update in 30 seconds..."
    done
}

# ============================================================
# Function: Collect Results
# ============================================================
collect_results() {
    echo ""
    echo "========================================================================"
    echo "Collecting Results"
    echo "========================================================================"
    
    local results_file="$EXP_DIR/deepsurv/results.csv"
    local summary_file="$EXP_DIR/deepsurv/summary.json"
    
    if [ -f "$results_file" ]; then
        echo "✓ Results file found: $results_file"
        echo ""
        cat "$results_file"
    else
        echo "⚠ Results file not found: $results_file"
    fi
    
    if [ -f "$summary_file" ]; then
        echo ""
        echo "✓ Summary file found: $summary_file"
        echo ""
        cat "$summary_file"
    else
        echo "⚠ Summary file not found: $summary_file"
    fi
}

# ============================================================
# Function: Check for Errors
# ============================================================
check_errors() {
    echo ""
    echo "========================================================================"
    echo "Error Check"
    echo "========================================================================"
    
    local errors_found=false
    
    for fold in $(seq 0 $((N_FOLDS-1))); do
        local log_file="$LOG_DIR/fold${fold}.log"
        
        if [ -f "$log_file" ]; then
            # 检查常见错误
            if grep -qi "error\|exception\|traceback" "$log_file"; then
                echo "⚠ Fold $fold: Errors detected in log"
                echo "  Last 10 lines of log:"
                tail -10 "$log_file" | sed 's/^/    /'
                errors_found=true
            else
                echo "✓ Fold $fold: No errors detected"
            fi
        else
            echo "⚠ Fold $fold: Log file not found"
            errors_found=true
        fi
    done
    
    if [ "$errors_found" = false ]; then
        echo ""
        echo "✓ All folds completed without errors!"
    fi
}

# ============================================================
# Main Execution
# ============================================================

echo ""
echo "========================================================================"
echo "Launching Parallel Training"
echo "========================================================================"

# 启动所有fold
for fold in $(seq 0 $((N_FOLDS-1))); do
    gpu=${GPUS[$fold]}
    launch_fold $fold $gpu
    sleep 2  # 短暂延迟，避免同时启动导致竞争
done

echo ""
echo "========================================================================"
echo "All folds launched successfully!"
echo "========================================================================"
echo ""
echo "PIDs saved in: $PID_DIR/"
for fold in $(seq 0 $((N_FOLDS-1))); do
    if [ -f "$PID_DIR/fold${fold}.pid" ]; then
        pid=$(cat "$PID_DIR/fold${fold}.pid")
        echo "  Fold $fold: PID $pid (GPU ${GPUS[$fold]})"
    fi
done

echo ""
echo "========================================================================"
echo "Useful Commands:"
echo "========================================================================"
echo "  Monitor logs in real-time:"
for fold in $(seq 0 $((N_FOLDS-1))); do
    echo "    tail -f $LOG_DIR/fold${fold}.log"
done
echo ""
echo "  Check GPU usage:"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "  Kill all processes:"
echo "    pkill -f exp1.py"
echo "========================================================================"

# 等待所有进程完成（可选）
read -p "Do you want to monitor progress? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    monitor_progress
    
    # 等待所有后台进程
    echo ""
    echo "Waiting for all processes to complete..."
    wait
    
    # 收集结果
    collect_results
    
    # 检查错误
    check_errors
    
    echo ""
    echo "========================================================================"
    echo "All Training Completed!"
    echo "========================================================================"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "Training processes are running in the background."
    echo "You can safely close this terminal."
    echo ""
    echo "To check status later, run:"
    echo "  tail -f $LOG_DIR/fold*.log"
fi

exit 0