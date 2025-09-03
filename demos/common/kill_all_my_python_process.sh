#!/bin/bash

# 获取当前用户名
CURRENT_USER=$(whoami)

echo "正在查找由用户 '$CURRENT_USER' 运行且命令以 'python -u' 开头的进程..."
echo "--------------------------------------------------"

# 使用 pgrep -af 列出当前用户所有进程的 PID 和完整命令
# 然后通过 awk 精确筛选：
# $2 == "python"  -> 第二个字段必须是 "python"
# $3 == "-u"      -> 第三个字段必须是 "-u"
PROCESS_LIST=$(pgrep -af -u "$CURRENT_USER" | awk '$2 == "python" && $3 == "-u"')

# 检查是否有符合条件的进程
if [ -z "$PROCESS_LIST" ]; then
    echo "没有找到以 'python -u' 命令开头的进程。"
    exit 0
fi

# 统计进程数量
PROCESS_COUNT=$(echo "$PROCESS_LIST" | wc -l)

echo "--- 预备终止的进程列表 ---"
echo "总共发现 $PROCESS_COUNT 个符合条件的进程："

# 提取PID用于后续操作
PIDS_TO_KILL=$(echo "$PROCESS_LIST" | awk '{print $1}')

# 打印将要被kill的进程的详细信息
echo "$PROCESS_LIST" | while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $1}')
    cmd=$(echo "$line" | cut -d' ' -f2-)
    echo "  PID: $pid, 命令: $cmd"
done
echo "--------------------------------"
echo

# ======================================================
#  关键步骤：在这里请求用户确认
# ======================================================
read -p ">>> 是否确认终止以上所有进程？(请输入 y 或 Y 以继续): " confirm
echo

# 检查用户的输入
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "操作已取消，没有进程被终止。"
    exit 0
fi
# ======================================================

# 如果用户确认，则开始循环kill掉所有找到的进程
echo "正在终止进程..."
for pid in $PIDS_TO_KILL; do
    if kill -9 "$pid"; then
        echo "  已成功终止进程 PID: $pid"
    else
        echo "  终止进程 PID: $pid 失败"
    fi
done

echo "--------------------------------------------------"
echo "操作完成。总共终止了 $PROCESS_COUNT 个进程。"
