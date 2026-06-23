#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs_improved outputs_improved

launch() {
  local gpu=$1
  local variant=$2
  local pid_file="logs_improved/${variant}_gpu${gpu}.pid"
  local log_file="logs_improved/${variant}_gpu${gpu}.log"

  : > "$pid_file"
  setsid -f bash -c "
    printf '%s\\n' \"\$\$\" > '$pid_file'
    exec nohup env CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \\
      .venv/bin/python code/train_cifar10_resnet.py \\
      --variant $variant \\
      --epochs 100 \\
      --batch-size 128 \\
      --num-workers 8 \\
      --data-dir data \\
      --output-dir outputs_improved \\
      --random-erasing 0.25 \\
      --label-smoothing 0.1 \\
      --no-amp \\
      > '$log_file' 2>&1 < /dev/null
  "
}

launch 0 resnet18
launch 1 plain18
launch 2 simplecnn

for _ in $(seq 1 50); do
  if [[ -s logs_improved/resnet18_gpu0.pid && \
        -s logs_improved/plain18_gpu1.pid && \
        -s logs_improved/simplecnn_gpu2.pid ]]; then
    break
  fi
  sleep 0.1
done

printf 'ResNet-18 PID: %s (GPU 0)\n' "$(cat logs_improved/resnet18_gpu0.pid)"
printf 'plain18 PID: %s (GPU 1)\n' "$(cat logs_improved/plain18_gpu1.pid)"
printf 'SimpleCNN PID: %s (GPU 2)\n' "$(cat logs_improved/simplecnn_gpu2.pid)"
printf 'Logs: logs_improved/*.log\n'
