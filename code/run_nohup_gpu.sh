#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

: > logs/resnet18_gpu2.pid
: > logs/plain18_gpu3.pid

# setsid creates an independent session so the runner cannot reap the nohup job.
setsid -f bash -c '
  printf "%s\n" "$$" > logs/resnet18_gpu2.pid
  exec nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
    .venv/bin/python code/train_cifar10_resnet.py \
    --variant resnet18 \
    --epochs 30 \
    --batch-size 128 \
    --num-workers 8 \
    --data-dir data \
    --output-dir outputs \
    > logs/resnet18_gpu2.log 2>&1 < /dev/null
'

setsid -f bash -c '
  printf "%s\n" "$$" > logs/plain18_gpu3.pid
  exec nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
    .venv/bin/python code/train_cifar10_resnet.py \
    --variant plain18 \
    --epochs 30 \
    --batch-size 128 \
    --num-workers 8 \
    --data-dir data \
    --output-dir outputs \
    > logs/plain18_gpu3.log 2>&1 < /dev/null
'

for _ in $(seq 1 50); do
  if [[ -s logs/resnet18_gpu2.pid && -s logs/plain18_gpu3.pid ]]; then
    break
  fi
  sleep 0.1
done

resnet_pid=$(cat logs/resnet18_gpu2.pid)
plain_pid=$(cat logs/plain18_gpu3.pid)

printf 'ResNet-18 PID: %s (GPU 2)\n' "$resnet_pid"
printf 'plain18 PID: %s (GPU 3)\n' "$plain_pid"
printf 'Logs: logs/resnet18_gpu2.log, logs/plain18_gpu3.log\n'
