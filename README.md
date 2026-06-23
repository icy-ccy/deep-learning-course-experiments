# Deep Learning Course Experiments

《深度学习导论》期末报告题目一：卷积神经网络（CNN）的演变与图像分类实践。

本仓库包含CIFAR-10上的三组PyTorch实验：

- `resnet18`：适配32x32输入的ResNet-18。
- `plain18`：保持层数和参数量不变、去除残差连接的消融模型。
- `simplecnn`：六层普通卷积神经网络基线。

## 实验结果

| 模型 | 最佳验证准确率 | 测试准确率 | 参数量 |
|---|---:|---:|---:|
| ResNet-18 | 95.68% | 95.29% | 11,173,962 |
| plain18 | 95.26% | 94.80% | 11,173,962 |
| SimpleCNN | 93.96% | 93.24% | 1,148,874 |

## 目录

- `code/`：模型、训练程序和多GPU后台启动脚本。
- `logs/`：三组100轮训练的完整控制台日志。
- `requirements.txt`：Python依赖及CUDA版PyTorch版本。

## 环境与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 code/train_cifar10_resnet.py \
  --variant all \
  --epochs 100 \
  --batch-size 128 \
  --random-erasing 0.25 \
  --label-smoothing 0.1 \
  --no-amp
```

三张GPU并行后台训练：

```bash
bash code/run_improved_nohup_gpu.sh
tail -f logs_improved/resnet18_gpu0.log
```

CIFAR-10由`torchvision`首次运行时自动下载。模型权重和数据集未提交到仓库。
