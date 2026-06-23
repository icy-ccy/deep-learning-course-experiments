import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int = 10, use_residual: bool = True):
        super().__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, use_residual=use_residual)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_residual=use_residual)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_residual=use_residual)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_residual=use_residual)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int, use_residual: bool) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, use_residual=use_residual))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """A conventional six-convolution baseline without residual connections."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(3, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(variant: str) -> nn.Module:
    if variant == "simplecnn":
        return SimpleCNN()
    return CifarResNet(BasicBlock, [2, 2, 2, 2], use_residual=variant == "resnet18")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if args.autoaugment:
        train_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    if args.random_erasing > 0:
        train_ops.append(transforms.RandomErasing(p=args.random_erasing, value="random"))
    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    if args.dry_run:
        train_set = datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=10, transform=train_transform)
        val_set = datasets.FakeData(size=128, image_size=(3, 32, 32), num_classes=10, transform=eval_transform)
        test_set = datasets.FakeData(size=128, image_size=(3, 32, 32), num_classes=10, transform=eval_transform)
    else:
        split_source = datasets.CIFAR10(args.data_dir, train=True, download=True)
        train_full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        val_full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=eval_transform)
        test_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=eval_transform)

        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(split_source), generator=generator).tolist()
        val_indices = indices[: args.val_size]
        train_indices = indices[args.val_size :]
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if is_train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += batch_size

    return total_loss / total, total_correct / total


def save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "variant", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip plotting.")
        return

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_variant(
    args: argparse.Namespace,
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(variant).to(device)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir) / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history: list[dict] = []

    print(
        f"Training {variant} on {device}; parameters={count_parameters(model):,}; "
        f"autoaugment={args.autoaugment}; random_erasing={args.random_erasing}; "
        f"label_smoothing={args.label_smoothing}; amp={amp_enabled}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, train_criterion, device, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(model, val_loader, eval_criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch,
            "variant": variant,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        }
        history.append(row)
        save_history(output_dir / "history.csv", history)
        print(
            f"[{variant}] epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": variant,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "params": count_parameters(model),
                },
                output_dir / "best.pt",
            )

    plot_history(history, output_dir / "curves.png")

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc = run_epoch(model, test_loader, eval_criterion, device)
    summary = {
        "variant": variant,
        "params": count_parameters(model),
        "best_epoch": checkpoint["epoch"],
        "best_val_acc": checkpoint["val_acc"],
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    print(f"[{variant}] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    return summary


def save_summary(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "params", "best_epoch", "best_val_acc", "test_loss", "test_acc"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN variants on CIFAR-10.")
    parser.add_argument(
        "--variant",
        choices=["resnet18", "plain18", "simplecnn", "both", "all"],
        default="resnet18",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autoaugment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use FakeData to verify the pipeline quickly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, test_loader = make_loaders(args)
    if args.variant == "both":
        variants = ["resnet18", "plain18"]
    elif args.variant == "all":
        variants = ["resnet18", "plain18", "simplecnn"]
    else:
        variants = [args.variant]
    summaries = [train_variant(args, variant, train_loader, val_loader, test_loader) for variant in variants]
    summary_name = "summary.csv" if args.variant == "both" else f"summary_{args.variant}.csv"
    save_summary(Path(args.output_dir) / summary_name, summaries)


if __name__ == "__main__":
    main()
