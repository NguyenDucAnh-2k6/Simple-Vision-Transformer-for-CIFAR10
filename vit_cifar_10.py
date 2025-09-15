"""
Vision Transformer (ViT) for CIFAR-10 classification.

Run: python vit_cifar10.py --epochs 30 --batch-size 128

Dependencies:
  - torch
  - torchvision
  - tqdm (optional)

This script includes:
  - Patch embedding via a Conv2d
  - Transformer encoder built from nn.MultiheadAttention
  - Training and evaluation loops
  - Checkpoint saving

Small, self-contained educational implementation (not optimized for max accuracy).
"""

import argparse
import math
import random
import os
from typing import Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x


# ----------------------------- Utilities -----------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- Model Parts --------------------------------

class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Uses a conv layer with kernel_size=patch_size and stride=patch_size
    to produce patches, then flattens spatial dims.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features: Optional[int] = None, dropout=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        # x: (B, N, C)
        y = self.norm1(x)
        # nn.MultiheadAttention expects (seq_len, batch, embed_dim)
        y = y.permute(1, 0, 2)
        attn_out, _ = self.attn(y, y, y)
        attn_out = attn_out.permute(1, 0, 2)
        x = x + self.dropout1(attn_out)

        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # init
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # head init
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, C)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits


# ----------------------------- Training -----------------------------------


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        acc1 = accuracy(outputs, target, topk=(1,))[0]
        running_acc += acc1 * batch_size / 100.0

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * (running_acc / total)
    print(f"Epoch {epoch}: Train loss {epoch_loss:.4f}, Train acc {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, split='Val'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == target).item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * running_corrects / total
    print(f"{split} loss {epoch_loss:.4f}, {split} acc {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# ----------------------------- Main ---------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='ViT CIFAR-10')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--patch-size', default=4, type=int)
    parser.add_argument('--embed-dim', default=192, type=int)
    parser.add_argument('--depth', default=6, type=int)
    parser.add_argument('--heads', default=3, type=int)
    parser.add_argument('--mlp-ratio', default=4.0, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='./checkpoints')
    parser.add_argument('--resume', default='', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)

    # Data
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = VisionTransformer(
        img_size=32,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed from {args.resume}, start epoch {start_epoch}, best acc {best_acc}")
    history={'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        loss,acc=train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, split='Test')
        history['loss'].append(loss)
        history['acc'].append(acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step()

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        ckpt_path = os.path.join(args.save_dir, f'vit_cifar10_epoch{epoch}.pth')
        torch.save(ckpt, ckpt_path)
        if is_best:
            best_path = os.path.join(args.save_dir, 'vit_cifar10_best.pth')
            torch.save(ckpt, best_path)
            print(f"Saved best model to {best_path}")

    print(f"Training complete. Best test accuracy: {best_acc:.2f}%")
    # Plot all curves
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
