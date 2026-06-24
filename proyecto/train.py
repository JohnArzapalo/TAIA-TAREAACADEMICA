"""
Script de entrenamiento - ST-GNN Comparativo
=============================================
Entrena ST-GCN, ST-GAT o ST-GIN en NTU RGB+D (split xsub)
y valida en el propio NTU xsub_val.

Uso:
    python train.py --model gcn
    python train.py --model gat
    python train.py --model gin
    python train.py --model gat --demo    # sin dataset, solo verifica codigo
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modelos import ST_GNN
from dataset_loader import build_ntu_loaders
from utils.metricas import top_k_accuracy, f1_score_macro


def get_args():
    parser = argparse.ArgumentParser(description='Entrenar ST-GNN en NTU RGB+D')
    parser.add_argument('--model',      type=str, default='gcn',
                        choices=['gcn', 'gat', 'gin'])
    parser.add_argument('--data_dir',   type=str,
                        default=str(ROOT.parent / 'datos' / 'ntu-rgb+d'))
    parser.add_argument('--num_class',  type=int, default=60)
    parser.add_argument('--epochs',     type=int, default=80)
    parser.add_argument('--batch',      type=int, default=32)
    parser.add_argument('--lr',         type=float, default=0.1)
    parser.add_argument('--dropout',    type=float, default=0.5)
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--workers',    type=int, default=0)
    parser.add_argument('--window',     type=int, default=150,
                        help='Ventana temporal en frames (default 150, paper=300)')
    parser.add_argument('--save_dir',   type=str, default='resultados')
    parser.add_argument('--demo',       action='store_true',
                        help='Datos sinteticos para verificar codigo sin dataset')
    parser.add_argument('--resume',     type=str, default=None,
                        help='Checkpoint .pth para continuar desde la epoca guardada')
    return parser.parse_args()


def demo_loaders(num_class, batch, window):
    T_demo = min(window, 50)
    V, M   = 25, 2
    def make(n):
        data   = torch.randn(n, 3, T_demo, V, M)
        labels = torch.randint(0, num_class, (n,))
        return DataLoader(TensorDataset(data, labels),
                          batch_size=min(batch, 8), shuffle=True)
    print(f"Modo DEMO - datos sinteticos T={T_demo}, batch<=8")
    return make(32), make(16)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for data, label in loader:
        data  = data.to(device)
        label = label.long().to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            output = model(data)
            loss   = criterion(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_acc  += top_k_accuracy(output, label, k=1)
        n += 1
    return total_loss / n, (total_acc / n) * 100


def eval_one_epoch(model, loader, criterion, device, num_class):
    model.eval()
    total_loss, top1, top5 = 0.0, 0.0, 0.0
    all_preds, all_labels  = [], []
    n = 0
    with torch.no_grad():
        for data, label in loader:
            data  = data.to(device)
            label = label.long().to(device)
            with autocast('cuda'):
                output = model(data)
                loss   = criterion(output, label)
            total_loss += loss.item()
            top1 += top_k_accuracy(output, label, k=1)
            top5 += top_k_accuracy(output, label, k=min(5, num_class))
            n += 1
            all_preds.extend(output.argmax(1).cpu().tolist())
            all_labels.extend(label.cpu().tolist())
    f1 = f1_score_macro(all_preds, all_labels, num_class)
    return total_loss / n, (top1 / n) * 100, (top5 / n) * 100, f1 * 100


def main():
    args   = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Modelo:    ST-{args.model.upper()}")
    print(f"  Dataset:   NTU RGB+D {args.num_class} clases (split xsub)")
    print(f"  Device:    {device}  |  AMP: {use_amp}")
    print(f"  Epocas:    {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"  Ventana T: {args.window} frames")
    print("=" * 65)

    if args.demo:
        train_loader, val_loader = demo_loaders(args.num_class, args.batch, args.window)
    else:
        print(f"Cargando NTU RGB+D desde: {args.data_dir}")
        train_loader, val_loader = build_ntu_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch,
            num_workers=args.workers,
            window_size=args.window,
        )

    model = ST_GNN(
        in_channels=3,
        num_class=args.num_class,
        graph_cfg={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
        gnn_type=args.model,
        edge_importance_weighting=True,
        data_bn=True,
        dropout=args.dropout,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Parametros entrenables: {n_params:,}\n")

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, nesterov=True, weight_decay=1e-4
    )
    m1 = max(1, args.epochs // 4)
    m2 = max(m1 + 1, args.epochs // 2)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[m1, m2], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler('cuda', enabled=use_amp)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_top5':   [], 'val_f1':    [],
    }
    best_val_acc = 0.0
    start_epoch  = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_val_acc = ckpt['val_acc']
        start_epoch  = ckpt['epoch'] + 1
        # Fine-tuning: LR fijo en el valor pasado por --lr (usa 0.001)
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr
        scheduler = None  # sin decay adicional, ya estamos en fine-tuning
        print(f"Checkpoint cargado: epoca {ckpt['epoch']}, "
              f"mejor val={best_val_acc:.2f}% → continuando desde epoca {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_top1, val_top5, val_f1 = eval_one_epoch(
            model, val_loader, criterion, device, args.num_class)

        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_top1)
        history['val_top5'].append(val_top5)
        history['val_f1'].append(val_f1)

        marker = " <-- mejor" if val_top1 > best_val_acc else ""
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None \
                     else optimizer.param_groups[0]['lr']
        print(f"[{epoch:3d}/{args.epochs}] "
              f"loss={tr_loss:.3f} | "
              f"train={tr_acc:.1f}% | "
              f"val-top1={val_top1:.1f}% | "
              f"val-top5={val_top5:.1f}% | "
              f"f1={val_f1:.1f}% | "
              f"lr={current_lr:.5f} | "
              f"{elapsed:.0f}s{marker}")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(
                {'epoch': epoch, 'state_dict': model.state_dict(),
                 'val_acc': val_top1, 'optimizer': optimizer.state_dict()},
                save_dir / f'mejor_{args.model}.pth'
            )

    results = {
        'model':        args.model,
        'dataset':      f'NTU-RGB+D-{args.num_class}',
        'epochs':       args.epochs,
        'window':       args.window,
        'best_val_acc': best_val_acc,
        'n_params':     n_params,
        'history':      history,
    }
    with open(save_dir / 'resultados.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 65)
    print(f"  Mejor Val Accuracy (NTU xsub): {best_val_acc:.2f}%")
    print(f"  Checkpoint: {save_dir / f'mejor_{args.model}.pth'}")
    print("=" * 65)


if __name__ == '__main__':
    main()
