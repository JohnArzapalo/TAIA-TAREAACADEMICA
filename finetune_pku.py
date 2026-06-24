"""
finetune_pku.py — Fase 3b
==========================
Fine-tuning de un modelo pre-entrenado en NTU RGB+D sobre PKU-MMD (training split).
LR reducido para evitar olvidar representaciones NTU (catastrophic forgetting).

Uso:
    py -3 finetune_pku.py ^
        --model         gcn ^
        --ckpt_in       proyecto/resultados/gcn/mejor_gcn.pth ^
        --pku_train_dir "D:/Dataset TAIA/Proyecto/PKUMMD/procesado_train" ^
        --pku_val_dir   "D:/Dataset TAIA/Proyecto/PKUMMD/procesado" ^
        --ckpt_out      proyecto/resultados/gcn/mejor_gcn_pku.pth ^
        --epochs 10 --lr 0.001
"""
import argparse
import json
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent / 'proyecto'
sys.path.insert(0, str(ROOT))
from modelos import ST_GNN


def load_loader(data_dir, batch, shuffle):
    data_dir = Path(data_dir)
    data = torch.from_numpy(
        np.load(str(data_dir / 'pku_data.npy'), mmap_mode='r').copy()
    ).float()
    with open(data_dir / 'pku_labels.pkl', 'rb') as f:
        labels = torch.from_numpy(pickle.load(f)).long()
    ds = TensorDataset(data, labels)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)


def top1_acc(output, label):
    return (output.argmax(1) == label).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',           required=True, choices=['gcn', 'gat', 'gin'])
    p.add_argument('--ckpt_in',         required=True,
                   help='Checkpoint NTU pre-entrenado (.pth)')
    p.add_argument('--pku_train_dir',   required=True,
                   help='Carpeta con pku_data.npy y pku_labels.pkl del TRAINING split')
    p.add_argument('--pku_val_dir',     required=True,
                   help='Carpeta con pku_data.npy y pku_labels.pkl del VALIDATION split')
    p.add_argument('--ckpt_out',        required=True,
                   help='Ruta de salida del mejor checkpoint fine-tuned')
    p.add_argument('--epochs',          type=int,   default=10)
    p.add_argument('--lr',              type=float, default=0.001)
    p.add_argument('--batch',           type=int,   default=32)
    p.add_argument('--gpu',             type=int,   default=0)
    p.add_argument('--num_class',       type=int,   default=60)
    p.add_argument('--patience',        type=int,   default=0,
                   help='Early stopping: detener si val_top1 no mejora en N epocas (0=desactivado)')
    p.add_argument('--max_minutes',     type=float, default=0,
                   help='Tope de tiempo total en minutos (0=desactivado). Si se excede, '
                        'termina tras la epoca en curso y guarda el mejor checkpoint visto.')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"  FINE-TUNING ST-{args.model.upper()} | Device: {device}")
    print(f"  LR: {args.lr}  |  Epochs: {args.epochs}")
    print(f"{'='*65}")

    # ── Cargar modelo pre-entrenado NTU ───────────────────────────────
    model = ST_GNN(
        in_channels=3,
        num_class=args.num_class,
        graph_cfg={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
        gnn_type=args.model,
        edge_importance_weighting=True,
        data_bn=True,
        dropout=0.5,
    ).to(device)

    ckpt_in = Path(args.ckpt_in)
    if not ckpt_in.exists():
        print(f"  ERROR: checkpoint no encontrado: {ckpt_in}")
        return

    ckpt = torch.load(str(ckpt_in), map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    ntu_acc = ckpt.get('val_acc', None)
    print(f"\n  Checkpoint cargado: {ckpt_in.name}")
    if ntu_acc:
        print(f"  NTU xsub acc (original): {ntu_acc:.2f}%")

    # ── Data loaders PKU ──────────────────────────────────────────────
    train_loader = load_loader(args.pku_train_dir, args.batch, shuffle=True)
    val_loader   = load_loader(args.pku_val_dir,   args.batch, shuffle=False)
    print(f"\n  Train PKU: {len(train_loader.dataset)} clips  (batch={args.batch})")
    print(f"  Val   PKU: {len(val_loader.dataset)} clips")
    if args.patience:
        print(f"  Early stopping: patience={args.patience} epocas")
    if args.max_minutes:
        print(f"  Tope de tiempo: {args.max_minutes:.0f} min")

    # ── Optimizer y scheduler ─────────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler('cuda') if device.type == 'cuda' else None

    # ── Training loop ─────────────────────────────────────────────────
    import time
    best_val = 0.0
    epochs_no_improve = 0
    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"\n  {'Ep':<4} {'LR':>9} {'TrainLoss':>10} {'TrainTop1':>10} {'ValTop1':>9} {'VRAM':>8} {'seg/ep':>7}")
    print(f"  {'-'*4} {'-'*9} {'-'*10} {'-'*10} {'-'*9} {'-'*8} {'-'*7}")

    for epoch in range(1, args.epochs + 1):
        t_ep_start = time.time()
        # -- Train --
        model.train()
        t_loss, t_acc, n = 0.0, 0.0, 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with autocast('cuda'):
                    output = model(data)
                    loss   = criterion(output, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss   = criterion(output, label)
                loss.backward()
                optimizer.step()
            t_loss += loss.item()
            t_acc  += top1_acc(output.detach(), label)
            n      += 1
        scheduler.step()

        # -- Val --
        model.eval()
        v_acc, nv = 0.0, 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                if scaler:
                    with autocast('cuda'):
                        output = model(data)
                else:
                    output = model(data)
                v_acc += top1_acc(output, label)
                nv    += 1

        lr_now    = scheduler.get_last_lr()[0]
        tr_loss   = t_loss / n
        tr_top1   = (t_acc / n) * 100
        val_top1  = (v_acc / nv) * 100
        ep_secs   = time.time() - t_ep_start
        vram_mb   = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0
        torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None

        marker = ' <-- MEJOR' if val_top1 > best_val else ''
        print(f"  {epoch:<4} {lr_now:>9.6f} {tr_loss:>10.4f} {tr_top1:>9.2f}% {val_top1:>8.2f}% "
              f"{vram_mb:>6.0f}MB {ep_secs:>6.1f}s{marker}", flush=True)

        if val_top1 > best_val:
            best_val = val_top1
            epochs_no_improve = 0
            torch.save({
                'state_dict':   model.state_dict(),
                'val_acc_pku':  val_top1,
                'ntu_acc':      ntu_acc,
                'epoch':        epoch,
                'model':        args.model,
            }, args.ckpt_out)
        else:
            epochs_no_improve += 1

        if args.patience and epochs_no_improve >= args.patience:
            print(f"\n  Early stopping: sin mejora en {args.patience} epocas consecutivas.")
            break

        elapsed_min = (time.time() - t_start) / 60
        if args.max_minutes and elapsed_min >= args.max_minutes:
            print(f"\n  Tope de tiempo alcanzado ({elapsed_min:.1f} min >= {args.max_minutes:.0f} min). Deteniendo.")
            break

    # ── Resultado final ───────────────────────────────────────────────
    print(f"\n  Mejor val PKU (fine-tuned): {best_val:.2f}%")
    if ntu_acc:
        print(f"  Zero-shot (sin FT):         {ntu_acc:.2f}% NTU → ver eval_pku_results.json")
    print(f"  Checkpoint guardado en:     {args.ckpt_out}")

    results_path = Path(args.ckpt_out).parent / f'resultados_pku_{args.model}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'model':        args.model,
            'best_val_pku': best_val,
            'ntu_acc':      ntu_acc,
            'epochs':       args.epochs,
            'lr':           args.lr,
        }, f, indent=2)
    print(f"  JSON guardado en:           {results_path}")


if __name__ == '__main__':
    main()
