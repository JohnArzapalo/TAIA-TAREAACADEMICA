"""
eval_pku.py — Fase 4
======================
Evalua los 3 modelos (GCN, GAT, GIN) entrenados en NTU RGB+D sobre
el subset de PKU-MMD con clases comunes (sin reentrenamiento).

Metrica principal: top-1 accuracy por modelo.
Metrica de generalizacion: caida = acc_NTU_xsub - acc_PKU (por modelo).

Uso:
    python eval_pku.py \
        --pku_dir     D:/pku-mmd/procesado \
        --ckpt_dir    proyecto/resultados \
        --data_dir    D:/Dataset TAIA/Proyecto/datos/ntu-rgb+d
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
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent / 'proyecto'
sys.path.insert(0, str(ROOT))

from modelos import ST_GNN
from utils.metricas import top_k_accuracy, f1_score_macro


def load_pku_loader(pku_dir, batch=32):
    pku_dir = Path(pku_dir)
    data_path   = pku_dir / 'pku_data.npy'
    labels_path = pku_dir / 'pku_labels.pkl'

    if not data_path.exists():
        raise FileNotFoundError(f"No encontrado: {data_path}\n"
                                f"Ejecuta primero: python preprocess_pku.py ...")

    data   = torch.from_numpy(np.load(str(data_path), mmap_mode='r').copy())
    with open(labels_path, 'rb') as f:
        labels = torch.from_numpy(pickle.load(f)).long()

    dataset = TensorDataset(data.float(), labels)
    loader  = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    print(f"  PKU dataset: {len(dataset)} clips, "
          f"{len(set(labels.tolist()))} clases NTU unicas")
    return loader


def load_ntu_best_acc(ckpt_dir, model_name):
    """Lee la mejor accuracy NTU xsub del JSON de resultados."""
    json_path = Path(ckpt_dir) / model_name / 'resultados.json'
    if not json_path.exists():
        return None
    with open(json_path) as f:
        res = json.load(f)
    return res.get('best_val_acc')


def eval_model(model, loader, device, num_class=60):
    model.eval()
    total_top1, total_top5, n = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, label in loader:
            data  = data.to(device)
            label = label.long().to(device)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                output = model(data)
            total_top1 += top_k_accuracy(output, label, k=1)
            total_top5 += top_k_accuracy(output, label, k=min(5, num_class))
            n += 1
            all_preds.extend(output.argmax(1).cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    top1 = (total_top1 / n) * 100
    top5 = (total_top5 / n) * 100
    f1   = f1_score_macro(all_preds, all_labels, num_class) * 100
    return top1, top5, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pku_dir',   required=True,
                   help='Carpeta con pku_data.npy y pku_labels.pkl')
    p.add_argument('--ckpt_dir',  default='proyecto/resultados',
                   help='Carpeta raiz de checkpoints (contiene gcn/, gat/, gin/)')
    p.add_argument('--batch',     type=int, default=32)
    p.add_argument('--gpu',       type=int, default=0)
    p.add_argument('--num_class', type=int, default=60)
    p.add_argument('--suffix',    default='',
                   help='Sufijo del checkpoint, e.g. "_pku" para cargar mejor_gcn_pku.pth')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    pku_loader = load_pku_loader(args.pku_dir, args.batch)

    print("\n" + "=" * 65)
    print("  EVALUACION CROSS-DATASET: NTU → PKU-MMD")
    print("=" * 65)

    results = {}
    for model_name in ['gcn', 'gat', 'gin']:
        ckpt_path = Path(args.ckpt_dir) / model_name / f'mejor_{model_name}{args.suffix}.pth'
        if not ckpt_path.exists():
            print(f"\n  [{model_name.upper()}] Checkpoint no encontrado: {ckpt_path}")
            continue

        # Cargar modelo
        model = ST_GNN(
            in_channels=3,
            num_class=args.num_class,
            graph_cfg={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            gnn_type=model_name,
            edge_importance_weighting=True,
            data_bn=True,
            dropout=0.5,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        ntu_acc_from_ckpt = ckpt.get('val_acc', None)

        print(f"\n  [{model_name.upper()}] Evaluando en PKU-MMD ...")
        top1, top5, f1 = eval_model(model, pku_loader, device, args.num_class)

        # Mejor acc NTU del JSON (mas fiable que el checkpoint si se hizo resume)
        ntu_acc = load_ntu_best_acc(args.ckpt_dir, model_name) or ntu_acc_from_ckpt
        drop = (ntu_acc - top1) if ntu_acc else None

        results[model_name] = {
            'ntu_acc': ntu_acc, 'pku_top1': top1,
            'pku_top5': top5, 'pku_f1': f1, 'drop': drop
        }

        print(f"    NTU xsub (mejor):  {ntu_acc:.2f}%" if ntu_acc else "    NTU xsub: N/A")
        print(f"    PKU top-1:         {top1:.2f}%")
        print(f"    PKU top-5:         {top5:.2f}%")
        print(f"    PKU F1:            {f1:.2f}%")
        if drop is not None:
            print(f"    Caida (NTU-PKU):   {drop:.2f} pp")

    # Resumen comparativo
    print("\n" + "=" * 65)
    print("  TABLA RESUMEN — Caida de generalizacion")
    print("=" * 65)
    print(f"  {'Modelo':<10} {'NTU xsub':>10} {'PKU top-1':>10} {'Caida':>8} {'PKU F1':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for name in ['gcn', 'gat', 'gin']:
        if name not in results:
            continue
        r = results[name]
        ntu  = f"{r['ntu_acc']:.2f}%" if r['ntu_acc'] else "N/A"
        pku  = f"{r['pku_top1']:.2f}%"
        drop = f"{r['drop']:.2f}pp"  if r['drop'] is not None else "N/A"
        f1   = f"{r['pku_f1']:.2f}%"
        print(f"  ST-{name.upper():<7} {ntu:>10} {pku:>10} {drop:>8} {f1:>8}")

    # Guardar JSON
    suffix_tag = args.suffix if args.suffix else '_zeroshot'
    out_json = Path(args.pku_dir) / f'eval_pku_results{suffix_tag}.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en: {out_json}")
    print("=" * 65)


if __name__ == '__main__':
    main()
