"""
Comparación de modelos entrenados — ST-GCN vs ST-GAT vs ST-GIN
==============================================================
Lee los resultados guardados por train.py y genera tablas y
gráficos comparativos.

Uso:
    python comparar.py
    python comparar.py --models gcn gat gin
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.visualizar import plot_training_curves, plot_comparison_bar


def load_results(results_dir, models):
    """Carga los resultados JSON de cada modelo."""
    history_dict = {}
    metrics_dict = {}

    for model in models:
        result_file = Path(results_dir) / model / 'resultados.json'
        if not result_file.exists():
            print(f"  [AVISO] No se encontraron resultados para '{model}' "
                  f"en {result_file}")
            continue

        with open(result_file) as f:
            data = json.load(f)

        history_dict[model] = data.get('history', {})
        metrics_dict[model] = {
            'top1_acc': data.get('best_val_acc', 0),
            'n_params': data.get('n_params', 0),
            'epochs':   data.get('epochs', 0),
        }
        print(f"  ✓ {model.upper()}: cargado ({len(history_dict[model].get('val_acc', []))} épocas)")

    return history_dict, metrics_dict


def print_comparison_table(metrics_dict):
    """Imprime tabla comparativa de resultados."""
    print(f"\n{'='*65}")
    print(f"  TABLA COMPARATIVA DE RESULTADOS")
    print(f"{'='*65}")
    print(f"{'Modelo':<12} {'Top-1 Acc':>12} {'Parámetros':>14} {'Mejora vs GCN':>14}")
    print(f"{'-'*55}")

    gcn_acc = metrics_dict.get('gcn', {}).get('top1_acc', None)

    for model, m in sorted(metrics_dict.items()):
        top1    = m.get('top1_acc', 0)
        n_param = m.get('n_params', 0)
        mejora  = ''
        if gcn_acc is not None and model != 'gcn':
            diff = top1 - gcn_acc
            mejora = f"{diff:+.2f}%"

        print(f"  ST-{model.upper():<8} {top1:>10.2f}%  "
              f"{n_param:>12,}  {mejora:>12}")

    print(f"{'='*65}\n")


def analyze_convergence(history_dict):
    """Analiza velocidad de convergencia de cada modelo."""
    print(f"\n── ANÁLISIS DE CONVERGENCIA ────────────────────────────────────")
    for model, hist in history_dict.items():
        val_acc = hist.get('val_acc', [])
        if not val_acc:
            continue
        best_epoch = int(np.argmax(val_acc)) + 1
        best_acc   = max(val_acc)
        # Época en que supera el 80% del mejor accuracy
        threshold  = best_acc * 0.80
        converge_epoch = next((i + 1 for i, a in enumerate(val_acc)
                               if a >= threshold), len(val_acc))
        print(f"  ST-{model.upper():<4}: mejor en época {best_epoch:3d} "
              f"({best_acc:.2f}%) | "
              f"convergencia ~80% en época {converge_epoch:3d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gcn', 'gat', 'gin'])
    parser.add_argument('--results_dir', type=str, default='resultados')
    args = parser.parse_args()

    print(f"\nCargando resultados desde: {args.results_dir}")
    history_dict, metrics_dict = load_results(args.results_dir, args.models)

    if not metrics_dict:
        print("\nNo hay resultados disponibles. Primero corre train.py.")
        print("Ejemplo: python train.py --model gcn --demo")
        return

    # Tabla comparativa
    print_comparison_table(metrics_dict)

    # Análisis de convergencia
    analyze_convergence(history_dict)

    # Gráficos
    results_dir = Path(args.results_dir)
    if history_dict:
        plot_training_curves(
            history_dict,
            save_path=str(results_dir / 'curvas_entrenamiento.png')
        )

    if metrics_dict:
        plot_comparison_bar(
            metrics_dict,
            metric='top1_acc',
            save_path=str(results_dir / 'comparacion_accuracy.png')
        )


if __name__ == '__main__':
    main()
