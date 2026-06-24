"""
Análisis del Grafo Esqueleto — Unidades 2, 3, 4 y 5 del curso
==============================================================
Examina las propiedades estructurales del grafo del esqueleto humano
y compara cómo cada arquitectura GNN lo ve.

Conexión con el sílabo:
  - Unidad 2: definición formal del grafo → A, V, E
  - Unidad 3: tipo de grafo → no dirigido, ponderado, conexo
  - Unidad 4: algoritmos → BFS, caminos cortos
  - Unidad 5: métricas → centralidad, clustering, diámetro

Uso:
    python analizar_grafo.py --layout ntu-rgb+d
    python analizar_grafo.py --layout openpose
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Forzar UTF-8 en consola Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modelos.graph import Graph
from utils.metricas import graph_metrics_summary, degree_centrality, betweenness_centrality
from utils.visualizar import plot_skeleton_graph, NTU_JOINT_NAMES


def shortest_paths_bfs(A_bin):
    """Calcula todas las distancias más cortas con BFS (Unidad 4)."""
    V = A_bin.shape[0]
    dist_matrix = np.full((V, V), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    for src in range(V):
        visited = [False] * V
        visited[src] = True
        queue = [(src, 0)]
        while queue:
            node, d = queue.pop(0)
            for neighbor in range(V):
                if A_bin[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    dist_matrix[src, neighbor] = d + 1
                    queue.append((neighbor, d + 1))

    return dist_matrix


def analyze_graph(layout='ntu-rgb+d', strategy='spatial', save_plots=True):
    print(f"\n{'='*65}")
    print(f"  ANÁLISIS DEL GRAFO: layout={layout}, strategy={strategy}")
    print(f"{'='*65}\n")

    # Construir grafo usando MMSkeleton
    graph = Graph(layout=layout, strategy=strategy)
    A = graph.A        # (K, V, V) — K particiones
    V = A.shape[1]
    K = A.shape[0]

    names = NTU_JOINT_NAMES if V == 25 else [f'Joint_{i}' for i in range(V)]

    # ── Descripción formal del grafo (Unidad 2) ───────────────────────────────
    A_combined = A.sum(axis=0)
    A_bin = (A_combined > 0).astype(float)
    np.fill_diagonal(A_bin, 0)

    num_nodes = V
    num_edges = int(A_bin.sum()) // 2  # no dirigido

    print("── PROPIEDADES BÁSICAS (Unidad 2) ──────────────────────────────")
    print(f"  Nodos (joints):          {num_nodes}")
    print(f"  Aristas (conexiones):    {num_edges}")
    print(f"  Particiones K del grafo: {K}")
    print(f"  Tipo: no dirigido, {'ponderado' if A_combined.max() > 1 else 'binario'}")

    # Verificar conectividad (Unidad 4)
    dist_matrix = shortest_paths_bfs(A_bin)
    is_connected = not np.any(np.isinf(dist_matrix))
    diameter = int(dist_matrix[~np.isinf(dist_matrix)].max())

    print(f"\n── CONECTIVIDAD (Unidad 4) ───────────────────────────────────────")
    print(f"  Grafo conexo:  {is_connected}")
    print(f"  Diámetro:      {diameter} (máxima distancia entre dos joints)")
    mean_dist = dist_matrix[~np.isinf(dist_matrix) & (dist_matrix > 0)].mean()
    print(f"  Distancia media entre joints: {mean_dist:.2f}")

    # ── Métricas de grafo (Unidad 5) ─────────────────────────────────────────
    print(f"\n── MÉTRICAS POR NODO (Unidad 5) ─────────────────────────────────")
    metrics = graph_metrics_summary(A_bin, joint_names=names)

    # ── Distribución de grados ────────────────────────────────────────────────
    degrees = (A_bin > 0).sum(axis=1).astype(int)
    print(f"\n── DISTRIBUCIÓN DE GRADOS ────────────────────────────────────────")
    print(f"  Grado mínimo: {degrees.min()}")
    print(f"  Grado máximo: {degrees.max()}")
    print(f"  Grado medio:  {degrees.mean():.2f}")

    unique, counts = np.unique(degrees, return_counts=True)
    for deg, cnt in zip(unique, counts):
        joints_with_deg = [names[i] for i in range(V) if degrees[i] == deg]
        print(f"  Grado {deg}: {cnt} joints → {', '.join(joints_with_deg[:4])}"
              f"{'...' if cnt > 4 else ''}")

    # ── Particiones del grafo (estrategia espacial) ───────────────────────────
    print(f"\n── PARTICIONES DEL GRAFO (K={K}) ────────────────────────────────")
    partition_names = {
        1: ['Self-loop (propio)'],
        3: ['Self-loop (propio)', 'Centripetal (hacia centro)', 'Centrifugal (hacia extremos)'],
    }
    pnames = partition_names.get(K, [f'Partición {i}' for i in range(K)])
    for k in range(K):
        n_edges_k = int((A[k] > 0).sum())
        print(f"  Partición {k} — {pnames[k]}: {n_edges_k} aristas")

    # ── Visualizaciones ───────────────────────────────────────────────────────
    results_dir = ROOT / 'resultados'
    results_dir.mkdir(exist_ok=True)

    if save_plots:
        # Grafo del esqueleto
        plot_skeleton_graph(
            A_bin, title=f'Grafo Esqueleto — {layout}',
            save_path=str(results_dir / f'grafo_{layout.replace("-", "_")}.png')
        )

        # Mapa de calor de la adyacencia
        fig, axes = plt.subplots(1, K, figsize=(5 * K, 5))
        if K == 1:
            axes = [axes]
        for k in range(K):
            axes[k].imshow(A[k], cmap='Blues')
            axes[k].set_title(f'Partición {k}: {pnames[k]}', fontsize=10)
            axes[k].set_xlabel('Joint destino')
            axes[k].set_ylabel('Joint fuente')
        plt.suptitle(f'Matrices de Adyacencia — {layout} / {strategy}',
                     fontsize=12)
        plt.tight_layout()
        path = str(results_dir / f'adyacencia_{layout.replace("-","_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nPlots guardados en: {results_dir}")
        plt.show()

        # Centralidad de nodos
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(V)
        width = 0.35
        ax.bar(x - width/2, metrics['degree_centrality'],
               width, label='Centralidad de Grado', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, metrics['betweenness_centrality'],
               width, label='Centralidad de Intermediación', color='tomato', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n[:8] for n in names], rotation=90, fontsize=7)
        ax.set_title('Centralidad de Joints del Esqueleto', fontsize=13)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        path2 = str(results_dir / f'centralidad_{layout.replace("-","_")}.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.show()

    return {'A': A, 'metrics': metrics, 'dist_matrix': dist_matrix}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout',   type=str, default='ntu-rgb+d',
                        choices=['ntu-rgb+d', 'openpose', 'ntu_edge', 'coco'])
    parser.add_argument('--strategy', type=str, default='spatial',
                        choices=['uniform', 'distance', 'spatial'])
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    analyze_graph(layout=args.layout, strategy=args.strategy,
                  save_plots=not args.no_plots)


if __name__ == '__main__':
    main()
