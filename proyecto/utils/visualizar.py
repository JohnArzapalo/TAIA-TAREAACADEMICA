"""
Visualización de grafos, atención y resultados
==============================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Layouts de joints para visualización ─────────────────────────────────────

NTU_JOINT_NAMES = [
    'base_spine', 'mid_spine', 'neck', 'head',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_hip', 'left_knee', 'left_ankle',
    'right_hip', 'right_knee', 'right_ankle',
    'spine', 'tip_left_hand', 'thumb_left',
    'tip_right_hand', 'thumb_right',
    'left_foot', 'right_foot',
    'tip_left_foot', 'tip_right_foot'
]

# Posición 2D aproximada de cada joint (para dibujar el esqueleto)
NTU_JOINT_POS = {
    0:  (0.5, 0.45),   # base_spine
    1:  (0.5, 0.55),   # mid_spine
    2:  (0.5, 0.70),   # neck
    3:  (0.5, 0.80),   # head
    4:  (0.35, 0.65),  # left_shoulder
    5:  (0.25, 0.55),  # left_elbow
    6:  (0.15, 0.45),  # left_wrist
    7:  (0.65, 0.65),  # right_shoulder
    8:  (0.75, 0.55),  # right_elbow
    9:  (0.85, 0.45),  # right_wrist
    10: (0.40, 0.35),  # left_hip
    11: (0.38, 0.20),  # left_knee
    12: (0.36, 0.05),  # left_ankle
    13: (0.60, 0.35),  # right_hip
    14: (0.62, 0.20),  # right_knee
    15: (0.64, 0.05),  # right_ankle
    16: (0.5, 0.60),   # spine
    17: (0.10, 0.40),  # tip_left_hand
    18: (0.12, 0.38),  # thumb_left
    19: (0.90, 0.40),  # tip_right_hand
    20: (0.88, 0.38),  # thumb_right
    21: (0.34, 0.00),  # left_foot
    22: (0.66, 0.00),  # right_foot
    23: (0.32, -0.02), # tip_left_foot
    24: (0.68, -0.02), # tip_right_foot
}


def plot_skeleton_graph(A, title='Grafo del Esqueleto', layout='ntu',
                        save_path=None):
    """
    Dibuja el grafo del esqueleto humano.
    A: (K, V, V) o (V, V) — adyacencia
    """
    if A.ndim == 3:
        A_draw = (A.sum(axis=0) > 0).astype(float)
    else:
        A_draw = (A > 0).astype(float)

    V = A_draw.shape[0]
    pos = NTU_JOINT_POS if V == 25 else {i: (np.cos(2*np.pi*i/V),
                                              np.sin(2*np.pi*i/V))
                                          for i in range(V)}
    names = NTU_JOINT_NAMES if V == 25 else [f'J{i}' for i in range(V)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Aristas
    for i in range(V):
        for j in range(i + 1, V):
            if A_draw[i, j] > 0:
                xi, yi = pos[i]
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj], 'gray', linewidth=1.5, alpha=0.7)

    # Nodos
    for i in range(V):
        x, y = pos[i]
        ax.scatter(x, y, s=100, c='steelblue', zorder=5)
        ax.annotate(f'{i}:{names[i][:8]}', (x, y),
                    textcoords='offset points', xytext=(5, 5), fontsize=6)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 0.95)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Guardado en: {save_path}")
    plt.show()


def plot_attention_heatmap(attention, title='Pesos de Atención GAT',
                           joint_names=None, layer_idx=0, save_path=None):
    """
    Visualiza la matriz de atención de una capa GAT.
    attention: (K, V, V) tensor o numpy array
    """
    if hasattr(attention, 'numpy'):
        att = attention.numpy()
    else:
        att = np.array(attention)

    # Usar la primera partición o promediar
    if att.ndim == 3:
        att_plot = att.mean(axis=0)  # promedio sobre particiones K
    else:
        att_plot = att

    V = att_plot.shape[0]
    names = joint_names if joint_names else [f'J{i}' for i in range(V)]
    short_names = [n[:8] for n in names]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(att_plot, cmap='Blues', aspect='auto')

    ax.set_xticks(range(V))
    ax.set_yticks(range(V))
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('Joint destino', fontsize=11)
    ax.set_ylabel('Joint fuente', fontsize=11)
    ax.set_title(f'{title} — Capa {layer_idx}', fontsize=13)
    plt.colorbar(im, ax=ax, label='Peso de atención α')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(history_dict, save_path=None):
    """
    Grafica curvas de entrenamiento de múltiples modelos.
    history_dict: {'gcn': {'train_acc': [...], 'val_acc': [...], 'loss': [...]},
                   'gat': {...}, 'gin': {...}}
    """
    colors = {'gcn': 'blue', 'gat': 'red', 'gin': 'green'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1 = axes[0]
    ax1.set_title('Accuracy de Validación', fontsize=13)
    for name, hist in history_dict.items():
        if 'val_acc' in hist:
            ax1.plot(hist['val_acc'], color=colors.get(name, 'black'),
                     label=f'ST-{name.upper()}', linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2 = axes[1]
    ax2.set_title('Loss de Entrenamiento', fontsize=13)
    for name, hist in history_dict.items():
        if 'train_loss' in hist:
            ax2.plot(hist['train_loss'], color=colors.get(name, 'black'),
                     label=f'ST-{name.upper()}', linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_bar(results_dict, metric='top1_acc', save_path=None):
    """
    Gráfico de barras comparando métricas finales.
    results_dict: {'gcn': {'top1_acc': 85.2, 'top5_acc': 95.1, ...}, ...}
    """
    models  = list(results_dict.keys())
    values  = [results_dict[m].get(metric, 0) for m in models]
    colors  = ['steelblue', 'tomato', 'mediumseagreen']
    labels  = [f'ST-{m.upper()}' for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors[:len(models)], width=0.5,
                  edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Comparación: {metric.replace("_", " ").title()}', fontsize=13)
    ax.set_ylim(0, max(values) * 1.1 + 2)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
