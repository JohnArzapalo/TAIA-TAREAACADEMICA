"""
Métricas de evaluación y análisis de grafos
============================================
Cubre métricas del curso:
  - Accuracy top-1 y top-5
  - F1-score macro
  - Métricas de grafo (Unidad 5): grado, centralidad, clustering
"""

import numpy as np
import torch


# ── Métricas de clasificación ─────────────────────────────────────────────────

def top_k_accuracy(output, target, k=1):
    """
    Accuracy top-k.
    output: (N, num_class) — logits
    target: (N,) — labels
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        pred = pred.t()                          # (k, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum()
        return (correct_k / batch_size).item()


def f1_score_macro(all_preds, all_labels, num_classes):
    """
    F1-score macro (media sobre todas las clases).
    all_preds, all_labels: listas o arrays de Python
    """
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1_per_class = []
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1)

    return float(np.mean(f1_per_class))


def confusion_matrix(all_preds, all_labels, num_classes):
    """Matriz de confusión normalizada por fila."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for pred, label in zip(all_preds, all_labels):
        matrix[label][pred] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


# ── Métricas de grafo (Unidad 5 del curso) ───────────────────────────────────

def degree_centrality(A):
    """
    Centralidad de grado de cada nodo.
    A: (V, V) numpy array — adyacencia
    Retorna: array (V,) con centralidad normalizada
    """
    V = A.shape[0]
    degree = (A > 0).sum(axis=1).astype(float)
    return degree / (V - 1)


def betweenness_centrality(A):
    """
    Centralidad de intermediación (betweenness).
    Usa el algoritmo de Brandes (BFS para grafos no ponderados).
    A: (V, V) numpy array
    """
    V = A.shape[0]
    betweenness = np.zeros(V)

    for s in range(V):
        # BFS desde nodo s
        stack  = []
        pred   = [[] for _ in range(V)]
        sigma  = np.zeros(V)
        sigma[s] = 1.0
        dist   = np.full(V, -1)
        dist[s] = 0
        queue  = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in range(V):
                if A[v, w] > 0:
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

        delta = np.zeros(V)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalizar
    betweenness /= ((V - 1) * (V - 2))
    return betweenness


def clustering_coefficient(A):
    """
    Coeficiente de clustering local de cada nodo.
    A: (V, V) numpy array
    """
    V = A.shape[0]
    A_bin = (A > 0).astype(float)
    np.fill_diagonal(A_bin, 0)

    coef = np.zeros(V)
    for v in range(V):
        neighbors = np.where(A_bin[v] > 0)[0]
        k = len(neighbors)
        if k < 2:
            coef[v] = 0.0
            continue
        # Triángulos entre vecinos
        triangles = 0
        for i in neighbors:
            for j in neighbors:
                if i != j and A_bin[i, j] > 0:
                    triangles += 1
        coef[v] = triangles / (k * (k - 1))
    return coef


def graph_metrics_summary(A, joint_names=None):
    """
    Resumen de métricas del grafo esqueleto.
    A: (V, V) o (K, V, V) — si K>1, usa la suma sobre particiones
    joint_names: lista de nombres de joints (opcional)
    """
    if A.ndim == 3:
        A_combined = A.sum(axis=0)
    else:
        A_combined = A

    V = A_combined.shape[0]
    if joint_names is None:
        joint_names = [f'joint_{i}' for i in range(V)]

    deg_cent = degree_centrality(A_combined)
    bet_cent = betweenness_centrality(A_combined)
    clust    = clustering_coefficient(A_combined)

    print(f"\n{'Joint':<20} {'Grado':>8} {'Centralidad':>12} {'Betweenness':>12} {'Clustering':>11}")
    print("-" * 65)
    for i in range(V):
        degree = int((A_combined[i] > 0).sum())
        print(f"{joint_names[i]:<20} {degree:>8} {deg_cent[i]:>12.4f} "
              f"{bet_cent[i]:>12.4f} {clust[i]:>11.4f}")

    print(f"\nJoint más central (grado):       {joint_names[np.argmax(deg_cent)]}")
    print(f"Joint más central (betweenness): {joint_names[np.argmax(bet_cent)]}")
    print(f"Clustering promedio del grafo:   {clust.mean():.4f}")

    return {
        'degree_centrality':     deg_cent,
        'betweenness_centrality': bet_cent,
        'clustering_coefficient': clust,
    }
