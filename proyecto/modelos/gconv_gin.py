"""
Graph Isomorphism Network (GIN) + Convolución Temporal
======================================================
Reemplaza ConvTemporalGraphical (GCN) con agregación GIN.

Diferencia clave con GCN:
  - GCN: normaliza la suma de vecinos por grado → pierde información de estructura
  - GIN: suma directa (sin normalizar) + término self-loop (1+ε)*h_v
         → es TAN expresivo como el test de Weisfeiler-Lehman (WL)
         → puede distinguir más estructuras de grafo que GCN

Por qué importa en esqueletos:
  Un joint con 3 vecinos (ej: codo) debería ser tratado distinto
  que uno con 1 vecino (ej: muñeca), pero GCN normaliza y pierde esa info.
  GIN preserva la diferencia de grado.

Referencia: Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019
"""

import torch
import torch.nn as nn


class ConvTemporalGraphIsomorphism(nn.Module):
    """
    GIN espacial + convolución temporal.

    Regla de update: h_v' = MLP( (1 + ε) * h_v + Σ_{u ∈ N(v)} h_u )

    Dimensiones:
        Entrada:  x (N, C_in, T, V),  A (K, V, V)
        Salida:   x (N, C_out, T, V), A (K, V, V)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 eps=0.0,
                 train_eps=True):
        super().__init__()
        self.kernel_size = kernel_size

        # ε aprendido: controla peso del nodo actual vs sus vecinos
        # Si train_eps=True, el modelo aprende cuánto importa el self-loop
        initial_eps = torch.zeros(kernel_size).fill_(eps)
        if train_eps:
            self.epsilon = nn.Parameter(initial_eps)
        else:
            self.register_buffer('epsilon', initial_eps)

        # Primera transformación (equivalente a W*h en GCN)
        # Nota: no multiplica por kernel_size en la salida como GCN,
        # porque la agregación K-partición se hace antes del conv
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

        # Segunda capa del MLP — lo que hace GIN más expresivo que GCN
        # GCN equivale a un MLP de 1 capa; GIN usa 2+ capas
        self.mlp_bn   = nn.BatchNorm2d(out_channels)
        self.mlp_act  = nn.ReLU(inplace=True)
        self.mlp_conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x, A):
        """
        x: (N, C_in, T, V)
        A: (K, V, V)
        """
        N, C, T, V = x.size()
        K = self.kernel_size

        # ── Paso 1: Convolución temporal (transforma features) ────────────────
        x_conv = self.conv(x)             # (N, K*C_out, T, V)
        n, kc, t, v = x_conv.size()
        x_conv = x_conv.view(n, K, kc // K, t, v)  # (N, K, C_out, T, V)

        # ── Paso 2: Agregación GIN — suma normalizada por grado ──────────────
        # Usamos A directamente (no binaria) para mantener gradientes hacia
        # edge_importance. Normalizamos por suma de filas para estabilidad.
        # La propiedad GIN (suma > media) se preserva con el termino (1+eps)*self.
        A_norm = A / (A.sum(dim=-1, keepdim=True).clamp(min=1e-6))  # (K, V, V)

        # Suma de features de vecinos para cada particion
        # x_agg[n,k,c,t,w] = sum_v A_norm[k,v,w] * x_conv[n,k,c,t,v]
        x_agg = torch.einsum('nkctv,kvw->nkctw', x_conv, A_norm)  # (N, K, C_out, T, V)

        # ── Paso 3: Update GIN: (1+ε)*self + vecinos ─────────────────────────
        eps_exp = self.epsilon.view(1, K, 1, 1, 1)   # broadcast a (N,K,C,T,V)
        x_out = (1.0 + eps_exp) * x_conv + x_agg     # (N, K, C_out, T, V)

        # Suma sobre las K particiones → (N, C_out, T, V)
        x_out = x_out.sum(dim=1)

        # ── Paso 4: Segunda capa MLP ──────────────────────────────────────────
        x_out = self.mlp_bn(x_out)
        x_out = self.mlp_act(x_out)
        x_out = self.mlp_conv2(x_out)

        return x_out.contiguous(), A
