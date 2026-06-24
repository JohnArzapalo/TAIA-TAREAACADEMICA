"""
Graph Convolutional Network (GCN) + Convolución Temporal — Baseline
====================================================================
Copiado de mmskeleton/ops/st_gcn/gconv_origin.py
Sin dependencia de mmcv. Es el BASELINE contra el que comparamos GAT y GIN.

Operación central:
    x_out[w] = Σ_k Σ_v  A[k,v,w] * W_k * x[v]

donde A es la matriz de adyacencia FIJA del grafo esqueleto.
"""

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    """
    GCN espacial + convolución temporal.

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
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A
