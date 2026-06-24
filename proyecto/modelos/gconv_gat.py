"""
Graph Attention Network (GAT) + Convolución Temporal
=====================================================
Reemplaza ConvTemporalGraphical (GCN) con mecanismo de atención.

Diferencia clave con GCN:
  - GCN: A es fija, todos los vecinos contribuyen con el mismo peso normalizado
  - GAT: A_att es aprendida, el modelo decide cuánto importa cada joint vecino
         según el CONTEXTO (las features actuales)

Referencia: Veličković et al., "Graph Attention Networks", ICLR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTemporalGraphAttention(nn.Module):
    """
    GAT espacial + convolución temporal.

    Dimensiones:
        Entrada:  x (N, C_in, T, V),  A (K, V, V)
        Salida:   x (N, C_out, T, V), A (K, V, V)

        N = batch, C = canales, T = frames, V = joints, K = particiones
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
                 att_dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size

        # Convolución temporal — igual que en GCN original
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

        # Parámetros de atención aditiva (GAT estándar)
        # att_src[k, c]: "cuánto emite" el joint fuente en partición k
        # att_dst[k, c]: "cuánto recibe" el joint destino en partición k
        self.att_src = nn.Parameter(torch.Tensor(kernel_size, in_channels))
        self.att_dst = nn.Parameter(torch.Tensor(kernel_size, in_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.att_dropout = nn.Dropout(p=att_dropout)

        nn.init.xavier_uniform_(self.att_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(-1))

        # Guarda atención del último forward para visualización
        self.last_attention = None

    def forward(self, x, A):
        """
        x: (N, C_in, T, V)
        A: (K, V, V) — adyacencia fija del grafo esqueleto
        """
        N, C, T, V = x.size()
        K = self.kernel_size

        # ── Paso 1: Calcular scores de atención ──────────────────────────────
        # Usamos la media temporal para eficiencia de memoria
        # (evita crear tensores N×K×T×V×V que serían ~360MB por batch)
        x_mean = x.mean(dim=2)          # (N, C, V) — media sobre frames
        x_t = x_mean.permute(0, 2, 1)   # (N, V, C)

        # Score de fuente: e_src[n, k, v] = x[n,v,:] · att_src[k,:]
        score_src = torch.einsum('nvc,kc->nkv', x_t, self.att_src)  # (N, K, V)
        score_dst = torch.einsum('nvc,kc->nkv', x_t, self.att_dst)  # (N, K, V)

        # Score de arista: e[n,k,src,dst] = e_src[src] + e_dst[dst]
        # (N,K,V,1) + (N,K,1,V) → broadcast → (N, K, V_src, V_dst)
        score = score_src.unsqueeze(-1) + score_dst.unsqueeze(-2)
        score = self.leaky_relu(score)

        # ── Paso 2: Enmascarar con la estructura del grafo ────────────────────
        # Solo se calcula atención donde existen aristas reales (A > 0)
        # Joints sin conexión reciben score -inf → alpha ≈ 0 tras softmax
        A_mask = (A > 0).float().unsqueeze(0)          # (1, K, V, V)
        score = score * A_mask + (-1e9) * (1.0 - A_mask)

        # Softmax sobre joints fuente para cada joint destino
        # → alpha[n,k,src,dst]: cuánto contribuye 'src' al update de 'dst'
        alpha = F.softmax(score, dim=2)   # (N, K, V_src, V_dst)
        alpha = self.att_dropout(alpha)

        # Guardar para visualización posterior (media sobre batch)
        self.last_attention = alpha.detach().mean(dim=0)  # (K, V, V)

        # ── Paso 3: Convolución temporal ──────────────────────────────────────
        x = self.conv(x)          # (N, K*C_out, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, K, kc // K, t, v)   # (N, K, C_out, T, V)

        # ── Paso 4: Agregación espacial con atención ─────────────────────────
        # Combinamos alpha (atencion aprendida) con la estructura A del grafo.
        # Esto asegura que el grafo sigue siendo el soporte de la convolucion
        # y ademas los gradientes fluyen correctamente.
        # x_out[c,t,w] = Σ_k Σ_v  alpha[k,v,w] * x_conv[k,c,t,v]
        x = torch.einsum('nkctv,nkvw->nctw', x, alpha)
        # Nota: alpha ya fue mascarado con A (linea anterior), por lo que
        # solo las aristas reales del grafo contribuyen a la agregacion.

        return x.contiguous(), A
