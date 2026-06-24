"""
ST-GNN: Backbone unificado para comparación GCN / GAT / GIN
============================================================
Basado en ST_GCN_18 de MMSkeleton (mmskeleton/models/backbones/st_gcn_aaai18.py)
con el módulo de convolución de grafo configurable.

Uso:
    # Baseline GCN (idéntico a ST_GCN_18 original)
    model = ST_GNN(in_channels=3, num_class=60,
                   graph_cfg={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                   gnn_type='gcn')

    # Con GAT
    model = ST_GNN(..., gnn_type='gat')

    # Con GIN
    model = ST_GNN(..., gnn_type='gin')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Módulos propios del proyecto (100% independientes de mmcv/mmskeleton)
from .graph import Graph
from .gconv_gcn import ConvTemporalGraphical
from .gconv_gat import ConvTemporalGraphAttention
from .gconv_gin import ConvTemporalGraphIsomorphism

# Mapa de tipos de GNN disponibles
GNN_MODULES = {
    'gcn': ConvTemporalGraphical,
    'gat': ConvTemporalGraphAttention,
    'gin': ConvTemporalGraphIsomorphism,
}


def zero(x):
    return 0


def iden(x):
    return x


class ST_GNN(nn.Module):
    """
    Spatial Temporal Graph Neural Network.

    Arquitectura: 10 bloques ST-GNN con las dimensiones originales de ST-GCN:
        64 → 64 → 64 → 64 → 128 → 128 → 128 → 256 → 256 → 256

    Args:
        in_channels (int): Canales de entrada (3 = x, y, confidence)
        num_class (int): Número de clases (60 para NTU RGB+D, 400 para Kinetics)
        graph_cfg (dict): Config del grafo {'layout': ..., 'strategy': ...}
        gnn_type (str): 'gcn', 'gat', o 'gin'
        edge_importance_weighting (bool): Pesos de importancia de aristas (como ST-GCN)
        data_bn (bool): Batch normalization sobre los datos de entrada
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 gnn_type='gcn',
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        if gnn_type not in GNN_MODULES:
            raise ValueError(f"gnn_type debe ser uno de {list(GNN_MODULES.keys())}, "
                             f"recibido: '{gnn_type}'")

        self.gnn_type = gnn_type
        gcn_module = GNN_MODULES[gnn_type]

        # ── Grafo del esqueleto ────────────────────────────────────────────────
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size  = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # ── Batch normalization de entrada ────────────────────────────────────
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else iden

        # ── 10 bloques ST-GNN ─────────────────────────────────────────────────
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gnn_networks = nn.ModuleList((
            st_gnn_block(in_channels, 64,  kernel_size, 1, residual=False,
                         gcn_module=gcn_module, **kwargs0),
            st_gnn_block(64,  64,  kernel_size, 1, gcn_module=gcn_module, **kwargs),
            st_gnn_block(64,  64,  kernel_size, 1, gcn_module=gcn_module, **kwargs),
            st_gnn_block(64,  64,  kernel_size, 1, gcn_module=gcn_module, **kwargs),
            st_gnn_block(64,  128, kernel_size, 2, gcn_module=gcn_module, **kwargs),
            st_gnn_block(128, 128, kernel_size, 1, gcn_module=gcn_module, **kwargs),
            st_gnn_block(128, 128, kernel_size, 1, gcn_module=gcn_module, **kwargs),
            st_gnn_block(128, 256, kernel_size, 2, gcn_module=gcn_module, **kwargs),
            st_gnn_block(256, 256, kernel_size, 1, gcn_module=gcn_module, **kwargs),
            st_gnn_block(256, 256, kernel_size, 1, gcn_module=gcn_module, **kwargs),
        ))

        # ── Pesos de importancia de aristas ───────────────────────────────────
        # GCN: se usan (igual que el paper original ST-GCN)
        # GAT: NO se usan — la atencion ya aprende pesos por arista
        # GIN: NO se usan — epsilon ya controla el peso del self-loop
        # Razon tecnica: (A>0).float() en GAT/GIN bloquea el gradiente
        # de estos parametros, haciendo que no aprendan nada.
        use_importance = edge_importance_weighting and (gnn_type == 'gcn')
        if use_importance:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gnn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gnn_networks)

        # ── Clasificador final ────────────────────────────────────────────────
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        """
        x: (N, C, T, V, M)
            N = batch, C = 3, T = frames, V = joints, M = personas
        """
        N, C, T, V, M = x.size()

        # Normalización de datos de entrada
        x = x.permute(0, 4, 3, 1, 2).contiguous()   # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        # Forward por los 10 bloques
        for gnn, importance in zip(self.st_gnn_networks, self.edge_importance):
            x, _ = gnn(x, self.A * importance)

        # Global average pooling temporal y espacial
        x = F.avg_pool2d(x, x.size()[2:])            # (N*M, 256, 1, 1)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)       # (N, 256, 1, 1) media sobre M personas

        # Clasificación
        x = self.fcn(x)                               # (N, num_class, 1, 1)
        x = x.view(x.size(0), -1)                     # (N, num_class)

        return x

    def get_attention_maps(self):
        """
        Retorna los mapas de atención de todas las capas GAT.
        Solo válido cuando gnn_type='gat'.
        Returns: lista de tensores (K, V, V) — uno por bloque
        """
        if self.gnn_type != 'gat':
            raise ValueError("get_attention_maps() solo disponible con gnn_type='gat'")
        maps = []
        for block in self.st_gnn_networks:
            if hasattr(block.gcn, 'last_attention') and block.gcn.last_attention is not None:
                maps.append(block.gcn.last_attention.cpu())
        return maps

    def count_parameters(self):
        """Cuenta parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class st_gnn_block(nn.Module):
    """
    Bloque ST-GNN: Graph Conv + Temporal Conv + Residual.
    Idéntico a st_gcn_block pero acepta cualquier módulo GNN.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 gcn_module=None):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        if gcn_module is None:
            gcn_module = ConvTemporalGraphical

        # Módulo de convolución sobre el grafo (GCN / GAT / GIN)
        self.gcn = gcn_module(in_channels, out_channels, kernel_size[1])

        # Convolución temporal (igual en todos los modelos)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # Conexión residual
        if not residual:
            self.residual = zero
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A
