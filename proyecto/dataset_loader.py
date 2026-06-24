"""
Cargadores de dataset — NTU RGB+D y Kinetics
=============================================

Estructura esperada de archivos:

    datos/
    ├── ntu-rgb+d/
    │   ├── xsub_train_data.npy       # (N, 3, 300, 25, 2) float32
    │   ├── xsub_train_label.pkl      # lista de (nombre, clase_int)
    │   ├── xsub_val_data.npy
    │   └── xsub_val_label.pkl
    │
    └── kinetics/
        ├── kinetics_train_data.npy   # (N, 3, 300, 18, 2) float32
        ├── kinetics_train_label.pkl
        ├── kinetics_val_data.npy
        └── kinetics_val_label.pkl

Los archivos .npy/.pkl de NTU RGB+D se obtienen desde:
    https://github.com/yysijie/st-gcn  →  Data Preparation  →  NTU RGB+D

Los archivos de Kinetics (esqueletos pre-extraídos) desde:
    https://github.com/yysijie/st-gcn  →  Data Preparation  →  Kinetics-skeleton

Nota: el dataset Kinetics 5% de Kaggle contiene videos crudos.
Para obtener los esqueletos sin procesar todos los videos,
usar los esqueletos pre-extraídos del repo st-gcn.
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class NTUDataset(Dataset):
    """
    Dataset NTU RGB+D para reconocimiento de acciones con esqueleto.

    Formato de datos: .npy (secuencias) + .pkl (etiquetas)
    Split estándar: xsub (cross-subject) — el más usado en benchmarks

    Augmentaciones de entrenamiento:
      - random_choose: selección aleatoria de frames
      - random_move: transformación afín aleatoria (simula variación de cámara)
    """

    def __init__(self, data_path, label_path,
                 window_size=300,
                 random_choose=False,
                 random_move=False,
                 mmap=True):
        self.data_path     = data_path
        self.window_size   = window_size
        self.random_choose = random_choose
        self.random_move   = random_move
        self._data         = None   # lazy: cada worker abre su propio handle

        # Cargar etiquetas (pequenas, siempre en memoria)
        with open(label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # Obtener shape sin mantener el mmap abierto (evita MemoryError en pickle)
        _tmp = np.load(data_path, mmap_mode='r')
        self._shape = _tmp.shape
        del _tmp

        print(f"NTU RGB+D cargado: {len(self.label)} muestras, "
              f"shape={self._shape}, "
              f"clases únicas={len(set(self.label))}")

    def _get_data(self):
        if self._data is None:
            self._data = np.load(self.data_path, mmap_mode='r')
        return self._data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data  = np.array(self._get_data()[idx])   # (C, T, V, M)
        label = self.label[idx]

        # Augmentación temporal: selección aleatoria de ventana
        if self.random_choose:
            data = self._random_choose(data, self.window_size)
        else:
            data = self._auto_pad(data, self.window_size)

        # Augmentación espacial: transformación afín aleatoria
        if self.random_move:
            data = self._random_move(data)

        return torch.tensor(data, dtype=torch.float32), label

    def _auto_pad(self, data, target_len):
        """Padding con ceros o crop central para llegar a target_len frames."""
        C, T, V, M = data.shape
        if T < target_len:
            pad = np.zeros((C, target_len - T, V, M), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=1)
        elif T > target_len:
            start = (T - target_len) // 2
            data = data[:, start:start + target_len, :, :]
        return data

    def _random_choose(self, data, target_len):
        """Crop temporal aleatorio."""
        C, T, V, M = data.shape
        if T < target_len:
            return self._auto_pad(data, target_len)
        start = np.random.randint(0, T - target_len + 1)
        return data[:, start:start + target_len, :, :]

    def _random_move(self, data):
        """Transformación afín aleatoria sobre coordenadas x,y."""
        C, T, V, M = data.shape
        theta = np.random.uniform(-0.3, 0.3)      # rotación
        tx    = np.random.uniform(-0.15, 0.15)    # traslación x
        ty    = np.random.uniform(-0.15, 0.15)    # traslación y
        scale = np.random.uniform(0.85, 1.15)     # escala

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Aplica solo a los canales x (0) e y (1)
        x = data[0].copy()
        y = data[1].copy()
        data[0] = scale * (cos_t * x - sin_t * y) + tx
        data[1] = scale * (sin_t * x + cos_t * y) + ty
        return data


class KineticsDataset(Dataset):
    """
    Dataset Kinetics con esqueletos pre-extraídos.
    Usado como prueba de GENERALIZACIÓN (no para entrenamiento).

    Formato: .npy + .pkl — igual que NTU pero con 18 joints (OpenPose)

    Diferencias con NTU RGB+D:
      - 18 joints (OpenPose) vs 25 joints (Kinect)
      - Videos de YouTube (más variabilidad de cámara y ruido)
      - 400 clases vs 60 clases
      - Puede tener oclusiones y joints no detectados
    """

    def __init__(self, data_path, label_path, window_size=300, mmap=True):
        with open(label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f, encoding='latin1')

        self.data = np.load(data_path, mmap_mode='r' if mmap else None)
        self.window_size = window_size

        print(f"Kinetics cargado: {len(self.label)} muestras, "
              f"shape={self.data.shape}, "
              f"clases únicas={len(set(self.label))}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data  = np.array(self.data[idx])
        label = self.label[idx]
        data  = self._auto_pad(data, self.window_size)
        return torch.tensor(data, dtype=torch.float32), label

    def _auto_pad(self, data, target_len):
        C, T, V, M = data.shape
        if T < target_len:
            pad = np.zeros((C, target_len - T, V, M), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=1)
        elif T > target_len:
            start = (T - target_len) // 2
            data = data[:, start:start + target_len, :, :]
        return data


def build_ntu_loaders(data_dir, batch_size=32, num_workers=0, window_size=300):
    """
    Construye DataLoaders para NTU RGB+D (split xsub).
    Retorna: (train_loader, val_loader)

    Estructura de carpetas esperada (segun ST-GCN oficial):
        data_dir/
        └── xsub/
            ├── train_data.npy    <- shape (N, 3, 300, 25, 2)
            ├── train_label.pkl   <- (sample_names, labels)
            ├── val_data.npy
            └── val_label.pkl

    data_dir debe apuntar a la carpeta padre de xsub/,
    por ejemplo: datos/ntu-rgb+d
    """
    xsub_dir = os.path.join(data_dir, 'xsub')
    if not os.path.exists(xsub_dir):
        # Intentar sin subcarpeta (por si los archivos estan directamente en data_dir)
        xsub_dir = data_dir

    train_set = NTUDataset(
        data_path  = os.path.join(xsub_dir, 'train_data.npy'),
        label_path = os.path.join(xsub_dir, 'train_label.pkl'),
        window_size=window_size,
        random_choose=False,   # crop central fijo — evita caer en zeros con T<300
        random_move=True,
    )
    val_set = NTUDataset(
        data_path  = os.path.join(xsub_dir, 'val_data.npy'),
        label_path = os.path.join(xsub_dir, 'val_label.pkl'),
        window_size=window_size,
        random_choose=False,
        random_move=False,
    )

    use_pin     = torch.cuda.is_available()
    persistent  = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=use_pin, persistent_workers=persistent
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=use_pin, persistent_workers=persistent
    )
    return train_loader, val_loader


def build_kinetics_loader(data_dir, batch_size=32, num_workers=0, window_size=300):
    """
    Construye DataLoader para Kinetics (solo evaluacion).

    Estructura esperada:
        data_dir/
        ├── val_data.npy    <- shape (N, 3, 300, 18, 2)
        └── val_label.pkl
    """
    val_set = KineticsDataset(
        data_path  = os.path.join(data_dir, 'val_data.npy'),
        label_path = os.path.join(data_dir, 'val_label.pkl'),
        window_size=window_size,
    )
    use_pin = torch.cuda.is_available()
    return torch.utils.data.DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=use_pin
    )
