"""
Preprocesamiento NTU RGB+D - Solo split xsub (cross-subject)
=============================================================
Script completamente independiente — no requiere instalar skvideo
ni ninguna dependencia extra del repo ST-GCN.

Convierte los 56,880 archivos .skeleton a formato .npy + .pkl
que espera nuestro dataset_loader.py

Tiempo estimado: 1-2 horas
Espacio generado: ~7.7 GB en datos/ntu-rgb+d/xsub/

Uso:
    py -3 preprocess_ntu.py
"""

import os
import sys
import pickle
import numpy as np
from numpy.lib.format import open_memmap

# ── Rutas ────────────────────────────────────────────────────────────────────

DATA_PATH    = r"D:\Dataset TAIA\Proyecto\nturgb+d_skeletons"
IGNORED_PATH = r"C:\Users\johnm\OneDrive\Desktop\TA IA - TAREA ACADEMICA\st-gcn\resource\NTU-RGB-D\samples_with_missing_skeletons.txt"
OUT_FOLDER   = r"D:\Dataset TAIA\Proyecto\datos\ntu-rgb+d"

# ── Configuracion xsub ───────────────────────────────────────────────────────

TRAINING_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
MAX_BODY  = 2
NUM_JOINT = 25
MAX_FRAME = 300

# ── Lectura de archivos .skeleton ─────────────────────────────────────────────

def read_skeleton(file):
    with open(file, 'r') as f:
        seq = {}
        seq['numFrame'] = int(f.readline())
        seq['frameInfo'] = []
        for _ in range(seq['numFrame']):
            frame = {'numBody': int(f.readline()), 'bodyInfo': []}
            for _ in range(frame['numBody']):
                keys = ['bodyID','clipedEdges','handLeftConfidence','handLeftState',
                        'handRightConfidence','handRightState','isResticted',
                        'leanX','leanY','trackingState']
                body = {k: float(v) for k, v in zip(keys, f.readline().split())}
                body['numJoint'] = int(f.readline())
                body['jointInfo'] = []
                for _ in range(body['numJoint']):
                    jkeys = ['x','y','z','depthX','depthY','colorX','colorY',
                             'orientationW','orientationX','orientationY',
                             'orientationZ','trackingState']
                    body['jointInfo'].append(
                        {k: float(v) for k, v in zip(jkeys, f.readline().split())}
                    )
                frame['bodyInfo'].append(body)
            seq['frameInfo'].append(frame)
    return seq


def read_xyz(file, max_body=2, num_joint=25):
    seq = read_skeleton(file)
    data = np.zeros((3, seq['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
    return data

# ── Generacion del split ──────────────────────────────────────────────────────

def gendata(data_path, out_path, ignored_sample_path, part='train'):
    if ignored_sample_path and os.path.exists(ignored_sample_path):
        with open(ignored_sample_path, 'r') as f:
            ignored = set(line.strip() + '.skeleton' for line in f)
    else:
        ignored = set()

    sample_name, sample_label = [], []
    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith('.skeleton') or filename in ignored:
            continue
        action  = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        is_train = (subject in TRAINING_SUBJECTS)

        if (part == 'train' and is_train) or (part == 'val' and not is_train):
            sample_name.append(filename)
            sample_label.append(action - 1)

    # Guardar etiquetas
    label_path = os.path.join(out_path, f'{part}_label.pkl')
    with open(label_path, 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    # Crear array numpy en disco (memory-mapped)
    data_out = open_memmap(
        os.path.join(out_path, f'{part}_data.npy'),
        dtype='float32', mode='w+',
        shape=(len(sample_label), 3, MAX_FRAME, NUM_JOINT, MAX_BODY)
    )

    n = len(sample_name)
    for i, s in enumerate(sample_name):
        # Barra de progreso simple
        pct = (i + 1) / n
        bar = int(40 * pct)
        sys.stdout.write(f'\r  [{i+1:>5}/{n}] [{"#"*bar}{"-"*(40-bar)}] {pct*100:.1f}%')
        sys.stdout.flush()

        data = read_xyz(os.path.join(data_path, s),
                        max_body=MAX_BODY, num_joint=NUM_JOINT)
        frames = min(data.shape[1], MAX_FRAME)
        data_out[i, :, :frames, :, :] = data[:, :frames, :, :]

    sys.stdout.write('\n')

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    xsub_path = os.path.join(OUT_FOLDER, 'xsub')
    os.makedirs(xsub_path, exist_ok=True)

    print("=" * 60)
    print("  Preprocesando NTU RGB+D — split xsub")
    print(f"  Entrada: {DATA_PATH}")
    print(f"  Salida:  {xsub_path}")
    print("=" * 60)

    import time

    print("\n[1/2] Generando TRAIN (~40,000 muestras) ...")
    t0 = time.time()
    gendata(DATA_PATH, xsub_path, IGNORED_PATH, part='train')
    print(f"  Listo en {(time.time()-t0)/60:.1f} minutos")

    print("\n[2/2] Generando VAL (~16,000 muestras) ...")
    t0 = time.time()
    gendata(DATA_PATH, xsub_path, IGNORED_PATH, part='val')
    print(f"  Listo en {(time.time()-t0)/60:.1f} minutos")

    print("\n" + "=" * 60)
    print("  Preprocesamiento completado.")
    print(f"  Archivos en: {xsub_path}")
    print("    train_data.npy   (~5.4 GB)")
    print("    train_label.pkl")
    print("    val_data.npy     (~2.3 GB)")
    print("    val_label.pkl")
    print("=" * 60)
    print("\nSiguiente paso:")
    print('  cd "D:\\Dataset TAIA\\Proyecto\\TAIA-TAREAACADEMICA\\proyecto"')
    print("  py -3 train.py --model gcn --data_dir \"D:\\Dataset TAIA\\Proyecto\\datos\\ntu-rgb+d\" --batch 64 --epochs 80")
