"""
preprocess_pku.py — Fase 3
============================
Extrae clips de PKU-MMD Phase 1 (subset 1 persona) con clases comunes a
NTU RGB+D y los convierte al mismo formato: float32, shape (3, 300, 25, 2).

Mapeo: 33 clases de Actions1pers.csv tienen equivalente directo en NTU-60.
El modelo tiene cabeza de 60 clases (NTU); los labels se mapean a IDs NTU.

Estructura esperada del dataset PKU-MMD:
    pku_root/
        Data/          <-- .txt de esqueletos XXXX-{L,M,R}.txt
        Label1pers/    <-- .txt de etiquetas 1 persona (formato CSV)
        Split/cross-subject-1pers.txt

Formato de etiquetas Label1pers/ (comma-separated):
    class_id,start_frame,end_frame,confidence
    donde class_id es el ID de Actions1pers.csv (0=nothing, 1-43=clases)

Uso:
    py -3 preprocess_pku.py ^
        --skel_dir  "D:/Dataset TAIA/Proyecto/PKUMMD/Data" ^
        --label_dir "D:/Dataset TAIA/Proyecto/PKUMMD/Label1pers" ^
        --split     "D:/Dataset TAIA/Proyecto/PKUMMD/Split/cross-subject-1pers.txt" ^
        --out_dir   "D:/Dataset TAIA/Proyecto/PKUMMD/procesado"
"""
import argparse
import pickle
import numpy as np
from pathlib import Path


# ── Mapeo Actions1pers (0-indexed) → NTU (0-indexed) ────────────────────────
# 33 clases con equivalente semantico claro.
# Actions1pers IDs son exactamente los que aparecen en Label1pers/*.txt
PKU1PERS_TO_NTU = {
    1:  32,   # bow                      → nod head/bow          (NTU 33)
    2:   3,   # brushing hair            → brush hair            (NTU  4)
    3:   2,   # brushing teeth           → brush teeth           (NTU  3)
    4:  30,   # check time (from watch)  → check time from watch (NTU 31)
    5:  21,   # cheer up                 → cheer up              (NTU 22)
    6:   9,   # clapping                 → clapping              (NTU 10)
    7:  37,   # cross hands in front     → cross hands in front  (NTU 38)
    8:   0,   # drink water              → drink water           (NTU  1)
    9:   4,   # drop                     → drop                  (NTU  5)
   10:   1,   # eat meal/snack           → eat meal              (NTU  2)
   11:  40,   # falling                  → falling               (NTU 41)
   12:  22,   # hand waving              → hand waving           (NTU 23)
   # 13 hopping (one foot): sin equivalente en NTU-60
   14:  24,   # jump up                  → jump up               (NTU 25)
   15:  23,   # kicking something        → kicking something     (NTU 24)
   16:  25,   # make a phone call        → make a phone call     (NTU 26)
   17:   5,   # pickup                   → pick up               (NTU  6)
   18:  26,   # playing with phone/tablet→ playing with phone    (NTU 27)
   19:  28,   # pointing to something    → pointing to something (NTU 29)
   20:  19,   # put on a hat/cap         → put on a hat/cap      (NTU 20)
   # 21 put something inside pocket: sin equivalente
   22:  10,   # reading                  → reading               (NTU 11)
   23:  31,   # rub two hands            → rub two hands         (NTU 32)
   24:  35,   # salute                   → salute                (NTU 36)
   25:   7,   # sitting down             → sit down              (NTU  8)
   26:   8,   # standing up              → stand up              (NTU  9)
   27:  20,   # take off a hat/cap       → take off a hat/cap    (NTU 21)
   28:  18,   # take off glasses         → take off glasses      (NTU 19)
   29:  14,   # take off jacket          → take off jacket       (NTU 15)
   # 30 take out from pocket: sin equivalente
   31:  29,   # taking a selfie          → taking a selfie       (NTU 30)
   32:  12,   # tear up paper            → tear up paper         (NTU 13)
   33:   6,   # throw                    → throw                 (NTU  7)
   # 34 touch back (backache): NTU 43 back pain — match debatible
   # 35 touch chest: sin equivalente claro
   36:  41,   # touch head (headache)    → touch head headache   (NTU 42)
   # 37 touch neck: NTU 44 neck pain — match debatible
   38:  27,   # typing on keyboard       → typing on keyboard    (NTU 28)
   39:  45,   # use a fan                → use a fan             (NTU 46)
   40:  13,   # wear jacket              → put on jacket         (NTU 14)
   41:  17,   # wear on glasses          → put on glasses        (NTU 18)
   42:  34,   # wipe face                → wipe face             (NTU 35)
   43:  11,   # writing                  → writing               (NTU 12)
}

# Nombres de Actions1pers (indices 0-43) para log
PKU1PERS_NAMES = [
    'nothing','bow','brushing_hair','brushing_teeth','check_time',
    'cheer_up','clapping','cross_hands','drink_water','drop',
    'eat_meal','falling','hand_waving','hopping','jump_up',
    'kicking','make_phone_call','pickup','playing_phone','pointing',
    'put_on_hat','put_in_pocket','reading','rub_hands','salute',
    'sitting_down','standing_up','take_off_hat','take_off_glasses','take_off_jacket',
    'take_out_pocket','taking_selfie','tear_paper','throw','touch_back',
    'touch_chest','touch_head','touch_neck','typing','use_fan',
    'wear_jacket','wear_glasses','wipe_face','writing'
]


def load_fileset(split_path, part='val'):
    """
    Lee Split/cross-subject-1pers.txt y devuelve el set de nombres de archivo
    del part indicado ('train' o 'val').
    Formato:
        Training
        0002-L.txt,0002-M.txt,...
        Validation
        0291-L.txt,...
    """
    lines = Path(split_path).read_text().strip().splitlines()
    key = 'Validation' if part == 'val' else 'Training'
    try:
        idx   = lines.index(key)
        files = {f.strip() for f in lines[idx + 1].split(',') if f.strip()}
    except (ValueError, IndexError):
        print(f"  AVISO: no se pudo parsear '{key}' en {split_path}. Usando todos los archivos.")
        files = None
    return files


def parse_pku_skeleton(filepath):
    """
    Lee archivo .txt PKU-MMD. Una linea por frame.
    75 valores  → 1 persona (25 joints × 3 coords)
    150 valores → 2 personas (usamos persona 1, primeros 75)
    Devuelve (T, 25, 3) o None si falla.
    """
    frames = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals = list(map(float, line.split()))
            except ValueError:
                continue
            if len(vals) == 75:
                frames.append(np.array(vals).reshape(25, 3))
            elif len(vals) == 150:
                frames.append(np.array(vals[:75]).reshape(25, 3))
    return np.stack(frames) if frames else None


def parse_pku_labels(filepath):
    """
    Lee archivo Label1pers/*.txt.
    Formato: class_id,start_frame,end_frame,confidence  (comma-separated)
    class_id es 0-indexed segun Actions1pers.csv (0=nothing, 1-43=acciones).
    Devuelve lista de (class_id, start, end).
    """
    instances = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                cls   = int(parts[0])   # ya 0-indexed per Actions1pers.csv
                start = int(parts[1])
                end   = int(parts[2])
                instances.append((cls, start, end))
            except ValueError:
                continue
    return instances


def clip_to_tensor(skel, start, end, window=300):
    """
    Extrae frames [start:end] del esqueleto y convierte al formato NTU:
    (C=3, T=300, V=25, M=2) — persona 1 en M=0, M=1 queda en cero.
    skel: (T_full, 25, 3)
    """
    clip = skel[start:end + 1]        # (T_clip, 25, 3)
    T_clip = clip.shape[0]

    out = np.zeros((3, window, 25, 2), dtype=np.float32)
    clip_t = clip.transpose(2, 0, 1)  # (3, T_clip, 25)

    if T_clip <= window:
        out[:, :T_clip, :, 0] = clip_t
    else:
        center = T_clip // 2
        half   = window // 2
        s = max(0, center - half)
        out[:, :, :, 0] = clip_t[:, s:s + window, :]

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skel_dir',  required=True,
                   help='Carpeta Data/ con .txt de esqueletos')
    p.add_argument('--label_dir', required=True,
                   help='Carpeta Label1pers/ con .txt de etiquetas')
    p.add_argument('--split',      default=None,
                   help='Split/cross-subject-1pers.txt (opcional)')
    p.add_argument('--split_part', default='val', choices=['train', 'val'],
                   help='Parte del split a procesar: train o val (default: val)')
    p.add_argument('--out_dir',   default='D:/Dataset TAIA/Proyecto/PKUMMD/procesado')
    p.add_argument('--window',    type=int, default=300)
    p.add_argument('--min_frames', type=int, default=10,
                   help='Minimo de frames para incluir un clip')
    args = p.parse_args()

    skel_dir  = Path(args.skel_dir)
    label_dir = Path(args.label_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split (opcional)
    val_files = None
    if args.split:
        val_files = load_fileset(args.split, args.split_part)
        if val_files:
            print(f"Split '{args.split_part}' cargado: {len(val_files)} archivos")

    skel_files = sorted(skel_dir.glob('*.txt'))
    print(f"\nArchivos de esqueleto en {skel_dir}: {len(skel_files)}")
    print(f"Clases comunes PKU-1pers <-> NTU: {len(PKU1PERS_TO_NTU)}")
    if val_files is not None:
        matching = [f for f in skel_files if f.name in val_files]
        print(f"Archivos en split '{args.split_part}': {len(matching)}")
        skel_files = matching

    data_list    = []
    labels_list  = []
    skipped_cls  = 0
    skipped_short = 0
    skipped_label = 0
    n_per_class  = {}

    for sf in skel_files:
        lf = label_dir / sf.name
        if not lf.exists():
            skipped_label += 1
            continue

        skel = parse_pku_skeleton(sf)
        if skel is None:
            print(f"  AVISO: no se parseo {sf.name}")
            continue

        instances = parse_pku_labels(lf)
        T_full = skel.shape[0]

        for (pku_cls, start, end) in instances:
            if pku_cls not in PKU1PERS_TO_NTU:
                skipped_cls += 1
                continue

            end_clip = min(end, T_full - 1)
            if end_clip - start < args.min_frames:
                skipped_short += 1
                continue

            ntu_cls = PKU1PERS_TO_NTU[pku_cls]
            tensor  = clip_to_tensor(skel, start, end_clip, args.window)
            data_list.append(tensor)
            labels_list.append(ntu_cls)
            n_per_class[pku_cls] = n_per_class.get(pku_cls, 0) + 1

    if not data_list:
        print("\nERR: Ningun clip extraido. Verifica rutas y formato de etiquetas.")
        return

    data_arr   = np.stack(data_list, axis=0)    # (N, 3, 300, 25, 2)
    labels_arr = np.array(labels_list, dtype=np.int64)

    out_data   = out_dir / 'pku_data.npy'
    out_labels = out_dir / 'pku_labels.pkl'
    np.save(str(out_data), data_arr)
    with open(out_labels, 'wb') as f:
        pickle.dump(labels_arr, f)

    print(f"\n── Resultado ───────────────────────────────────────────────")
    print(f"  Clips extraidos:          {len(data_list)}")
    print(f"  Clips sin mapeo de clase: {skipped_cls}")
    print(f"  Clips muy cortos:         {skipped_short}")
    print(f"  Archivos sin etiqueta:    {skipped_label}")
    print(f"  Shape data:               {data_arr.shape}")
    print(f"  Clases NTU unicas:        {len(set(labels_list))}")
    print(f"\n  Clips por clase PKU-1pers:")
    for pku_cls in sorted(n_per_class):
        ntu_cls = PKU1PERS_TO_NTU[pku_cls]
        name    = PKU1PERS_NAMES[pku_cls] if pku_cls < len(PKU1PERS_NAMES) else f'cls_{pku_cls}'
        print(f"    Actions1pers {pku_cls:2d} → NTU {ntu_cls+1:2d}  ({name}): {n_per_class[pku_cls]} clips")

    print(f"\n  Guardado en: {out_dir}")
    print(f"    {out_data.name}  ({data_arr.nbytes / 1e6:.1f} MB)")
    print(f"    {out_labels.name}")
    print(f"\n  Siguiente paso:")
    print(f"    py -3 eval_pku.py --pku_dir \"{out_dir}\" --ckpt_dir proyecto/resultados")


if __name__ == '__main__':
    main()
