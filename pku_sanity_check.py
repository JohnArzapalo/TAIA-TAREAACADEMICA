"""
pku_sanity_check.py — Fase 1
==============================
Verifica compatibilidad de PKU-MMD Phase 1 con el pipeline NTU RGB+D.

Chequeos:
  (a) 25 joints x 3 coords por frame (mismo que NTU)
  (b) Escala de coords en metros (rango Kinect tipico)
  (c) Visualizacion del esqueleto PKU vs topologia NTU

Uso:
    py -3 pku_sanity_check.py --skel_file "D:/Dataset TAIA/Proyecto/PKUMMD/Data/0002-L.txt"
    py -3 pku_sanity_check.py --skel_file "D:/Dataset TAIA/Proyecto/PKUMMD/Data/0002-L.txt" --ntu_sample "D:/Dataset TAIA/Proyecto/datos/ntu-rgb+d/.../S001C001P001R001A001.skeleton"
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Topologia NTU / Kinect v2 (identica en PKU-MMD)
NTU_EDGES = [(i-1, j-1) for (i, j) in [
    (1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),
    (9,21),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),
    (16,15),(17,1),(18,17),(19,18),(20,19),
    (22,23),(23,8),(24,25),(25,12)
]]

JOINT_NAMES = [
    'SpineBase','SpineMid','Neck','Head',
    'ShoulderLeft','ElbowLeft','WristLeft','HandLeft',
    'ShoulderRight','ElbowRight','WristRight','HandRight',
    'HipLeft','KneeLeft','AnkleLeft','FootLeft',
    'HipRight','KneeRight','AnkleRight','FootRight',
    'SpineShoulder','HandTipLeft','ThumbLeft','HandTipRight','ThumbRight'
]


def parse_pku_file(filepath):
    """
    Parsea un .txt de PKU-MMD. Formato esperado: una linea por frame,
    75 valores (1 persona) o 150 valores (2 personas).
    Devuelve (T, 25, 3) usando persona 1.
    """
    frames = []
    n_vals_seen = set()
    with open(filepath, 'r') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                vals = list(map(float, line.split()))
            except ValueError:
                if ln <= 3:
                    print(f"  AVISO linea {ln}: no numerica, saltando")
                continue
            n_vals_seen.add(len(vals))
            if len(vals) == 75:
                frames.append(np.array(vals).reshape(25, 3))
            elif len(vals) == 150:
                frames.append(np.array(vals[:75]).reshape(25, 3))
            elif len(vals) > 0 and ln <= 5:
                print(f"  AVISO linea {ln}: {len(vals)} valores (esperado 75 o 150)")

    return (np.stack(frames) if frames else None), n_vals_seen


def parse_ntu_first_frame(filepath):
    """Lee primer frame de un .skeleton NTU. Devuelve (25, 3)."""
    with open(filepath, 'r') as f:
        int(f.readline())
        for _ in range(1):
            num_bodies = int(f.readline())
            joints = []
            for b in range(num_bodies):
                f.readline()
                num_j = int(f.readline())
                body = []
                for _ in range(num_j):
                    v = list(map(float, f.readline().split()))
                    body.append(v[:3])
                if b == 0:
                    joints = body
    return np.array(joints) if joints else None


def check_scale(skel):
    """
    Kinect v2: X lateral [-1.5,1.5]m, Y vertical [-0.5,2.5]m, Z depth [0.5,5]m.
    El span Y del cuerpo entero debe ser ~1-2 m.
    """
    print("\n── Estadisticas de coordenadas ────────────────────────────")
    for i, ax in enumerate(['X', 'Y', 'Z']):
        v = skel[:, :, i].flatten()
        print(f"  {ax}: min={v.min():.3f}  max={v.max():.3f}  "
              f"mean={v.mean():.3f}  std={v.std():.3f}")
    y_span = skel[:, :, 1].max() - skel[:, :, 1].min()
    if 0.5 < y_span < 3.5:
        print(f"\n  OK  Span Y = {y_span:.2f} m → coordenadas en METROS (compatible NTU)")
        return True
    elif y_span > 100:
        print(f"\n  ERR Span Y = {y_span:.1f} → posiblemente en mm. Dividir por 1000.")
        return False
    else:
        print(f"\n  ?   Span Y = {y_span:.3f} — verificar manualmente")
        return None


def draw_skeleton(frame, title, save_path):
    """Dibuja esqueleto (vista frontal X-Y). frame: (25,3)"""
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_title(title, fontsize=11)
    for (i, j) in NTU_EDGES:
        ax.plot([frame[i,0], frame[j,0]], [frame[i,1], frame[j,1]],
                'b-', lw=1.5, alpha=0.6)
    ax.scatter(frame[:,0], frame[:,1], s=50, c='steelblue', zorder=5)
    for idx in [0, 3, 20, 4, 8, 12, 16]:
        ax.annotate(JOINT_NAMES[idx][:9], (frame[idx,0], frame[idx,1]),
                    fontsize=6, color='darkred', ha='left')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skel_file', required=True,
                   help='Archivo .txt de esqueleto PKU-MMD (ej: 0001-L.txt)')
    p.add_argument('--ntu_sample', default=None,
                   help='Opcional: .skeleton NTU para comparar visualmente')
    p.add_argument('--out_dir', default='resultados')
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    print("=" * 60)
    print("  SANITY CHECK — PKU-MMD Phase 1 vs NTU RGB+D")
    print("=" * 60)
    print(f"\nArchivo: {args.skel_file}")

    skel, n_vals = parse_pku_file(args.skel_file)
    if skel is None:
        print("ERR: No se parseo ningun frame. Verifica la ruta y formato.")
        sys.exit(1)

    T, V, C = skel.shape
    print(f"\n── Formato ─────────────────────────────────────────────────")
    print(f"  Valores/linea: {n_vals}")
    print(f"  Frames (T):    {T}")
    print(f"  Joints (V):    {V}  {'OK = 25' if V==25 else 'ERR != 25'}")
    print(f"  Coords (C):    {C}  {'OK = 3' if C==3 else 'ERR != 3'}")

    if V != 25 or C != 3:
        print("\nERR: Formato incompatible con NTU RGB+D.")
        sys.exit(1)

    scale_ok = check_scale(skel)

    print("\n── Visualizaciones ─────────────────────────────────────────")
    draw_skeleton(skel[0], f'PKU-MMD: {Path(args.skel_file).stem} (frame 0)',
                  out / 'sanity_pku.png')

    if args.ntu_sample:
        ntu = parse_ntu_first_frame(args.ntu_sample)
        if ntu is not None:
            draw_skeleton(ntu, f'NTU RGB+D: {Path(args.ntu_sample).stem} (frame 0)',
                          out / 'sanity_ntu.png')
            print("\n── Comparacion de rangos (frame 0) ─────────────────────────")
            print(f"  {'Eje':<3} {'PKU_min':>8} {'PKU_max':>8} {'NTU_min':>8} {'NTU_max':>8}")
            for i, ax in enumerate(['X','Y','Z']):
                print(f"  {ax:<3} {skel[0,:,i].min():>8.3f} {skel[0,:,i].max():>8.3f} "
                      f"{ntu[:,i].min():>8.3f} {ntu[:,i].max():>8.3f}")

    print("\n" + "=" * 60)
    if V == 25 and C == 3 and scale_ok is not False:
        print("  OK  SANITY CHECK PASADO — compatible con NTU RGB+D")
        print("  Siguiente: python preprocess_pku.py --skel_dir ... --label_dir ...")
    else:
        print("  ERR SANITY CHECK FALLIDO — revisar formato o escala de coords")
    print("=" * 60)


if __name__ == '__main__':
    main()
