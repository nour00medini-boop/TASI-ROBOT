from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = ROOT / "runs" / "detect" / "runs_tasi" / "B_taco_webots" / "weights" / "best.pt"

JEUX = {
    "TACO (reel)":     ROOT / "dataset" / "images" / "val",
    "Webots (simul.)": ROOT / "Webots_Robot_Collecteur" / "dataset_webots" / "images" / "val",
}

def lire_labels(img):
    lab = img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")
    if not lab.exists():
        return []
    return [list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()]

def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aire = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (aire + 1e-9)

if __name__ == "__main__":
    assert WEIGHTS.exists(), WEIGHTS
    model = YOLO(str(WEIGHTS))
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (nom, dossier) in zip(axes, JEUX.items()):
        assert dossier.exists(), dossier
        tp, fp = [], []
        for img in sorted(dossier.glob("*.*")):
            r = model.predict(str(img), conf=0.05, imgsz=640, device=0, verbose=False)[0]
            h, w = r.orig_shape
            gts = [(int(c), [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
                   for c, x, y, bw, bh in lire_labels(img)]
            for b in r.boxes:
                cls, cf = int(b.cls), float(b.conf)
                ok = any(c == cls and iou(g, b.xyxy[0].tolist()) >= 0.5 for c, g in gts)
                (tp if ok else fp).append(cf)

        ax.hist(tp, bins, alpha=.6, label=f"vrais positifs ({len(tp)})")
        ax.hist(fp, bins, alpha=.6, label=f"faux positifs ({len(fp)})")
        ax.set_title(nom); ax.set_xlabel("confiance YOLO"); ax.set_ylabel("nb de detections")
        ax.legend()

        print(f"\n=== {nom} ===")
        for lo, hi in [(0.05, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01)]:
            n_tp = sum(lo <= c < hi for c in tp)
            n_fp = sum(lo <= c < hi for c in fp)
            tot = n_tp + n_fp
            part = f"{100*n_tp/tot:.0f} % justes" if tot else "aucune detection"
            print(f"conf {lo:.2f}-{min(hi,1):.2f} : {tot:4d} detections, {part}")

    plt.tight_layout()
    sortie = HERE / "hist_conf.png"
    plt.savefig(sortie, dpi=150)
    print("\nImage enregistree :", sortie)