# hist_conf.py
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

IMG_DIRS = [r"C:\Users\nourmedini\Desktop\TASI-ROBOT\dataset\images\val", r"C:\Users\nourmedini\Desktop\TASI-ROBOT\Webots_Robot_Collecteur\dataset_webots\images\val"]   # à adapter
model = YOLO(r"runs_tasi/A_taco_seul/weights/best.pt")

def load_gt(img_path):
    lab = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
    if not lab.exists(): return []
    return [list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-9)

tp, fp = [], []
for d in IMG_DIRS:
    for p in Path(d).glob("*.*"):
        r = model.predict(str(p), conf=0.05, verbose=False, device=0)[0]
        h, w = r.orig_shape
        gts = [(int(c), [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
               for c, x, y, bw, bh in load_gt(p)]
        for b in r.boxes:
            cls, cf = int(b.cls), float(b.conf)
            ok = any(c == cls and iou(g, b.xyxy[0].tolist()) >= 0.5 for c, g in gts)
            (tp if ok else fp).append(cf)

bins = np.linspace(0, 1, 21)
plt.hist(tp, bins, alpha=.6, label=f"vrais positifs ({len(tp)})")
plt.hist(fp, bins, alpha=.6, label=f"faux positifs ({len(fp)})")
plt.xlabel("confiance YOLO"); plt.ylabel("nb détections"); plt.legend()
plt.savefig("hist_conf.png", dpi=150)
print("TP médiane:", np.median(tp), " FP médiane:", np.median(fp) if fp else None)