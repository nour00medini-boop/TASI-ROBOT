import json, random, shutil
from pathlib import Path
from PIL import Image, ImageOps
from sympy import im

TACO = Path("TACO/data")
OUT = Path("dataset")
SEED = 42
CLASSES = {
    "canette": ["Drink can"],
    "bouteille": ["Clear plastic bottle", "Glass bottle", "Other plastic bottle"],
    "paper_cup": ["Paper cup"],
    "tissues": ["Tissues"],
    "autre_objet": ["Food Can", "Aerosol", "Disposable plastic cup",
                    "Foam cup", "Plastic bottle cap"],
}

data = json.load(open(TACO / "annotations.json", encoding="utf-8"))
cats = {c["id"]: c["name"] for c in data["categories"]}
print("Catégories TACO :", sorted(cats.values()))  # pour vérifier les noms

name2cls = {n.lower(): i for i, names in enumerate(CLASSES.values()) for n in names}
missing = set(name2cls) - {n.lower() for n in cats.values()}
if missing:
    raise SystemExit(f"Noms introuvables, corrige CLASSES : {missing}")
id2cls = {cid: name2cls[n.lower()] for cid, n in cats.items() if n.lower() in name2cls}

imgs = {im["id"]: im for im in data["images"]}
per_img = {}
for a in data["annotations"]:
    if a["category_id"] in id2cls:
        per_img.setdefault(a["image_id"], []).append(a)

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
n_cls = len(CLASSES)

# nombre d'instances de chaque classe, par image et au total
img_cnt, tot = {}, [0] * n_cls
for iid, anns in per_img.items():
    c = [0] * n_cls
    for a in anns:
        c[id2cls[a["category_id"]]] += 1
    img_cnt[iid] = c
    for k in range(n_cls):
        tot[k] += c[k]

# ordre : mélange fixe, puis images à classe rare d'abord (le tri est stable)
ids = sorted(per_img)
random.Random(SEED).shuffle(ids)
ids.sort(key=lambda i: min(tot[k] for k in range(n_cls) if img_cnt[i][k] > 0))

# affectation gloutonne vers l'ensemble le plus en retard sur sa cible
cur = {s: [0] * n_cls for s in RATIOS}
split = {s: [] for s in RATIOS}
for iid in ids:
    c = img_cnt[iid]
    def retard(s):
        return sum(c[k] * (RATIOS[s] * tot[k] - cur[s][k]) / max(tot[k], 1) for k in range(n_cls))
    best = max(RATIOS, key=retard)
    split[best].append(iid)
    for k in range(n_cls):
        cur[best][k] += c[k]

for s, lst in split.items():
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    for iid in lst:
        src = TACO / imgs[iid]["file_name"]
        if not src.exists():
            continue
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        w, h = im.size
        name = f"taco_{iid:05d}"
        im.save(OUT / "images" / s / f"{name}.jpg", quality=95)
        lines = []
        for a in per_img[iid]:
            x, y, bw, bh = a["bbox"]
            lines.append(f"{id2cls[a['category_id']]} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        (OUT / "labels" / s / f"{name}.txt").write_text("\n".join(lines))
    print(s, len(lst), "images")

(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: canette\n  1: bouteille\n"
)