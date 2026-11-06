import cv2, random
from pathlib import Path

names = ["canette", "bouteille", "paper_cup", "tissues", "autre_objet"]
colors = [(0, 200, 0), (255, 100, 0), (0, 0, 255), (200, 0, 200), (0, 200, 200)]
out = Path("check")
out.mkdir(exist_ok=True)

imgs = sorted(Path("dataset/images/train").glob("*.jpg"))
random.Random(0).shuffle(imgs)

for p in imgs[:16]:
    im = cv2.imread(str(p))
    s = 900 / max(im.shape[:2])
    im = cv2.resize(im, None, fx=s, fy=s)   # on réduit d'abord, puis on dessine
    h, w = im.shape[:2]
    lab = Path("dataset/labels/train") / (p.stem + ".txt")
    for line in lab.read_text().splitlines():
        c, x, y, bw, bh = line.split()
        c = int(c)
        x, y, bw, bh = float(x), float(y), float(bw), float(bh)
        x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), colors[c], 2)
        cv2.putText(im, names[c], (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[c], 2)
    cv2.imwrite(str(out / p.name), im)
    print(p.name, "portrait" if h > w else "paysage")