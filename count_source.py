import json
from pathlib import Path
from collections import Counter

data = json.load(open("TACO/data/annotations.json", encoding="utf-8"))
cats = {c["id"]: c["name"] for c in data["categories"]}
wanted = ["Drink can", "Clear plastic bottle", "Glass bottle", "Other plastic bottle",
          "Paper cup", "Tissues", "Food Can", "Aerosol",
          "Disposable plastic cup", "Foam cup", "Plastic bottle cap", "Cigarette"]

inst, imgs = Counter(), {n: set() for n in wanted}
for a in data["annotations"]:
    n = cats[a["category_id"]]
    if n in wanted:
        inst[n] += 1
        imgs[n].add(a["image_id"])

for n in wanted:
    print(f"{n:28s} instances={inst[n]:4d}  images={len(imgs[n]):4d}")