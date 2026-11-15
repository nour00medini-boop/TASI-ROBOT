from pathlib import Path
from collections import Counter

names = ["canette", "bouteille", "paper_cup", "tissues", "autre_objet"]
for s in ["train", "val", "test"]:
    c = Counter()
    for f in Path(f"dataset/labels/{s}").glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    print(s, {names[i]: c[i] for i in range(5)})