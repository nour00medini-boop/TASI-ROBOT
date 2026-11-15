import glob, collections
ROOT = r"C:\Users\nourmedini\Desktop\TASI-ROBOT\dataset"
names = ["canette", "bouteille", "paper_cup", "tissues", "autre_objet"]

for s in ("train", "val", "test"):
    imgs = glob.glob(f"{ROOT}/images/{s}/*.*")
    labs = glob.glob(f"{ROOT}/labels/{s}/*.txt")
    c, empty = collections.Counter(), 0
    for f in labs:
        L = [l for l in open(f) if l.strip()]
        empty += (len(L) == 0)
        for l in L:
            c[int(l.split()[0])] += 1
    print(f"{s}: {len(imgs)} images, {len(labs)} étiquettes, {empty} vides")
    print("   ", {names[k]: v for k, v in sorted(c.items())})