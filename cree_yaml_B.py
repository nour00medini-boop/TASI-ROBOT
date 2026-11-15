from pathlib import Path

# ⚠️ À adapter : tes vrais dossiers (ceux qui contiennent images/ et labels/)
TACO   = Path(r"C:\Users\nourmedini\Desktop\TASI-ROBOT\dataset")
WEBOTS = Path(r"C:\Users\nourmedini\Desktop\TASI-ROBOT\Webots Robot Collecteur\dataset_webots")

dirs = {
    "train": [TACO / "images" / "train", WEBOTS / "images" / "train"],
    "val":   [TACO / "images" / "val",   WEBOTS / "images" / "val"],
}

lines = []
for split, paths in dirs.items():
    lines.append(f"{split}:")
    for p in paths:
        assert p.exists(), f"Dossier introuvable : {p}"
        lab = p.parent.parent / "labels" / p.name
        assert lab.exists(), f"Labels introuvables : {lab}"
        lines.append(f"  - {p.as_posix()}")

lines += ["names:",
          "  0: bouteille", "  1: canette", "  2: paper_cup",
          "  3: tissues", "  4: autre_objet"]

Path("data_B.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(Path("data_B.yaml").read_text(encoding="utf-8"))