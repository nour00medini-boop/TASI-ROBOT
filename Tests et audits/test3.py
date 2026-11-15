import os

BASE = r"C:\Users\nourmedini\Desktop\TASI-ROBOT"

for root, dirs, files in os.walk(BASE):
    if "venv" in root.lower() or "runs" in root.lower():
        continue
    for f in files:
        if f.lower().endswith((".yaml", ".yml")):
            p = os.path.join(root, f)
            print("=" * 70)
            print(p)
            print(open(p, encoding="utf-8", errors="replace").read())