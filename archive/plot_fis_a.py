"""Tests de FIS-A : couverture de la base de regles + cas de bon sens."""
import itertools
import time
import numpy as np
from archive.fis_a import score_collecte, SEUIL_COLLECTE

# ---------------------------------------------------------------- 1) COUVERTURE
# On balaie une grille d'entrees. Si aucune regle ne s'active en un point,
# score_collecte renvoie None : c'est un trou dans la base de regles.
confs = np.linspace(0, 1, 11)
dists = np.linspace(0, 5, 11)
batts = np.linspace(0, 100, 6)
bacs = np.linspace(0, 100, 6)

t0 = time.time()
trous, n = [], 0
for c, d, b, z in itertools.product(confs, dists, batts, bacs):
    n += 1
    if score_collecte(c, d, b, z) is None:
        trous.append((round(c, 2), round(d, 2), round(b), round(z)))
print(f"Couverture : {n} points testes en {time.time() - t0:.0f} s, "
      f"{len(trous)} trou(s)")
for t in trous[:15]:
    print("  sans regle active (conf, dist, batt, bac) =", t)

# ---------------------------------------------------------------- 2) BON SENS
# (conf, dist, batt, bac, attendu_collecter, commentaire)
cas = [
    (0.90, 0.5, 80, 20, True, "sure, tout pres, batterie OK, bac vide"),
    (0.55, 1.0, 80, 20, True, "exemple du cours : moyenne, proche"),
    (0.90, 2.0, 80, 20, True, "sure, distance moyenne"),
    (0.90, 4.0, 80, 20, None, "sure mais loin (cas limite, batterie haute)"),
    (0.90, 4.0, 40, 20, False, "sure mais loin, batterie moyenne"),
    (0.20, 0.5, 80, 20, False, "confiance faible"),
    (0.90, 0.5, 80, 95, False, "bac plein"),
    (0.90, 3.0, 10, 20, False, "batterie faible et objet pas proche"),
    (0.45, 0.5, 80, 20, None, "confiance entre faible et moyenne (cas limite)"),
]
print("\nCas de bon sens (seuil = %.0f) :" % SEUIL_COLLECTE)
ok = True
for c, d, b, z, att, txt in cas:
    s = score_collecte(c, d, b, z)
    if s is None:
        print(f"  [ECHEC] {txt}: aucune regle active")
        ok = False
        continue
    dec = s > SEUIL_COLLECTE
    flag = "info " if att is None else ("OK   " if dec == att else "ECHEC")
    if flag == "ECHEC":
        ok = False
    print(f"  [{flag}] score={s:5.1f} -> {'collecter' if dec else 'ignorer '}"
          f"  | {txt}")

# ---------------------------------------------------------------- 3) MONOTONIE
# A confiance fixe, le score ne doit pas augmenter quand l'objet s'eloigne
# (de 'proche' a 'moyenne'), tout le reste etant constant.
print("\nScore en fonction de la distance (conf=0.9, batt=80, bac=20) :")
prev = None
for d in np.linspace(0.5, 4.5, 9):
    s = score_collecte(0.9, d, 80, 20)
    print(f"  dist={d:.1f} m -> score={s:5.1f}")
print("\nResultat global :", "TOUS LES CAS OK" if ok and not trous
      else "A CORRIGER (voir ci-dessus)")