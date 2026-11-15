import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ---------- FIS-A (Mamdani) ----------
def build_fis_a():
    conf = ctrl.Antecedent(np.arange(0, 1.001, 0.01), "conf")
    dist = ctrl.Antecedent(np.arange(0, 5.001, 0.05), "dist")
    batt = ctrl.Antecedent(np.arange(0, 100.1, 1), "batt")
    bac = ctrl.Antecedent(np.arange(0, 100.1, 1), "bac")
    score = ctrl.Consequent(np.arange(0, 100.1, 1), "score",
                            defuzzify_method="centroid")

    conf["faible"] = fuzz.trapmf(conf.universe, [0, 0, .3, .5])
    conf["moyenne"] = fuzz.trimf(conf.universe, [.3, .55, .8])
    conf["elevee"] = fuzz.trapmf(conf.universe, [.6, .8, 1, 1])

    dist["proche"] = fuzz.trapmf(dist.universe, [0, 0, .8, 1.5])
    dist["moyenne"] = fuzz.trimf(dist.universe, [.8, 2, 3.2])
    dist["loin"] = fuzz.trapmf(dist.universe, [2.5, 3.5, 5, 5])

    batt["faible"] = fuzz.trapmf(batt.universe, [0, 0, 15, 30])
    batt["moyenne"] = fuzz.trimf(batt.universe, [15, 40, 70])
    batt["haute"] = fuzz.trapmf(batt.universe, [50, 70, 100, 100])

    bac["vide"] = fuzz.trapmf(bac.universe, [0, 0, 30, 60])
    bac["pleine"] = fuzz.trapmf(bac.universe, [50, 80, 100, 100])

    score["tres_basse"] = fuzz.trimf(score.universe, [0, 0, 25])
    score["basse"] = fuzz.trimf(score.universe, [10, 30, 50])
    score["moyenne"] = fuzz.trimf(score.universe, [35, 55, 75])
    score["haute"] = fuzz.trimf(score.universe, [60, 78, 92])
    score["tres_haute"] = fuzz.trimf(score.universe, [80, 100, 100])

    rules = [
        ctrl.Rule(conf["elevee"] & dist["proche"], score["tres_haute"]),                        # R1
        ctrl.Rule(conf["elevee"] & dist["moyenne"] & ~batt["faible"], score["haute"]),          # R2
        ctrl.Rule(conf["elevee"] & dist["loin"] & batt["haute"], score["moyenne"]),             # R3
        ctrl.Rule(conf["elevee"] & dist["loin"] & ~batt["haute"], score["basse"]),              # R4
        ctrl.Rule(conf["moyenne"] & dist["proche"], score["haute"]),                            # R5
        ctrl.Rule(conf["moyenne"] & dist["moyenne"], score["moyenne"]),                         # R6
        ctrl.Rule(conf["moyenne"] & dist["loin"], score["basse"]),                              # R7
        ctrl.Rule(conf["faible"], score["tres_basse"]),                                         # R8
        ctrl.Rule(batt["faible"] & ~dist["proche"], score["tres_basse"]),                       # R9
        ctrl.Rule(bac["pleine"], score["tres_basse"]),                                          # R10
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

_SIM = build_fis_a()

def fis_a_score(c, d, b, k):
    """Score de collecte [0,100]. Lève ValueError si aucune règle n'est activée."""
    _SIM.input["conf"] = float(np.clip(c, 0, 1))
    _SIM.input["dist"] = float(np.clip(d, 0, 5))
    _SIM.input["batt"] = float(np.clip(b, 0, 100))
    _SIM.input["bac"] = float(np.clip(k, 0, 100))
    _SIM.compute()
    return float(_SIM.output["score"])

def fis_a_decide(c, d, b, k, seuil=50.0):
    s = fis_a_score(c, d, b, k)
    return s > seuil, s

# ---------- Contrôleur de référence (seuils) ----------
def seuil_decide(c, d, b, k, conf_min=0.5, dist_max=2.0, batt_min=20, bac_max=90):
    return (c > conf_min) and (d < dist_max) and (b > batt_min) and (k < bac_max)

# ---------- Tests ----------
def coverage_test():
    fails, scores = [], []
    for c in np.linspace(0, 1, 11):
        for d in np.linspace(0, 5, 11):
            for b in (0, 15, 40, 70, 100):
                for k in (0, 50, 80, 100):
                    try:
                        scores.append(fis_a_score(c, d, b, k))
                    except ValueError:
                        fails.append((c, d, b, k))
    print(f"Couverture : {len(fails)} échec(s) sur {len(fails) + len(scores)} points")
    if scores:
        print(f"Score min/max : {min(scores):.1f} / {max(scores):.1f}")
    for f in fails[:10]:
        print("  sans règle activée :", f)

def sanity_cases():
    cas = [
        ("évident, proche",        0.90, 0.5, 80, 10),
        ("évident, loin",          0.90, 4.0, 80, 10),
        ("ambigu 0.55, proche",    0.55, 1.0, 80, 10),
        ("ambigu 0.55, moyen",     0.55, 2.0, 80, 10),
        ("faible conf",            0.20, 0.5, 80, 10),
        ("bac plein",              0.90, 0.5, 80, 95),
        ("batterie faible, loin",  0.90, 3.0, 10, 10),
        ("batterie faible, proche",0.90, 0.5, 10, 10),
    ]
    print(f"\n{'cas':26s} {'flou':>6s} {'décision':>9s} {'seuils':>8s}")
    for nom, c, d, b, k in cas:
        dec, s = fis_a_decide(c, d, b, k)
        print(f"{nom:26s} {s:6.1f} {str(dec):>9s} {str(seuil_decide(c, d, b, k)):>8s}")

def control_surface(b=70, k=20):
    cs, ds = np.linspace(0, 1, 21), np.linspace(0, 5, 21)
    Z = np.array([[fis_a_score(c, d, b, k) for c in cs] for d in ds])
    plt.figure(figsize=(6, 4.5))
    plt.imshow(Z, origin="lower", aspect="auto", extent=[0, 1, 0, 5], vmin=0, vmax=100)
    plt.colorbar(label="score")
    plt.contour(cs, ds, Z, levels=[50], colors="white")
    plt.xlabel("conf"); plt.ylabel("dist (m)")
    plt.title(f"FIS-A : batt={b} %, bac={k} %")
    plt.tight_layout(); plt.savefig("surface_FIS_A.png", dpi=150)
    print("surface_FIS_A.png enregistrée")

if __name__ == "__main__":
    coverage_test()
    sanity_cases()
    control_surface()