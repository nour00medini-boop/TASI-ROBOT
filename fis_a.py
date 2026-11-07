"""FIS-A : controleur flou de decision de collecte (Mamdani, scikit-fuzzy).

Entrees  : conf [0,1], dist [0,5] m, batt [0,100] %, bac [0,100] %
Sortie   : score [0,100]  (score > SEUIL_COLLECTE => on collecte)

Methode (cours TASI, partie 1, chapitre 2) :
  fuzzification -> base de regles -> inference Max-Min -> defuzzification
  par centre de gravite (centroide).
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

SEUIL_COLLECTE = 50.0


def build_fis_a():
    """Construit le systeme de controle et renvoie (simulation, variables)."""
    # ---- Univers de discours (U) : plages physiques de chaque variable ----
    conf = ctrl.Antecedent(np.arange(0, 1.001, 0.01), "conf")
    dist = ctrl.Antecedent(np.arange(0, 5.001, 0.05), "dist")
    batt = ctrl.Antecedent(np.arange(0, 100.1, 1), "batt")
    bac = ctrl.Antecedent(np.arange(0, 100.1, 1), "bac")
    score = ctrl.Consequent(np.arange(0, 100.1, 1), "score")

    # ---- Fuzzification : fonctions d'appartenance (trapeze / triangle) ----
    conf["faible"] = fuzz.trapmf(conf.universe, [0, 0, 0.3, 0.5])
    conf["moyenne"] = fuzz.trimf(conf.universe, [0.3, 0.55, 0.8])
    conf["elevee"] = fuzz.trapmf(conf.universe, [0.6, 0.8, 1, 1])

    dist["proche"] = fuzz.trapmf(dist.universe, [0, 0, 0.8, 1.5])
    dist["moyenne"] = fuzz.trimf(dist.universe, [0.8, 2, 3.2])
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

    # ---- Base de regles (ET = min, NON = 1 - mu) ----
    rules = [
        ctrl.Rule(conf["elevee"] & dist["proche"], score["tres_haute"]),            # R1
        ctrl.Rule(conf["elevee"] & dist["moyenne"] & ~batt["faible"],
                  score["haute"]),                                                  # R2
        ctrl.Rule(conf["elevee"] & dist["loin"] & batt["haute"],
                  score["moyenne"]),                                                # R3
        ctrl.Rule(conf["elevee"] & dist["loin"] & ~batt["haute"],
                  score["basse"]),                                                  # R4
        ctrl.Rule(conf["moyenne"] & dist["proche"], score["haute"]),                # R5
        ctrl.Rule(conf["moyenne"] & dist["moyenne"], score["moyenne"]),             # R6
        ctrl.Rule(conf["moyenne"] & dist["loin"], score["basse"]),                  # R7
        ctrl.Rule(conf["faible"], score["tres_basse"]),                             # R8
        ctrl.Rule(batt["faible"] & ~dist["proche"], score["tres_basse"]),           # R9
        ctrl.Rule(bac["pleine"], score["tres_basse"]),                              # R10
    ]

    # Defuzzification : centroide (centre de gravite) = valeur par defaut
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim, dict(conf=conf, dist=dist, batt=batt, bac=bac, score=score)


_SIM, _VARS = build_fis_a()


def score_collecte(conf, dist, batt, bac):
    """Score de collecte (0-100) pour un objet detecte.

    Renvoie None si aucune regle n'est activee (trou dans la base de regles).
    """
    # on borne les entrees a leur univers (le capteur peut deborder un peu)
    _SIM.input["conf"] = float(np.clip(conf, 0, 1))
    _SIM.input["dist"] = float(np.clip(dist, 0, 5))
    _SIM.input["batt"] = float(np.clip(batt, 0, 100))
    _SIM.input["bac"] = float(np.clip(bac, 0, 100))
    try:
        _SIM.compute()
    except (ValueError, KeyError):
        return None
    return float(_SIM.output["score"])


def decider(conf, dist, batt, bac, seuil=SEUIL_COLLECTE):
    """Renvoie (collecter: bool, score: float|None)."""
    s = score_collecte(conf, dist, batt, bac)
    return (s is not None and s > seuil), s


if __name__ == "__main__":
    # Exemple du cours : confiance 0.55, a 1 m, batterie 80 %, bac a 20 %
    s = score_collecte(0.55, 1.0, 80, 20)
    print(f"score = {s:.1f} -> {'COLLECTER' if s > SEUIL_COLLECTE else 'IGNORER'}")