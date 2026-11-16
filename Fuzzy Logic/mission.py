from fuzzy_decision import fis_a_decide, seuil_decide

def decider(methode, conf, dist, batt, bac):
    if methode == "flou":
        return fis_a_decide(conf, dist, batt, bac)[0]
    return seuil_decide(conf, dist, batt, bac)

def etape(etat, objets, batt, bac, methode):
    """
    etat   : état actuel (texte)
    objets : liste de (confiance, distance) des objets vus
    retourne (nouvel_etat, cible)
    """
    # Priorité absolue : batterie faible ou bac plein
    if batt < 20 or bac > 90:
        return "RETOUR_BASE", None

    if etat == "EXPLORER":
        return ("EVALUER", None) if objets else ("EXPLORER", None)

    if etat == "EVALUER":
        ok = [(c, d) for c, d in objets if decider(methode, c, d, batt, bac)]
        if ok:
            cible = min(ok, key=lambda o: o[1])      # le plus proche
            return "APPROCHER", cible
        return "EXPLORER", None

    if etat == "APPROCHER":
        if not objets:
            return "EXPLORER", None
        cible = min(objets, key=lambda o: o[1])
        if cible[1] < 0.3:
            return "COLLECTER", cible
        return "APPROCHER", cible

    if etat == "COLLECTER":
        return "EXPLORER", None

    return "EXPLORER", None

# ----- Test avec de fausses données -----
if __name__ == "__main__":
    scenario = [
        ("EXPLORER",  [],             80, 10),
        ("EXPLORER",  [(0.55, 2.5)],  80, 10),
        ("EVALUER",   [(0.55, 2.5)],  80, 10),
        ("APPROCHER", [(0.55, 1.0)],  80, 10),
        ("APPROCHER", [(0.55, 0.2)],  80, 10),
        ("COLLECTER", [],             80, 15),
        ("EXPLORER",  [(0.9, 1.0)],   10, 15),   # batterie faible
    ]
    for methode in ("flou", "seuil"):
        print(f"\n--- méthode : {methode} ---")
        for etat, objs, b, k in scenario:
            nouveau, cible = etape(etat, objs, b, k, methode)
            print(f"{etat:10s} objets={objs} batt={b} → {nouveau}")