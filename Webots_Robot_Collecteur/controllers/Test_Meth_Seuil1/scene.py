import math
import random

TYPES = ["Canette", "Bouteille", "PaperCup", "Tissues", "AutreObjet"]
TAILLE = 10.0      # arène 10 x 10 m, centrée en (0, 0)
MARGE = 0.6        # distance minimale aux murs
Z_UP = True        # True : sol = plan x,y (Webots récent). False : sol = plan x,z
HAUTEUR = 0.05     # hauteur de dépose (les objets retombent au sol)


def generer_objets(robot, graine=42, n_par_type=6, dist_min=0.7,
                   zone_robot=1.5, interdit=()):
    """
    graine      : même graine = même disposition
    n_par_type  : nombre d'objets de chaque type (6 -> 30 objets)
    dist_min    : distance minimale entre deux objets (m)
    zone_robot  : rayon vide autour du robot au départ (m)
    interdit    : liste de zones à éviter, ex. [(2.0, 1.0, 0.8)] = (x, y, rayon)
    """
    rnd = random.Random(graine)
    p = robot.getSelf().getPosition()
    rx, ry = (p[0], p[1]) if Z_UP else (p[0], p[2])
    demi = TAILLE / 2 - MARGE
    enfants = robot.getRoot().getField("children")

    liste = [t for t in TYPES for _ in range(n_par_type)]
    rnd.shuffle(liste)

    points, poses = [], 0
    for nom in liste:
        for _ in range(300):
            x, y = rnd.uniform(-demi, demi), rnd.uniform(-demi, demi)
            if math.hypot(x - rx, y - ry) < zone_robot:
                continue
            if any(math.hypot(x - cx, y - cy) < r for cx, cy, r in interdit):
                continue
            if any(math.hypot(x - px, y - py) < dist_min for px, py in points):
                continue
            break
        else:
            print("Pas de place pour un objet", nom)
            continue

        points.append((x, y))
        yaw = rnd.uniform(0, 2 * math.pi)
        if Z_UP:
            pos, rot = f"{x:.3f} {y:.3f} {HAUTEUR}", f"0 0 1 {yaw:.3f}"
        else:
            pos, rot = f"{x:.3f} {HAUTEUR} {y:.3f}", f"0 1 0 {yaw:.3f}"
        enfants.importMFNodeFromString(-1, f"{nom} {{ translation {pos} rotation {rot} }}")
        poses += 1

    print(f"{poses} objets places (graine {graine})")