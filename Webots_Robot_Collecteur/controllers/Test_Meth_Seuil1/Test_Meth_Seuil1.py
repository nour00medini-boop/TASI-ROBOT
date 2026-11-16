import math
from pathlib import Path
import numpy as np
from controllers import Supervisor
from ultralytics import YOLO
from scene import generer_objets, Z_UP


def log(*a):
    print(*a, flush=True)


# ---------------- Réglages ----------------
DEVICE = 0                    # mettre "cpu" si Webots plante encore (plus lent)
CIBLES = {"canette", "bouteille", "paper_cup", "tissues"}
TOUS = CIBLES | {"autre_objet"}
CORPS, SEUIL = 0.30, 0.80
INVERSER_CAM = False
D_COLLECTE = 0.45
GRAINE, N_PAR_TYPE = 42, 6
HORS_ARENE = 60.0             # où partent les objets ramassés

# ---------------- Robot et capteurs ----------------
robot = Supervisor()
dt = int(robot.getBasicTimeStep())

gauche = robot.getDevice("left wheel")
droite = robot.getDevice("right wheel")
gauche.setPosition(float("inf"))
droite.setPosition(float("inf"))
lidar = robot.getDevice("lidar");   lidar.enable(dt)
camera = robot.getDevice("camera"); camera.enable(64)
W, H = camera.getWidth(), camera.getHeight()
FOV = camera.getFov()
log("DBG: capteurs ok")

model = YOLO(str(Path(__file__).resolve().parent / "best_B.pt"))
log("DBG: YOLO charge")

# ---------------- Création des objets (une seule fois) ----------------
robot.step(dt)
generer_objets(robot, graine=GRAINE, n_par_type=N_PAR_TYPE)
robot.step(dt)

OBJETS = []          # liste fixe : {"node", "nom", "tr", "pris"}
_enfants = robot.getRoot().getField("children")
for _i in range(_enfants.getCount()):
    _n = _enfants.getMFNode(_i)
    _f = _n.getField("model")
    if _f is not None and _f.getSFString() in TOUS:
        OBJETS.append({"node": _n, "nom": _f.getSFString(),
                       "tr": _n.getField("translation"), "pris": False})
log(f"DBG: {len(OBJETS)} objets dans la liste")

ME = robot.getSelf()

# ---------------- État du robot ----------------
batt, bac = 100.0, 0.0
etat, t_etat = "EXPLORER", 0.0
cible, vu_a, cible_id = None, -100.0, None
pas_avant = 0.0
sens = 0


def vitesse(g, d):
    gauche.setVelocity(g)
    droite.setVelocity(d)


def changer(e):
    global etat, t_etat
    etat, t_etat = e, robot.getTime()
    log("ETAT ->", e)


# ---------------- Lidar : évitement ----------------
def secteurs():
    r = lidar.getRangeImage()
    r = [x if (x == x and x != float("inf") and x > CORPS) else 99.0 for x in r]
    p = len(r) // 8
    return [min(r[k * p:(k + 1) * p]) for k in range(8)]


# ---------------- Capteur de distance simulé ----------------
def plan(p):
    return (p[0], p[1]) if Z_UP else (p[0], p[2])


def repere_robot():
    p = plan(ME.getPosition())
    R = ME.getOrientation()
    f = (R[0], R[3]) if Z_UP else (R[0], R[6])
    n = math.hypot(*f) or 1.0
    f = (f[0] / n, f[1] / n)
    g = (-f[1], f[0]) if Z_UP else (f[1], -f[0])
    return p, f, g


def viser(x_img):
    """(distance, index) de l'objet le plus proche dans la direction visée."""
    ang_cam = -math.atan((x_img - 0.5) * 2 * math.tan(FOV / 2))
    p, f, g = repere_robot()
    bd, bi = 99.0, None
    for i, o in enumerate(OBJETS):
        if o["pris"]:
            continue
        q = plan(o["node"].getPosition())
        dx, dy = q[0] - p[0], q[1] - p[1]
        a = dx * f[0] + dy * f[1]
        l = dx * g[0] + dy * g[1]
        if a <= 0:
            continue
        if abs(math.atan2(l, a) - ang_cam) < 0.15:
            d = math.hypot(dx, dy)
            if d < bd:
                bd, bi = d, i
    return bd, bi


def mesure(i):
    """Distance et angle (positif = objet à gauche) vers l'objet numéro i."""
    p, f, g = repere_robot()
    q = plan(OBJETS[i]["node"].getPosition())
    dx, dy = q[0] - p[0], q[1] - p[1]
    a = dx * f[0] + dy * f[1]
    l = dx * g[0] + dy * g[1]
    ang = math.atan2(l, a)
    return math.hypot(dx, dy), (-ang if INVERSER_CAM else ang)


# ---------------- Perception : YOLO ----------------
def detecter():
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((H, W, 4))[:, :, :3]
    res = model.predict(img, conf=0.25, imgsz=640, device=DEVICE, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        nom = model.names[int(b.cls)]
        if nom in CIBLES:
            xc = ((x1 + x2) / 2) / W
            d, i = viser(xc)
            out.append((nom, float(b.conf), xc, (y2 - y1) / H, d, i))
    return sorted(out, key=lambda o: o[4])


# ---------------- Décision : seuils ----------------
def seuil_decide(c, d, b, k):
    return c > 0.5 and d < 2.0 and b > 20 and k < 90


# ---------------- Boucle principale ----------------
vus, dernier, premiere = [], 0.0, True
log("DBG: boucle demarre")
while robot.step(dt) != -1:
    t = robot.getTime()
    batt = max(0.0, 100.0 - 0.1 * t)
    s = secteurs()
    devant = min(s[3], s[4])
    ca, cb = min(s[1], s[2]), min(s[5], s[6])

    if t - dernier > 0.15:
        vus, dernier = detecter(), t
        if premiere:
            log("DBG: 1ere detection ok")
            premiere = False
    if vus:
        cible, vu_a = vus[0], t
    visible = (cible is not None) and ((t - vu_a) < 0.4)

    if etat != "RETOUR_BASE" and (batt < 20 or bac >= 90):
        changer("RETOUR_BASE")

    # ----- EXPLORER -----
    if etat == "EXPLORER":
        if sens == 0 and devant < SEUIL:
            sens = 1 if ca > cb else -1
        elif sens != 0 and devant > SEUIL + 0.3:
            sens = 0
        if sens == 0:
            vitesse(3.0, 3.0)
        else:
            vitesse(-2.0 * sens, 2.0 * sens)
        if visible and t > pas_avant:
            changer("ALIGNER")

    # ----- ALIGNER + EVALUER -----
    elif etat == "ALIGNER":
        if not visible and t - t_etat > 1.5:
            changer("EXPLORER")
        elif visible:
            err = cible[2] - 0.5
            if INVERSER_CAM:
                err = -err
            if abs(err) < 0.06:
                nom, conf, dist, idx = cible[0], cible[1], cible[4], cible[5]
                ok = seuil_decide(conf, dist, batt, bac) and idx is not None
                log(f"EVALUER : {nom} conf={conf:.2f} dist={dist:.2f} -> {'COLLECTER' if ok else 'IGNORER'}")
                if ok:
                    cible_id = idx
                    changer("APPROCHER")
                elif conf > 0.5 and dist > 2.0:
                    log("   objet sur mais loin : on avance puis on reevalue")
                    pas_avant = t + 1.5
                    sens = 0
                    changer("EXPLORER")
                else:
                    changer("IGNORER")
            else:
                k = max(-2.0, min(2.0, -err * 8))
                vitesse(-k, k)
        else:
            vitesse(0, 0)

    # ----- APPROCHER -----
    elif etat == "APPROCHER":
        if cible_id is None or OBJETS[cible_id]["pris"]:
            cible_id, cible, vu_a, vus = None, None, -100.0, []
            changer("EXPLORER")
            continue
        d, ang = mesure(cible_id)
        if d < D_COLLECTE:
            changer("COLLECTER")
        elif t - t_etat > 25:
            log("   approche trop longue : abandon")
            cible_id, cible, vu_a, vus = None, None, -100.0, []
            changer("EXPLORER")
        elif abs(ang) > 0.5:
            k = max(-2.0, min(2.0, ang * 4))
            vitesse(-k, k)
        else:
            k = ang * 3
            vitesse(2.5 - k, 2.5 + k)

    # ----- COLLECTER (l'objet est déplacé hors de l'arène) -----
    elif etat == "COLLECTER":
        vitesse(0, 0)
        ok = False
        if cible_id is not None and not OBJETS[cible_id]["pris"]:
            d, _a = mesure(cible_id)
            if d < 0.9:
                o = OBJETS[cible_id]
                o["pris"] = True
                n_pris = sum(1 for x in OBJETS if x["pris"])
                o["tr"].setSFVec3f([HORS_ARENE + n_pris, HORS_ARENE, 0.5] if Z_UP
                                   else [HORS_ARENE + n_pris, 0.5, HORS_ARENE])
                try:
                    o["node"].resetPhysics()
                except Exception:
                    pass
                ok = True
        if ok:
            bac += 10
            log(f"COLLECTE ! bac={bac:.0f} %  batt={batt:.0f} %")
        else:
            log("echec de collecte")
        cible_id, cible, vu_a, vus = None, None, -100.0, []
        dernier = t
        pas_avant = t + 1.0
        changer("EXPLORER")

    # ----- IGNORER -----
    elif etat == "IGNORER":
        vitesse(-2.0, 2.0)
        if t - t_etat > 0.8:
            pas_avant = t + 2.0
            changer("EXPLORER")

    # ----- RETOUR_BASE (pas encore codé) -----
    elif etat == "RETOUR_BASE":
        vitesse(0, 0)