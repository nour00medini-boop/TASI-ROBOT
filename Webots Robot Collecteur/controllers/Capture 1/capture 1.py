import os, math, random, csv
import numpy as np
import cv2
from controller import Supervisor

# ---------------- Configuration ----------------
N_IMAGES    = 500
SEED        = 42
SPLIT       = {"train": 0.70, "val": 0.20, "test": 0.10}
CLASSES     = ["canette", "bouteille", "paper_cup", "tissues", "autre_objet"]  # = names de data.yaml
CLASS_W     = [0.28, 0.28, 0.14, 0.14, 0.16]       # probabilité de tirage par classe
ARENA       = 1.8        # demi-côté utilisable (arène 4 x 4 m)
MIN_PX      = 8          # taille minimale d'une boîte (px)
MIN_VISIBLE = 0.6        # fraction minimale de la boîte visible dans l'image
PREVIEW_EVERY = 50

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT  = os.path.join(ROOT, "dataset_webots")
TEX_DIR = os.path.join(ROOT, "protos", "textures")   # fichiers : canette_*.jpg, bouteille_*.jpg ...

# ---------------- Initialisation ----------------
robot = Supervisor()
ts = int(robot.getBasicTimeStep())
cam = robot.getDevice("camera")
cam.enable(ts)
cam.recognitionEnable(ts)
W, H = cam.getWidth(), cam.getHeight()

me = robot.getSelf()
tr_f, rot_f = me.getField("translation"), me.getField("rotation")
z0 = tr_f.getSFVec3f()[2]
sun = robot.getFromDef("SUN")
children = robot.getRoot().getField("children")

random.seed(SEED)
for sub in ("images", "labels", "preview"):
    for s in (SPLIT if sub != "preview" else [""]):
        os.makedirs(os.path.join(OUT, sub, s), exist_ok=True)

# ---------------- Utilitaires ----------------
def tex_for(prefix):
    if not os.path.isdir(TEX_DIR):
        return ""
    files = [f for f in os.listdir(TEX_DIR)
             if f.lower().startswith(prefix) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return os.path.join(TEX_DIR, random.choice(files)).replace("\\", "/") if files else ""

def rnd_color(lo=0.1, hi=1.0):
    return " ".join(f"{random.uniform(lo, hi):.2f}" for _ in range(3))

def axis_angle(yaw, lying):
    """Rotation = lacet autour de Z, précédé (si couché) d'un basculement de 90 deg autour de X."""
    if not lying:
        return f"0 0 1 {yaw:.4f}"
    w1, z1 = math.cos(yaw / 2), math.sin(yaw / 2)       # qz
    w2, x2 = math.cos(math.pi / 4), math.sin(math.pi / 4)  # qx(90 deg)
    w, x, y, z = w1 * w2, w1 * x2, z1 * x2, z1 * w2
    ang = 2 * math.acos(max(-1, min(1, w)))
    s = math.sqrt(max(0.0, 1 - w * w))
    ax = (x / s, y / s, z / s) if s > 1e-6 else (0, 0, 1)
    return f"{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f} {ang:.4f}"

def node_string(cls, x, y):
    yaw = random.uniform(-math.pi, math.pi)
    lying = cls in ("canette", "bouteille") and random.random() < 0.25
    z = 0.034 if lying else 0.002
    head = f"translation {x:.3f} {y:.3f} {z} rotation {axis_angle(yaw, lying)} "
    if cls == "canette":
        t = tex_for("canette")
        body = (f'texture "{t}" color 1 1 1' if t else f"color {rnd_color(0.5, 1.0)}")
    elif cls == "bouteille":
        t = tex_for("bouteille")
        body = (f'texture "{t}" ' if t else f"labelColor {rnd_color()} ")
        body += (f"bodyColor {random.choice(['0.80 0.90 0.95', '0.75 0.92 0.80', '0.85 0.85 0.85'])} "
                 f"capColor {rnd_color()} transparency {random.uniform(0.5, 0.8):.2f}")
    elif cls == "paper_cup":
        t = tex_for("cup")
        body = (f'texture "{t}" color 1 1 1' if t else f"color {rnd_color(0.6, 1.0)}")
    elif cls == "tissues":
        t = tex_for("tissue")
        body = (f'texture "{t}" ' if t else "") + \
               f"radius {random.uniform(0.025, 0.045):.3f} seed {random.randint(1, 10**6)}"
    else:  # autre_objet
        shape = random.choice(["box", "cylinder", "sphere", "cone"])
        sx, sz = random.uniform(0.04, 0.09), random.uniform(0.05, 0.14)
        body = (f'shape "{shape}" size {sx:.3f} {sx:.3f} {sz:.3f} color {rnd_color()} '
                f"metalness {random.choice([0.0, 0.0, 0.8])} roughness {random.uniform(0.2, 0.8):.2f}")
    proto = {"canette": "Canette", "bouteille": "Bouteille", "paper_cup": "PaperCup",
             "tissues": "Tissues", "autre_objet": "AutreObjet"}[cls]
    return f"{proto} {{ {head} {body} }}"

def spawn(s):
    before = children.getCount()
    children.importMFNodeFromString(-1, s)
    if children.getCount() <= before:
        print("[!] import échoué :", s[:90])
        return None
    return children.getMFNode(children.getCount() - 1)

def settle(nodes, max_steps=80):
    for k in range(max_steps):
        robot.step(ts)
        if k < 8 or not nodes:
            continue
        vmax = max(max(abs(v) for v in n.getVelocity()) for n in nodes)
        if vmax < 0.01:
            break
    robot.step(ts)      # un pas de plus pour que la caméra rende l'état final

def grab():
    img = np.frombuffer(cam.getImage(), np.uint8).reshape(H, W, 4)[:, :, :3].copy()
    lines, boxes = [], []
    for o in cam.getRecognitionObjects():
        model = o.getModel()
        if model not in CLASSES:
            continue
        cx, cy = o.getPositionOnImage()
        sw, sh = o.getSizeOnImage()
        x0, y0, x1, y1 = cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2
        area = (x1 - x0) * (y1 - y0)
        a0, b0, a1, b1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        w, h = a1 - a0, b1 - b0
        if area <= 0 or w < MIN_PX or h < MIN_PX or (w * h) / area < MIN_VISIBLE:
            continue
        c = CLASSES.index(model)
        lines.append(f"{c} {(a0+a1)/2/W:.6f} {(b0+b1)/2/H:.6f} {w/W:.6f} {h/H:.6f}")
        boxes.append((int(a0), int(b0), int(a1), int(b1), model))
    return img, lines, boxes

# ---------------- Boucle de capture ----------------
splits = [s for s, r in SPLIT.items() for _ in range(round(N_IMAGES * r))]
random.shuffle(splits)
splits = (splits + ["train"] * N_IMAGES)[:N_IMAGES]

spawned, n_boxes = [], 0
with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as mf:
    wr = csv.writer(mf)
    wr.writerow(["image", "split", "n_objects", "n_labels", "sun_intensity", "robot_x", "robot_y", "yaw"])

    for i in range(N_IMAGES):
        for n in spawned:
            n.remove()
        spawned = []

        # 1) Pose du robot
        rx, ry = random.uniform(-ARENA + 0.3, ARENA - 0.3), random.uniform(-ARENA + 0.3, ARENA - 0.3)
        yaw = random.uniform(-math.pi, math.pi)
        tr_f.setSFVec3f([rx, ry, z0])
        rot_f.setSFRotation([0, 0, 1, yaw])
        me.resetPhysics()

        # 2) Éclairage
        inten = random.uniform(0.4, 2.5)
        if sun:
            sun.getField("intensity").setSFFloat(inten)
            sun.getField("direction").setSFVec3f([random.uniform(-1, 1), random.uniform(-1, 1), -random.uniform(0.4, 1.5)])

        # 3) Objets, placés surtout dans le champ de la caméra (HFOV 1 rad => +-0.5 rad)
        n_obj = random.choices(range(0, 7), weights=[5, 15, 20, 20, 15, 15, 10])[0]
        placed = []
        for _ in range(n_obj):
            for _try in range(30):
                d = random.uniform(0.6, 3.5)
                b = random.uniform(-0.45, 0.45) if random.random() < 0.85 else random.uniform(-math.pi, math.pi)
                x, y = rx + d * math.cos(yaw + b), ry + d * math.sin(yaw + b)
                if abs(x) < ARENA and abs(y) < ARENA and all((x-px)**2 + (y-py)**2 > 0.12**2 for px, py in placed):
                    placed.append((x, y))
                    cls = random.choices(CLASSES, weights=CLASS_W)[0]
                    nd = spawn(node_string(cls, x, y))
                    if nd is not None:
                        spawned.append(nd)
                    break

        # 4) Stabilisation puis capture
        settle(spawned)
        img, lines, boxes = grab()

        name = f"img_{i:05d}"
        sp = splits[i]
        cv2.imwrite(os.path.join(OUT, "images", sp, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(OUT, "labels", sp, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        wr.writerow([name, sp, len(spawned), len(lines), f"{inten:.2f}", f"{rx:.2f}", f"{ry:.2f}", f"{yaw:.2f}"])
        n_boxes += len(lines)

        if i % PREVIEW_EVERY == 0:
            for (a, b, c, d, m) in boxes:
                cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 1)
                cv2.putText(img, m, (a, max(b - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(OUT, "preview", name + ".jpg"), img)
        if i % 25 == 0:
            print(f"[{i}/{N_IMAGES}] {n_boxes} boîtes")

for n in spawned:
    if n.getId() != me.getId():
        n.remove()
spawned = []

with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write(f"path: {OUT.replace(chr(92), '/')}\n")
    f.write("train: images/train\nval: images/val\ntest: images/test\n")
    f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

print("Terminé :", OUT)
robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)