from pathlib import Path
import numpy as np
from controller import Robot
from ultralytics import YOLO

robot = Robot()
dt = int(robot.getBasicTimeStep())

gauche = robot.getDevice("left wheel")
droite = robot.getDevice("right wheel")
gauche.setPosition(float("inf"))
droite.setPosition(float("inf"))

lidar = robot.getDevice("lidar")
lidar.enable(dt)

camera = robot.getDevice("camera")
camera.enable(64)
W, H = camera.getWidth(), camera.getHeight()

model = YOLO(str(Path(__file__).resolve().parent / "best_B.pt"))
print("Classes du modele :", model.names)

CORPS, SEUIL, INVERSER = 0.30, 0.80, False

def vitesse(g, d):
    gauche.setVelocity(g)
    droite.setVelocity(d)

def secteurs():
    r = lidar.getRangeImage()
    r = [x if (x == x and x != float("inf") and x > CORPS) else 99.0 for x in r]
    pas = len(r) // 8
    return [min(r[k*pas:(k+1)*pas]) for k in range(8)]

def detecter():
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((H, W, 4))[:, :, :3]
    res = model.predict(img, conf=0.25, imgsz=640, device=0, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append((model.names[int(b.cls)], float(b.conf),
                    round(((x1 + x2) / 2) / W, 2), round((y2 - y1) / H, 2)))
    return out

sens = 0
dernier = 0
while robot.step(dt) != -1:
    s = secteurs()
    devant = min(s[3], s[4])
    cote_a, cote_b = min(s[1], s[2]), min(s[5], s[6])

    if sens == 0 and devant < SEUIL:
        sens = 1 if cote_a > cote_b else -1
        if INVERSER:
            sens = -sens
    elif sens != 0 and devant > SEUIL + 0.3:
        sens = 0

    if sens == 0:
        vitesse(3.0, 3.0)
    else:
        vitesse(-2.0 * sens, 2.0 * sens)

    if robot.getTime() - dernier > 0.5:
        dets = detecter()
        if dets:
            print("VU :", [(n, round(c, 2), f"x={x}", f"haut={h}") for n, c, x, h in dets])
        dernier = robot.getTime()