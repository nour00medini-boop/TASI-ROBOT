import os
import numpy as np
import cv2
from controller import Robot

robot = Robot()
ts = int(robot.getBasicTimeStep())
cam = robot.getDevice("camera")
cam.enable(ts)
cam.recognitionEnable(ts)

for _ in range(15):          # let the renderer stabilize
    robot.step(ts)

w, h = cam.getWidth(), cam.getHeight()
img = np.frombuffer(cam.getImage(), np.uint8).reshape(h, w, 4)[:, :, :3].copy()

objs = cam.getRecognitionObjects()
print(f"{len(objs)} object(s) recognized")
for o in objs:
    cx, cy = o.getPositionOnImage()
    sw, sh = o.getSizeOnImage()
    x0, y0 = int(cx - sw / 2), int(cy - sh / 2)
    x1, y1 = int(cx + sw / 2), int(cy + sh / 2)
    print(f"{o.getModel():12s} center=({cx},{cy}) size=({sw},{sh})")
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    cv2.putText(img, o.getModel(), (x0, max(y0 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.png")
cv2.imwrite(out, img)
print("saved:", out)