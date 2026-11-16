"""Controleur de test Webots.

Verifie 3 choses :
  1) que Webots utilise bien le Python du venv (ultralytics + torch importables) ;
  2) la liste des peripheriques du robot (noms exacts des moteurs, capteurs) ;
  3) ce que la camera reconnait (noeud Recognition) et la distance lidar minimale.
"""
import sys
from controllers import Robot

robot = Robot()
pas = int(robot.getBasicTimeStep())

# ---- 1) Python et bibliotheques -------------------------------------------
print("Python :", sys.executable)
try:
    import torch
    import ultralytics  # noqa: F401
    print("ultralytics OK | torch", torch.__version__,
          "| cuda", torch.cuda.is_available())
except Exception as e:
    print("ECHEC import ultralytics/torch :", e)

# ---- 2) Peripheriques ------------------------------------------------------
print("Peripheriques du robot :")
for i in range(robot.getNumberOfDevices()):
    print("  -", robot.getDeviceByIndex(i).getName())

camera = robot.getDevice("camera")
lidar = robot.getDevice("lidar")
if camera is not None:
    camera.enable(pas)
    if camera.hasRecognition():
        camera.recognitionEnable(pas)
    else:
        print("ATTENTION : la camera n'a pas de noeud Recognition")
else:
    print("ATTENTION : aucun peripherique nomme 'camera' (champ name du noeud)")
if lidar is not None:
    lidar.enable(pas)
else:
    print("ATTENTION : aucun peripherique nomme 'lidar' (champ name du noeud)")


def champ(obj, nom):
    """Lit un champ d'un objet reconnu (le nom exact depend de la version)."""
    for n in ("get_" + nom, "get" + nom.title().replace("_", "")):
        f = getattr(obj, n, None)
        if callable(f):
            return f()
    return getattr(obj, nom, "?")


# ---- 3) Boucle de simulation ----------------------------------------------
t = 0
afficher_attributs = True
while robot.step(pas) != -1:
    t += 1
    if t % 20:          # on n'affiche qu'un pas sur 20 pour ne pas saturer
        continue
    if camera is not None and camera.hasRecognition():
        objs = camera.getRecognitionObjects()
        print(f"[{t}] objets reconnus : {len(objs)}")
        for o in objs:
            print("   ", champ(o, "model"),
                  "| centre image", champ(o, "position_on_image"),
                  "| taille image", champ(o, "size_on_image"),
                  "| position relative", champ(o, "position"))
        if objs and afficher_attributs:
            print("   attributs disponibles :",
                  [a for a in dir(objs[0]) if not a.startswith("_")])
            afficher_attributs = False
    if lidar is not None:
        dist = [x for x in lidar.getRangeImage() if x != float("inf")]
        print("   lidar : distance min =", round(min(dist), 2) if dist else "rien")