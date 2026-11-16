from controller import Robot

robot = Robot()
dt = int(robot.getBasicTimeStep())

gauche = robot.getDevice("left wheel")
droite = robot.getDevice("right wheel")
gauche.setPosition(float("inf"))
droite.setPosition(float("inf"))

lidar = robot.getDevice("lidar")
lidar.enable(dt)

CORPS = 0.30       # on ignore tout ce qui est plus proche (le robot lui-même)
SEUIL = 0.80       # distance d'évitement devant
INVERSER = False   # mettre True si le robot tourne VERS l'obstacle

def vitesse(g, d):
    gauche.setVelocity(g)
    droite.setVelocity(d)

def secteurs():
    r = lidar.getRangeImage()
    r = [x if (x == x and x != float("inf") and x > CORPS) else 99.0 for x in r]
    pas = len(r) // 8
    return [min(r[k*pas:(k+1)*pas]) for k in range(8)]

sens = 0
dernier = 0
while robot.step(dt) != -1:
    s = secteurs()
    devant = min(s[3], s[4])
    cote_a = min(s[1], s[2])
    cote_b = min(s[5], s[6])

    if robot.getTime() - dernier > 1.0:
        print(f"a={cote_a:.2f}  devant={devant:.2f}  b={cote_b:.2f}")
        dernier = robot.getTime()

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