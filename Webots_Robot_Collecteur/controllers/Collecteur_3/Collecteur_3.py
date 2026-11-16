from controller import Robot

robot = Robot()
dt = int(robot.getBasicTimeStep())

gauche = robot.getDevice("left wheel")
droite = robot.getDevice("right wheel")
gauche.setPosition(float("inf"))
droite.setPosition(float("inf"))

def vitesse(g, d):
    gauche.setVelocity(g)
    droite.setVelocity(d)

while robot.step(dt) != -1:
    t = robot.getTime()
    if t < 3:
        vitesse(3.0, 3.0)      # avance
    elif t < 5:
        vitesse(2.0, -2.0)     # tourne sur place
    else:
        vitesse(0.0, 0.0)      # s'arrête