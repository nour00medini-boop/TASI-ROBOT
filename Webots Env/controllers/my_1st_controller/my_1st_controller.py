"""my_1st_controller controller."""

from controller import Robot

if __name__ == "__main__": 
    MyBot = Robot()      # create the Robot instance.
    timestep = 64   # get the time step of the current world.
    max_speed = 10
    # Motor Instances
    leftMot = MyBot.getDevice('Motor 2')
    rightMot = MyBot.getDevice('Motor 1')
    leftMot.setPosition(float('inf'))
    leftMot.setVelocity(0.0)
    rightMot.setPosition(float('inf'))
    rightMot.setVelocity(0.0)
    
    while MyBot.step(timestep) != -1:
        left_speed = 0.9 * max_speed
        right_speed = 0.9 * max_speed
        
        leftMot.setVelocity(left_speed)
        rightMot.setVelocity(right_speed)
        