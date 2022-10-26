from gpiozero import Servo
from time import sleep

x = int()
servo = Servo(x)  # where x is the pin number on the raspberry pie that the motor is connected to
category = str()

if category == 'Plastic':
    servo.max()
    sleep(4)
    servo.mid()

else:
    servo.min()
    sleep(4)
    servo.mid()

sleep(20)
