from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time


def sort_waste(pin_nr: int, category: str, error_bool: bool):
    """Connect the servo motor and make it turn the right way based on the category ('plastic' or 'other')."""
    factory = PiGPIOFactory(host="169.254.115.105")
    servo = Servo(pin_nr=22, pin_factory=factory)

    if category == 'plastic':
        servo.max()
        time.sleep(4)
        servo.mid()

    elif category == 'other':
        servo.min()
        time.sleep(4)
        servo.mid()

    elif error_bool:
        raise ValueError("category has to be either 'plastic' or 'other'")


# sort_waste(pin_nr = 22, "plastic", error_bool=True)
