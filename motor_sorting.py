from gpiozero import Servo
import time


def sort_waste(pin_nr: int, category: str, error_bool: bool):
    """ """
    servo = Servo(pin_nr)  # where pin_nr is the pin number on the raspberry pie that the motor is connected to

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
