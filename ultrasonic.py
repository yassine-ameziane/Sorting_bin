import RPi.GPIO as GPIO
import time


def set_pins(trigger_pin: int, echo_pin):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(trigger_pin, GPIO.OUT)
    GPIO.setup(echo_pin, GPIO.IN)


def get_distance(trigger_out, echo_in):
    """Returns distance of object from ultrasonic sensor in cm."""
    time1 = time.time()
    time2 = time.time()
    GPIO.output(trigger_out, GPIO.LOW)
    time.sleep(0.01)
    GPIO.output(trigger_out, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trigger_out, GPIO.LOW)

    while GPIO.input(echo_in) == 0:
        time1 = time.time()
    while GPIO.input(echo_in) == 1:
        time2 = time.time()

    time_difference = time2 - time1
    distance_cm = (time_difference * 34300) / 2  # in cm

    return distance_cm


def check_distance_loop(trigger, echo, period: int, pause=1):
    """Check the distance for a certain amount of time and print it."""
    set_pins(trigger_pin=trigger, echo_pin=echo)

    try:
        for i in range(period):
            distance = get_distance(trigger_out=trigger, echo_in=echo)
            print("The distance from object is ", distance, "cm")
            time.sleep(pause)
    finally:
        GPIO.cleanup()
        print("cleaned GPIO-setup")


# check_distance_loop(trigger=16, echo=18, period=30, pause=1)
