from machine import Pin
import utime


def set_pins(pin_out: int, pin_in):
    trigger_out = Pin(pin_out, Pin.OUT)
    echo_in = Pin(pin_in, Pin.IN)
    return trigger_out, echo_in


def get_distance(trigger_out, echo_in):
    """Returns distance of object from ultrasonic sensor in cm."""
    trigger_out.low()
    utime.sleep_us(2)
    trigger_out.high()
    utime.sleep_us(5)
    trigger_out.low()
    while echo_in.value() == 0:
        signaloff = utime.ticks_us()
    while echo_in.value() == 1:
        signalon = utime.ticks_us()
    timepassed = signalon - signaloff
    distance = (timepassed * 0.0343) / 2
    print("The distance from object is ", distance, "cm")  # this is not needed for our product
    return distance

trigger_out, echo_in = set_pins(pin_out = 16, pin_in=18)
distance = get_distance(trigger_out, echo_in)
