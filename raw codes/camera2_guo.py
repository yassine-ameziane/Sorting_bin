from picamera import PiCamera
from time import sleep
from fractions import Fraction

# Force sensor mode 3 (the long exposure mode)
# Set the framerate to 1/6fps
# Set the shutter speed to 6s
# Set ISO to 800 (for maximum gain)

camera = PiCamera(
    resolution = (1280, 720),
    framerate = Fraction(1, 6),
    sensor_mode = 3)
camera.shutter_speed = 6000000
camera.iso = 800
# Give the camera a good long time to set gains and
# measure AWB (you may wish to use fixed AWB instead)
sleep(30)
camera.exposure_mode = 'off'
# Finally, capture an image with a 6s exposure.
# Due to mode switching on the still part, this will take 
# Longer than 6 seconds
camera.capture('dark.jpg')