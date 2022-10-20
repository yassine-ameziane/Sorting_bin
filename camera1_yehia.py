from picamera import PiCamera
import time

Camera = PiCamera()
Camera.resolution = (640,480)
Camera.vflip = True
Camera.start_preview()

time.sleep(2)

Camera.capture("file_name_of_picture.jpeg")