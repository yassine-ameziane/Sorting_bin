from picamera import PiCamera
import time


def take_picture(picture_name: str, flip_camera: bool, resolution: tuple, sleep_time: float, img_type="jpeg",
                 folder_name="trash_pictures", window_pos_size=(1400, 0, 500, 500), conf_msg=True):
    """Take a picture with the camera and save it in the "trash_pictures" folder."""
    camera = PiCamera()
    camera.resolution = resolution
    camera.vflip = flip_camera

    camera.start_preview(fullscreen=False, window=window_pos_size)

    time.sleep(sleep_time)
    camera.capture(f"{folder_name}/{picture_name}.{img_type}")

    camera.stop_preview()

    if conf_msg:
        print("picture taken")


take_picture(picture_name="picture1", flip_camera=True, resolution=(640, 480), sleep_time=2)
