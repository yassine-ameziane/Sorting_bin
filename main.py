from raspz_camera import take_picture

# use function to see whether there is an object:
import RPi.GPIO as GPIO
import time
from ultrasonic import set_pins, get_distance

# make a prediction on the category using the neural network:
from neural_network import get_image, load_model, predict_category

# move the waste to the correct bin:
from motor_sorting import sort_waste

# initiate variables and set pin settings
set_pins(trigger_pin=16, echo_pin=18)
categories_dict = {0: "other", 1: "plastic"}
model = False

try:
    while True:
        distance_object = get_distance(trigger_out=trigger, echo_in=echo)
        trash_is_present = distance_object < 5  # choose threshold distance

        if trash_is_present:
            take_picture(picture_name="picture1", flip_camera=True, resolution=(640, 480), sleep_time=2)
            image = get_image(picture_name="picture1", Target_size=(224, 224))

            if not model:  # load the model if it hasn't be loaded yet
                model = load_model()

            trash_category = predict_category(model1=model, img_array=image)  # 0: other trash, 1: plastic
            sort_waste(pin_nr=, category=categories_dict[trash_category], error_bool=True)

            time.sleep(10)  # wait for a few seconds
        else:
            time.sleep(1)
finally:
    GPIO.cleanup()