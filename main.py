from raspz_camera import take_picture

# use function to see whether there is an object:
# from ultrasonic import object_close

# make a prediction on the category using the neural network:
# from neural_network import predict_category

# move the waste to the correct bin:
# from motor_sorting import sort_waste


while True:
    trash_is_present = object_close()

    if trash_is_present:
        take_picture(picture_name="picture1", flip_camera=True, resolution=(640, 480), sleep_time=2)
        trash_category = predict_category()  # 0: other trash, 1: plastic
        sort_waste(trash_category)
        time.sleep(10) # wait for a few seconds
