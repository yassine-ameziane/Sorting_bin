# import pickle
import tensorflow as tf

from picamera import PiCamera
import time

# file = open("model/X_test, Y_test.pickle", "rb")
# X_test, Y_test = pickle.load(file)
# file.close()


# Load model 1
def load_model():
    return tf.keras.models.load_model("model/Neural_Network_Model.h5")


def load_weights():
    double_convLayers = [(8, (3,3), "relu", "same", (2,2)),
                         (16, (3,3), "relu", "same", (2,2))]
    dense_layers = [(64, "relu")]
    model_2 = create_encoder(double_convLayers, dense_layers, Input_shape = (224,224,3), Output_size = 2)
    model_2.load_weights("model/Neural_Network_Weights.h5")
    return model_2


model1 = load_model()

# take picture
camera = PiCamera()
camera.vflip()
camera.start_preview()
time.sleep(2)
camera.capture("picture_taken/picture 1.jpg")

# reformat picture
image = get_image(Path = "picture_taken/picture 1.jpg", Target_size = (224, 224))
img_array = np.array([image])

# predict category
predictio_array = model1.predict(img_array)
category_pred = predictio_array.argmax(axis=1)[0] # 0 is trash, 1 is plastic