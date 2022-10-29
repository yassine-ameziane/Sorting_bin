import tensorflow as tf
import numpy as np


def get_image(picture_name, Target_size: tuple, folder_name="trash_pictures", img_type="jpeg"):
    """Return a preprocessed image from a path with a target size."""
    image = tf.keras.preprocessing.image.load_img(path=f"{folder_name}/{picture_name}.{img_type}",
                                                  target_size=Target_size, color_mode='rgb')
    img_array = np.array([image])
    return img_array


def load_model(model_name, folder="model"):
    """Load the model and return it."""
    return tf.keras.models.load_model(f"{folder}/{model_name}")


def load_weights():
    """Load the weights, create the encoder, and return the model."""
    double_convLayers = [(8, (3,3), "relu", "same", (2,2)),
                         (16, (3,3), "relu", "same", (2,2))]
    dense_layers = [(64, "relu")]
    model_2 = create_encoder(double_convLayers, dense_layers, Input_shape = (224,224,3), Output_size = 2)
    model_2.load_weights("model/Neural_Network_Weights.h5")
    return model_2


def predict_category(model1, img_array: list):
    """Predict the category of the given image with the given model. Returns 0 (other) or 1 (plastic)."""
    prediction_array = model1.predict(img_array)
    category_predict = prediction_array.argmax(axis=1)[0]  # 0 is trash, 1 is plastic
    return category_predict


