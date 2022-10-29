import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import sklearn.model_selection
import numpy as np

import os
import itertools
from typing import Dict, Tuple, List


def get_file_paths(main_path: str) -> Dict[str, list]:
    """
    Input: main_path: directory name (str) which contains directories with the
                      category as their names and garbage pictures as content.
    Output: file_paths: dictionary with names of directories as
                        keys and the file paths as a list as values.
    """
    categories = os.listdir(main_path)
    file_paths = {category: [] for category in categories}

    for category in file_paths:
        sub_path = os.path.join(main_path, category)
        filenames = os.listdir(sub_path)
        file_paths_category = [os.path.join(sub_path, filename) for filename in filenames]
        file_paths[category] = file_paths_category

    return file_paths


def filesizes_graph_and_array(file_paths: Dict[str, list], display_graph=False) -> np.array:
    """Get the filesizes in the file_paths dictionary and return them.
    Optional: graph the distribution of the filesizes."""
    filesizes = []
    for category in file_paths:
        for path in file_paths[category]:
            filesize = os.path.getsize(path) / 1024  # file size in kilobyte (kb)
            filesizes.append(filesize)
    filesizes = np.array(filesizes)

    if display_graph:
        fig1, ax1 = plt.subplots(figsize=(3, 2))
        sns.histplot(filesizes, color=(0.12, 0.46, 0.80))
        ax1.set_title("Images in Kaggle dateset")
        ax1.set_xlabel("File size (kb)")
        ax1.set_ylabel("Amount")

    return filesizes


def get_image(Path: str, Target_size: Tuple[int, int]):
    """Return a prepoccesed image from a path with a target size"""
    return tf.keras.preprocessing.image.load_img(path=Path, target_size=Target_size, color_mode='rgb')


def get_images_and_categories(all_paths: dict, dict_map: dict, Target_Size: tuple) -> Tuple[np.array, np.array]:
    """Use all paths to create arrays of images and their respective categories."""
    imgs_arrays = []
    categories_array = []

    for category in all_paths:
        paths_for_category = all_paths[category]
        for path in paths_for_category:
            image = get_image(Path=path, Target_size=Target_Size)
            img_array = np.array(image)

            imgs_arrays.append(img_array)
            categories_array.append([dict_map[category]])

    return np.array(imgs_arrays), np.array(categories_array)


def create_train_test(X, Y, Test_size=0.2, Random_state=None):
    """Give X and Y arrays as input, return X_train, X_test, Y_train, Y_test"""
    return sklearn.model_selection.train_test_split(X, Y, test_size=Test_size, random_state=Random_state)


def create_encoder(Double_ConvLayers: List[tuple], Dense_Layers, Input_shape: Tuple[int], Output_size: int):
    """Initiate the model architecture and set up the layers of the neural network."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(scale=1. / 255, input_shape=Input_shape))

    for layer_param in Double_ConvLayers:
        Filters, Kernel_size, Activation, Padding, Pool_size = layer_param
        model.add(tf.keras.layers.Conv2D(filters=Filters,
                                         kernel_size=Kernel_size,
                                         activation=Activation,
                                         padding=Padding))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=Pool_size))

    model.add(tf.keras.layers.Flatten())

    for layer_param in Dense_Layers:
        Units, Activation = layer_param
        model.add(tf.keras.layers.Dense(units=Units,
                                        activation=Activation))

    model.add(tf.keras.layers.Dense(units=Output_size))

    return model


def add_optimizer(model, Learning_Rate) -> None:
    """Tell the model how to train, optimize and minimize error."""
    AdamOptim = tf.keras.optimizers.Adam(learning_rate=Learning_Rate)
    model.compile(optimizer=AdamOptim,
                  #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  #               loss= "categorical_crossentropy",
                  #               loss= "sparse_categorical_crossentropy",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  #               metrics=["accuracy"])
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


class Model():
    """"""

    def __init__(self, Double_ConvLayers: List[tuple], Dense_Layers: List[tuple], Input_shape: Tuple[int],
                 Output_size: int, Learning_Rate: float, Model_name: int):
        self.model = None
        self.double_convlayers = Double_ConvLayers
        self.dense_layers = Dense_Layers
        self.input_shape = Input_shape
        self.output_size = Output_size
        self.learning_rate = Learning_Rate
        self.model_name = Model_name
        self.training_accuracy = None
        self.validation_accuracy = None

    def train_model(self, Epochs: int):
        """Train the model"""
        if self.model is None:
            self.create_encoder()
            self.compile_encoder()

        self.model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test),
                       batch_size=32, epochs=Epochs, verbose=1)

        self.set_accuracies()

    def create_encoder(self):
        """Initiate the model architecture and set up the layers of the neural network."""
        # initiate model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Rescaling(scale=1. / 255, input_shape=self.input_shape))

        # variable 2d convolutional layers
        for layer_param in self.double_convlayers:
            Filters, Kernel_size, Activation, Padding, Pool_size = layer_param
            self.model.add(tf.keras.layers.Conv2D(filters=Filters,
                                                  kernel_size=Kernel_size,
                                                  activation=Activation,
                                                  padding=Padding))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=Pool_size))

        # flatten 2d convolutional layer
        self.model.add(tf.keras.layers.Flatten())

        # variable dense layers
        for layer_param in self.dense_layers:
            Units, Activation = layer_param
            self.model.add(tf.keras.layers.Dense(units=Units,
                                                 activation=Activation))

        # converge model to output format
        self.model.add(tf.keras.layers.Dense(units=self.output_size))

    def compile_encoder(self):
        # tell the model how to train, optimize and minimize error
        AdamOptim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=AdamOptim,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=["accuracy"])

    #                       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def set_accuracies(self):
        training_accuracy = self.model.history.history["accuracy"]
        validation_accuracy = self.model.history.history["val_accuracy"]

        if self.validation_accuracy is not None:
            training_accuracy = list(self.training_accuracy) + training_accuracy
            validation_accuracy = list(self.validation_accuracy) + validation_accuracy

        self.training_accuracy = np.array(training_accuracy)
        self.validation_accuracy = np.array(validation_accuracy)

    def plot_accuracy(self, ax=plt, label=None):
        ax.plot(self.validation_accuracy, label=label)

double_convLayers = [(16, (3,3), "relu", "same", (2,2)),
                     (32, (3,3), "relu", "same", (2,2))] # Filters, Kernel_size, Activation, Padding, Pool_size
dense_layers = [(128, "relu"), (64, "relu")] # Units, Activation

model1 = Model(double_convLayers, dense_layers, Input_shape = (224,224,3), Output_size = len(map_categories),
              Learning_Rate=0.001)
model1.train_model(Epochs=2)

def get_possible_conv_layers(filters_param, kernel_size_param, pool_size_param):
    conv_layers = []
    for filter1 in filters_param:
        for kernal_size1 in kernel_size_param:
            for pool_size1 in pool_size_param:
                conv_layers.append((filter1, kernal_size1, "relu", "same", pool_size1))
    return conv_layers

def get_all_possible_conv_layers(filters_param, kernel_size_param, pool_size_param, nr_conv_layers):
    possible_conv_layers = get_possible_conv_layers(filters_param, kernel_size_param, pool_size_param)
    combs = []
    for i in range(1, nr_conv_layers+1):
        comb = list(itertools.combinations_with_replacement(possible_conv_layers, i))
        combs.extend(comb)
    return combs

conv_layers = get_all_possible_conv_layers(filters_param, kernel_size_param, pool_size_param, nr_conv_layers=1)

# 1. get file paths
main_path = f"data_comb"
file_paths: Dict[str, List[str]] = get_file_paths(main_path)
########## !!!!!!Change folowwing by duplicating plastic!!!!!
# file_paths["plastic"] = file_paths["plastic"] + file_paths["plastic"]
# file_paths["other trash"] = np.random.choice(file_paths["other trash"], size=len(file_paths["plastic"]), replace=False).tolist()

map_categories: Dict[str, int] = {category:i for i, category in enumerate(file_paths)}
# redundant filesize check
filesizes = filesizes_graph_and_array(file_paths=file_paths, display_graph=False)

# 2. Create input and output
imgs_arrays, categories_array = get_images_and_categories(file_paths, map_categories, (224, 224))

# 3. Split into training and test set
X_train, X_test, Y_train, Y_test = create_train_test(imgs_arrays, categories_array, Test_size=0.2, Random_state=1)

# 4. Create the model and set up the layers
double_convLayers = [(8, (3,3), "relu", "same", (2,2)),
                     (16, (3,3), "relu", "same", (2,2))]
dense_layers = [(64, "relu")]
model = create_encoder(double_convLayers, dense_layers, Input_shape = (224,224,3), Output_size = len(map_categories))

# 5. Decide the loss function of the model
AdamOptim = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = AdamOptim,
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# 6. Train the model
history = model.fit(x = X_train, y = Y_train, validation_data = (X_test, Y_test), epochs = 10, verbose = 1)

