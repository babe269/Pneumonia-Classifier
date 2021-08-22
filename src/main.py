#!/usr/bin/env python3

# Imports!
import os
import pathlib
import random

import cv2
import numpy as np
import shutil

import tensorflow as tf
from keras.layers import Activation, Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

import src.model

# Fucntions!


def test_new_input (data_dir:str,  img_height: str,img_width: str, class_names: str, model):
    """ This function is used to pass new input data to evaluate the accuracy of the model. It will print the predicted
        label and the confidence in that prediction.
        INPUTS: data_dir       -- dirctory to the data
                img_height     -- specified height of the image to be used
                img_width      -- specified width of the image to be used
                class_names    -- labels
                model          -- the model used"""
    xray_path = pathlib.Path(data_dir)

    img = keras.preprocessing.image.load_img(
        xray_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def preproccess_data(data_dir: str, height, width, batch_size ):

    """This function takes in parameters for each image aswell as the directory in which they are stored and uses kera's
    inbuilt preprocessing tools to create inputs for the model. Note that this function is configured to create a 80/20
    training-testing split.

    INPUTS: data_dir   -- dirctory to the data
            height     -- specified height of the image to be used
            width      -- specified width of the image to be used
            batch_size -- batch size for each batch of data created.

    OUTPUTS: train_ds  -- training dataset.
             val_ds    -- validation dataset"""

    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)

    return train_ds, val_ds


def restructure_data(data_dir: str):

    """This function takes the directory containing the image files and restructures it so that it contains each image in
    an appropriate sub-folder for its correct class. It creates a new directory called BACTERIAL and moves all bacterial
    pneumonia images there. Then, it renames the PNEUMONIA directory to VIRAL as it only contains Viral Pneumonia images

    INPUTS: directory where images are located (chest_xray)

     """

    data_dir = pathlib.Path(data_dir)
    dest_path = os.path.join(data_dir, "BACTERIAL")
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
        for filename in os.listdir(os.path.join(data_dir, "PNEUMONIA")):
             if filename.split('_')[1] == 'bacteria':
                src_path = os.path.join(data_dir, "PNEUMONIA", filename)
                shutil.move(src_path, dest_path)

        os.rename(os.path.join(data_dir, "PNEUMONIA"), os.path.join(data_dir, "VIRAL"))


def main():
    # Specify directory containing images (1 level higher than current working directory).
    data_dir = "../chest_xray"
    # Restructure data to suit task (See function above).
    restructure_data(data_dir)
    data_dir = pathlib.Path(data_dir)

    # Print total image count for testing.
    image_count = len(list(data_dir.glob('*/*.jpeg')))
    print(image_count)

    # Specify data parameters for preprocessing.
    batch_size = 32
    img_height = 180
    img_width = 180

    # Preprocess data (see function above).
    train_ds, val_ds = preproccess_data(data_dir,img_height, img_width, batch_size)

    # Print to see if three classes were identified correctly.
    class_names = train_ds.class_names
    print(class_names)

    # Data prefetching using tensorflow to improve runtime performance.
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



    # Call the model (see model.py file) with the data augmentation layers, width and height.
    m = src.model.model_zero(img_height, img_width )
    # Summary of the model.
    m.summary()
    print(m.get_layer(name=None, index=4))

    # Train the model. This has been set to 50 epochs as the data begins to saturate with the current batch size
    epochs = 100
    history = m.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Plot data to evaluate performance.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # This is used to test new inputs. Simply replace the below with the path to any new inputs and run the code.
    test_path = pathlib.Path("C:/users/binun/Downloads/Normal_1.jpg")
    test_new_input(test_path, img_height, img_width, class_names,m)

    pass


if __name__ == "__main__":

    main()
