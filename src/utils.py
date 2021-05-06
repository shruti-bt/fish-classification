import os
import shutil
import pandas as pd
from pathlib import Path
import cv2
import random 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_model(input_shape, n_classes):
    # Define the model
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model


def get_generators(
        path, 
        image_size, 
        batch_size, 
        test_path,
        num_val_samples=500, 
        num_test_samples=9
    ):

    image_size = (image_size, image_size)
    folder_path = Path(path)
    file_Path = list(folder_path.glob(r"**/*.png"))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_Path))

    file_Path = pd.Series(file_Path).astype(str)
    labels = pd.Series(labels)
    df = Main_Data = pd.concat([file_Path, labels],axis=1)
    df.columns = ['image', 'label']

    df = df[df["label"].apply(lambda x: x[-2:] != "GT")]
    df = df.reset_index(drop=True)
    rand = random.sample(range(len(df)), 500)
    validation_set = pd.DataFrame(df.iloc[rand, :].values, columns=['image', 'label'])
    df.drop(rand, inplace=True)

    rand = random.sample(range(len(validation_set)), 9)
    test_set = pd.DataFrame(validation_set.iloc[rand, :].values, columns=['image', 'label'])
    validation_set.drop(rand, inplace=True)
    
    train_data_generator = ImageDataGenerator(
        rescale=1./255, 
        shear_range=0.2, 
        zoom_range=0.2
    )
    data_generator = ImageDataGenerator(
        rescale=1./255
    )
    training_data_frame = train_data_generator.flow_from_dataframe(
        dataframe=df, 
        directory="", 
        x_col='image', 
        y_col='label', 
        target_size=image_size, 
        class_mode='categorical',
        batch_size=batch_size
    )
    validation_data_frame = data_generator.flow_from_dataframe(
        dataframe=validation_set, 
        directory="", 
        x_col='image', 
        y_col='label', 
        target_size=image_size, 
        class_mode='categorical',
        batch_size=batch_size
    )

    for test_img in test_set['image']:
        shutil.copy(test_img, test_path)

    # test_data_frame = data_generator.flow_from_dataframe(
    #     dataframe=test_set, 
    #     directory=DATADIR, 
    #     x_col='image', 
    #     y_col='label', 
    #     target_size=image_size, 
    #     class_mode='categorical', 
    #     shuffle=False,
    #     batch_size=batch_size
    # )

    return (
        training_data_frame,
        validation_data_frame,
        # test_data_frame
    )
