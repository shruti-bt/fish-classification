import os
import random
random.seed(99)

import cv2
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.losses import categorical_crossentropy

from utils import get_model, get_generators

def train(args):
    input_shape = (args.img_size, args.img_size, 3)
    train_gen, valid_gen = get_generators(
        args.data_path, 
        args.img_size, 
        args.batch_size, 
        args.test_img
    )

    model = get_model(input_shape, args.n_classes)
    model.compile(
        RMSprop(lr=args.learning_rate), 
        loss="categorical_crossentropy", 
        metrics=[metrics.AUC(name='auc'), 'accuracy']
    )

    es_callback = EarlyStopping(
        monitor='val_auc', 
        mode='max', 
        patience=5,
        verbose=1, 
        min_delta=0.005, 
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples//args.batch_size,
        epochs = args.num_epochs,
        validation_data=valid_gen,
        validation_steps=valid_gen.samples//args.batch_size,
        callbacks= [es_callback],
        verbose=1
    )

    model.save_weights(f"{args.weigths_path}/fish_classification")

    
def predict(img_path, args):
    input_shape = (args.img_size, args.img_size, 3)
    model = get_model(input_shape, args.n_classes)
    model.load_weights(f"{args.weigths_path}/fish_classification")
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape_image = cv2.resize(image_rgb, (args.img_size, args.img_size))
    preds = model(np.expand_dims(reshape_image / 255.0, 0))
    preds = tf.math.argmax(preds, -1).numpy()
    return img_path, preds


def test(args):
    imgs_list = []
    pred_list = []
    if os.path.isdir(args.test_img):
        for img_p in sorted(os.listdir(args.test_img)):
            img, pred = predict(os.path.join(args.test_img, img_p), args)
            imgs_list.append(img)
            pred_list.append(pred)
    else:
        img, pred = predict(args.test_img, args)
        imgs_list.append(img)
        pred_list.append(pred)

    return (imgs_list, pred_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fish Ckassification')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--test_img', required=True, type=str)
    parser.add_argument('--weigths_path', required=True, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--n_classes', default=9, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()

    class_dict = { 
        0:'Black Sea Sprat',
        1:'Gilt-Head Bream',
        2:'Hourse Mackerel',
        3:'Red Mullet',
        4:'Red Sea Bream',
        5:'Sea Bass',
        6:'Shrimp',
        7:'Striped Red Mullet',
        8:'Trout'
    }

    if args.train:
        train(args)
    elif args.test:
        imgs, preds = test(args)
        for im, ps in zip(imgs, preds):
            print(f"image path: {im} -> pred: {class_dict[ps[0]]}")
    else:
        raise("Please pass `train` or `test`")


        
