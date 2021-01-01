import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import os

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

if __name__ == '__main__':

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    folder_path = 'dataset/test'
    img_height = 250
    img_width = 250

    test_ds = []

    normalize_image = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255)
    ])

    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = normalize_image(img)
        test_ds.append(img)

    print(test_ds)

    exit(0)
