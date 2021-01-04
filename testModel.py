import numpy as np
import tensorflow as tf
import pathlib
import os
import csv

from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    img_height = 224
    img_width = 224
    images = 12500

    model = tf.keras.models.load_model('models/sequential-model')
    model.summary()

    fields = ['id', 'label']
    file_name = 'submissions.csv'

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

    for i in range(images):
        data_dir = pathlib.Path('dataset/test/animal-' + str(i + 1) + '.jpg')
        test_ds = tf.keras.preprocessing.image.load_img(data_dir, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(test_ds)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        class_names = ['cat', 'dog']

        label = np.argmax(score)

        csv_data = [i+1, label]

        with open(file_name, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_data)

        print(
            "{} : This image most likely belongs to {} with a {:.2f} percent confidence.".format(i + 1, class_names[label], 100 * np.max(score))
        )

    exit(0)
