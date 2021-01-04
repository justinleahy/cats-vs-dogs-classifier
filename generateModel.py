import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pathlib
import csv

from tensorflow.keras import layers

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

if __name__ == '__main__':

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    data_dir = pathlib.Path('dataset/train')
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    cats = list(data_dir.glob('cat/*'))
    dogs = list(data_dir.glob('dog/*'))

    batch_size = 32
    img_height = 224
    img_width = 224

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='binary',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='binary',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    #    plt.figure(figsize=(10,10))
    #    for images, labels in train_ds.take(1):
    #        for i in range(9):
    #            ax = plt.subplot(3, 3, i + 1)
    #            plt.imshow(images[i].numpy().astype("uint8"))
    #            plt.title(class_names[labels[i]])
    #            plt.axis("off")
    #
    #    plt.show()

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.Rescaling(1. / 255,
                                                        input_shape=(img_height,
                                                                     img_width,
                                                                     3)),
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomZoom(0.2)
        ]
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
    val_ds = val_ds.map(lambda x, y: (data_augmentation(x), y))

    num_classes = 2

    # https://arxiv.org/abs/1409.1556
    vgg16_application = tf.keras.applications.VGG16(include_top=False,
                                                    input_shape=(img_height,
                                                                 img_width,
                                                                 3))
    for layer in vgg16_application.layers:
        layer.trainable = False

    dropout_layer = layers.Dropout(0.2)(vgg16_application.layers[-1].output)
    flat_layer = layers.Flatten()(dropout_layer)
    class_layer = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat_layer)
    output = layers.Dense(num_classes, activation='sigmoid')(class_layer)

    model = tf.keras.models.Model(inputs=vgg16_application.inputs, outputs=output)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save('models/sequential-model')

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
