import os
from utils.utils import split_data, encode_single_sample, build_model, decode_single_prediction, decode_batch_predictions
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path 
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Path to the data directory
data_dir = Path("./samples/")
# Path to the model directory
model_dir = Path("./model/")

# Get list of all the images
images = list(map(str, list(data_dir.glob("*.jpg"))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
characters = [x for x in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 250
img_height = 80

# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(characters), num_oov_indices=0, mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = (
    train_dataset.map(
        lambda x, y: encode_single_sample(x, y, img_height, img_width, char_to_num), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        lambda x, y: encode_single_sample(x, y, img_height, img_width, char_to_num), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

try:
    prediction_model = keras.models.load_model(model_dir, compile=False)
except OSError:
    # Get the model
    model = build_model(img_width, img_height, characters)

    epochs = 100
    early_stopping_patience = 10

    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_dataset, 
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )

    prediction_model.save(model_dir)

#test
for image in images[30:40]:
    test = np.reshape(encode_single_sample(image, "unkown", img_height, img_width, char_to_num)["image"], (1, img_width, img_height, 1))
    pred = prediction_model.predict(test)
    pred_texts, acc = decode_single_prediction(pred, num_to_char)
    print(image + " prediction: " + pred_texts + " log_prob: " + str(-acc) + " acc: " + str(np.exp(-acc)))
