from cgitb import small
import os
import math
import numpy as np
import tensorflow as tf

from typing import Any, Tuple
from tensorflow import keras
import matplotlib.pyplot as plt
from model import RFDNNet
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import *

seed = 1
upscale_factor = 4
batch_size = 8

epochs = 10


def load_images(orig_size, small_size, orig_dir, small_dir):
    # Make two ImageDataGenerator instances
    datagen1 = ImageDataGenerator()
    datagen2 = ImageDataGenerator()

    # Data generator for inputs
    generator1 = datagen1.flow_from_directory(
        orig_dir,
        class_mode=None,
        seed=seed,
        target_size=(orig_size, orig_size),
        color_mode="rgb",
        batch_size=1,
        shuffle=False,
    )
    # Data generator for outputs
    generator2 = datagen2.flow_from_directory(
        small_dir,
        class_mode=None,
        seed=seed,
        target_size=(small_size, small_size),
        color_mode="rgb",
        batch_size=1,
        shuffle=False,
    )

    assert len(generator1.filenames) == len(
        generator2.filenames
    ), "ensure equal number of input and output images"

    # combine generators into one which yields input-output pair
    combined_generator = zip(generator1, generator2)
    print(generator1.filenames)
    print(generator2.filenames)

    dataset = tf.data.Dataset.from_generator(
        lambda: combined_generator,
        output_signature=(
            tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32),
        ),
    )

    dataset = dataset.batch(batch_size)

    def dimension_adjuster(
        *inputs: Tuple[tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        img1, img2 = inputs
        return tf.squeeze(img1, axis=1) / 255.0, tf.squeeze(img2, axis=1) / 255.0

    # Drop extra dimension
    return dataset.map(dimension_adjuster, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = load_images(orig_size = 1024, small_size = 256, orig_dir="/content/datasets/faces/original", small_dir="/content/datasets/faces/X4")
val_ds = load_images(orig_size = 1024, small_size = 256, orig_dir="/content/datasets/faces_val/original", small_dir="/content/datasets/faces_val/X4")

test_path = "/content/dataset/faces_val/X4"
test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".png")
    ]
)

#dataset = dataset.enumerate()
#for element in dataset:
#    img = element[1][0][0]
#    fig, ax = plt.subplots()
#    img = ax.imshow(img)
#    plt.show()
#    break

print("Data loaded, creating model")

rfanet_x = RFDNNet()
x = Input(shape=(None, None, 3))
out = rfanet_x.main_model(x, upscale_factor)
model = Model(inputs=x, outputs=out)
model.summary()

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

checkpoint_filepath = "weights"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath + '/content/RFDNet/best.h5',
    monitor="loss",
    mode="min",
    save_best_only=True,
    period=1
)

callbacks = [ESPCNCallback(test_img_paths), early_stopping_callback, model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

print("Compiling model...")
model.compile(
    optimizer=optimizer, loss=loss_fn,
)
print("Training model...")
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, #validation_data=val_ds, 
    verbose=2
)
print('done!')

