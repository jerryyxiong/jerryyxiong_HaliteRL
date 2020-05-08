import math
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from bot_fast import FastBot
from engine import Player, Game
from entity import MoveCommand, SpawnShipCommand, ConstructDropoffCommand
from viewer import Replayer


def create_unet():
    frames_input = tf.keras.Input(shape=(128, 128, 4), name='frame_input')
    d7 = layers.Conv2D(64, kernel_size=1, padding='same', kernel_initializer='he_normal')(frames_input)

    d6 = layers.Activation('relu')(d7)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(d6)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d6)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d6)

    d5 = layers.Activation('relu')(d6)
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d5)
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d5)

    d4 = layers.Activation('relu')(d5)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(d4)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d4)

    d3 = layers.Activation('relu')(d4)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d3)

    d2 = layers.Activation('relu')(d3)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d2)

    d1 = layers.Activation('relu')(d2)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d1)

    lf = layers.Activation('relu')(d1)
    lf = layers.BatchNormalization()(lf)
    lf = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        lf)
    lf = layers.BatchNormalization()(lf)
    lf = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(lf)
    lf = layers.Reshape((64,))(lf)  # latent frame

    meta_input = tf.keras.Input(shape=(4,), name='meta_input')  # map size, turns remaining, my bank, enemy bank
    meta = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(meta_input)
    meta = layers.BatchNormalization()(meta)
    meta = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(meta)
    meta = layers.BatchNormalization()(meta)

    # latent game state
    lg = layers.Concatenate()([lf, meta])
    lg = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(lg)
    lg = layers.BatchNormalization()(lg)

    spawn = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(lg)
    spawn = layers.BatchNormalization()(spawn)
    spawn = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(spawn)
    spawn = layers.BatchNormalization()(spawn)
    spawn = layers.Dense(1, activation='sigmoid', name='spawn_output')(spawn)  # build predictions

    u1 = layers.Reshape((1, 1, 128))(lg)
    u1 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u1)
    u1 = layers.add([d1, u1])
    u1 = layers.Activation('relu')(u1)
    u1 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u1)
    u1 = layers.BatchNormalization()(u1)

    u2 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u1)
    u2 = layers.add([u2, d2])
    u2 = layers.Activation('relu')(u2)
    u2 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u2)
    u2 = layers.BatchNormalization()(u2)

    u3 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u2)
    u3 = layers.add([u3, d3])
    u3 = layers.Activation('relu')(u3)
    u3 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u3)
    u3 = layers.BatchNormalization()(u3)

    u4 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u3)
    u4 = layers.add([u4, d4])
    u4 = layers.Activation('relu')(u4)
    u4 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u4)
    u4 = layers.BatchNormalization()(u4)

    u5 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u4)
    u5 = layers.add([u5, d5])
    u5 = layers.Activation('relu')(u5)
    u5 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u5)
    u5 = layers.BatchNormalization()(u5)

    u6 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u5)
    u6 = layers.add([u6, d6])
    u6 = layers.Activation('relu')(u6)
    u6 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u6)
    u6 = layers.BatchNormalization()(u6)

    u7 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u6)
    u7 = layers.add([u7, d7])
    u7 = layers.Activation('relu')(u7)

    moves = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u7)
    moves = layers.BatchNormalization()(moves)
    moves = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(moves)
    moves = layers.BatchNormalization()(moves)
    moves = layers.Conv2D(6, kernel_size=1, name='moves_output')(moves)  # move logits

    # model = tf.keras.Model(inputs=[frames_input, meta_input], outputs=[moves, spawn])
    model = tf.keras.Model(inputs=[frames_input, meta_input], outputs=moves)
    model.summary()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, file_names, batch_size):
        self.file_names = file_names
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.file_names) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.file_names[idx * self.batch_size: (idx + 1) * self.batch_size]
        loaded = [np.load(file) for file in batch_files]
        # {'moves_output': data['moves'], 'spawn_output': data['spawn']}
        frames = np.array([data['frames'] for data in loaded])
        frames[:, :, 0] = frames[:, :, 0] / 1000
        return [frames, np.array([data['meta'] for data in loaded])], np.array([data['moves'] for data in loaded])


if __name__ == '__main__':
    folder = 'data/'
    files = [folder + f for f in os.listdir(folder)]
    train_dataset = Dataset(files, 16)

    val_folder = 'val/'
    files = [val_folder + f for f in os.listdir(val_folder)]
    val_dataset = Dataset(files, 16)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/cp5000_{epoch:02d}.ckpt',
        verbose=1,
        save_weights_only=True
    )

    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=3
    )

    my_model = create_unet()
    my_model.load_weights(tf.train.latest_checkpoint('checkpoints/'))
    my_model.fit(train_dataset, epochs=10, callbacks=[cp_callback, es_callback], validation_data=val_dataset)

