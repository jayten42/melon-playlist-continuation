# -*- coding: utf-8 -*-
from collections import Counter

import fire
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from arena_util import load_json
from arena_util import write_json
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data, batch_size=512, noisy=False, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.noisy = noisy
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        train, test = [], []
        for i in range(index, index + self.batch_size):
            x_songs = np.zeros(707989)
            x_tags = np.zeros(30653)
            y_songs = np.zeros(707989)
            y_tags = np.zeros(30653)

            songs = self.data[i]['songs']
            tags = self.data[i]['tags']
            y_songs[songs] = 1
            y_tags[tags] = 1
            if self.noisy:
                songs = np.random.choice(songs, int(len(songs)/2))
                tags = np.random.choice(tags, int(len(tags)/2))
            if len(songs) > 0:
                x_songs[songs] = 1
            if len(tags) > 0:
                x_tags[tags] = 1
            train.append(np.concatenate([x_songs, x_tags]))
            test.append(np.concatenate([y_songs, y_tags]))
        return np.array(train), np.array(test)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)


class Encoder(keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encode = keras.layers.Dense(hidden_dim, name="encode")

    def call(self, inputs):
        return self.encode(inputs)


class Decoder(keras.layers.Layer):
    def __init__(self, orig_dim):
        super().__init__()
        self.decode = keras.layers.Dense(orig_dim, name="decode", activation="sigmoid")

    def call(self, inputs):
        return self.decode(inputs)


class AutoEncoder(tf.keras.Model):
    def __init__(self, hidden_dim, orig_dim):
        super(AutoEncoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(orig_dim=orig_dim)

    def call(self, inputs):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)
        return decode

    def train_step(self, data):
        data = data[0]
        with tf.GradientTape() as tape:
            encode = self.encoder(data)
            decode = self.decoder(encode)
            loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, decode))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": loss
        }


def run(tag_to_id_fname, id_to_tag_fname, train_fname, test_fname):
    print("Loading tag_to_id...")
    tag_to_id = load_json(tag_to_id_fname)
    print("Loading id_to_tag...")
    id_to_tag = load_json(id_to_tag_fname)
    print("Loading train file...")
    train_data = load_json(train_fname)
    for ply in train_data:
        ply['tags'] = [tag_to_id[tag] for tag in ply['tags']]

    print("Loading test file...")
    test_data = load_json(test_fname)
    for ply in test_data:
        ply['tags'] = [tag_to_id[tag] for tag in ply['tags']]
    # print("Writing answers...")
    # answers = self._generate_answers(song_meta_json, train_data, questions)
    # write_json(answers, "results/results.json")
    print("Make Training dataset...")

    train_gen = DataGenerator(train_data, noisy=True)
    test_gen = DataGenerator(test_data, noisy=True)

    model = AutoEncoder(hidden_dim=128, orig_dim=707989+30653)
    model.compile(optimizer=keras.optimizers.Adam())
    print("Train Loop...")
    model.fit(train_gen, epochs=20, validation_data=test_gen)
    model.save('saved_model')
    print("Predict...")
    #
    preds = model(test_gen)
    #
    pred_songs = preds[:, :707989]
    pred_tags = [id_to_tag[idx] for idx in preds[:, 707989:]]
    #
    print(pred_songs)
    print(pred_tags)



if __name__ == "__main__":
    fire.Fire()
