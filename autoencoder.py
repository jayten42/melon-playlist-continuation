# -*- coding: utf-8 -*-
from collections import Counter

import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from arena_util import load_json
from arena_util import write_json


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu)

    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class AutoEncoder(tf.keras.Model):
    def __init__(self, intermediate_dim, original_dim):
        super(AutoEncoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed


def loss(preds, real):
    return tf.reduce_mean(tf.square(tf.subtract(preds, real)))


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        preds = model(original)
        reconstruction_error = loss(preds, original)
    gradients = tape.gradient(reconstruction_error, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)

    return reconstruction_error


def train_loop(model, opt, loss, dataset, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch_featrues in enumerate(dataset):
            loss_values = train(loss, model, opt, batch_featrues)
            epoch_loss += loss_values
        model.loss.append(epoch_loss)
        print('Epoch {}/{}. Loss: {}'.format(epoch + 1, epochs, epoch_loss.numpy()))


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

    def train_generator():
        for x in train_data:
            songs = np.zeros(707989)
            tags = np.zeros(30653)
            songs[x['songs']] = 1
            tags[x['tags']] = 1
            yield np.concatenate([songs, tags])

    def test_generator():
        for x in test_data:
            songs = np.zeros(707989)
            tags = np.zeros(30653)
            songs[x['songs']] = 1
            tags[x['tags']] = 1
            yield np.concatenate([songs, tags])

    training_dataset = tf.data.Dataset.from_generator(generator=train_generator, output_types=tf.float32,
                                                      output_shapes=tf.TensorShape([707989+30653])).batch(256)
    test_dataset = tf.data.Dataset.from_generator(generator=test_generator, output_types=tf.float32,
                                                  output_shapes=tf.TensorShape([707989+30653])).batch(256)

    model = AutoEncoder(intermediate_dim=128, original_dim=707989+30653)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    print("Train Loop...")

    train_loop(model, opt, loss, training_dataset, 20)
    print("Predict...")
    #
    preds = model(test_dataset)
    #
    pred_songs = preds[:, :707989]
    pred_tags = [id_to_tag[idx] for idx in preds[:, 707989:]]
    #
    print(pred_songs)
    print(pred_tags)
    model.save('saved_model')


if __name__ == "__main__":
    fire.Fire()
