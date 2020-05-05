# -*- coding: utf-8 -*-
from collections import Counter

import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from arena_util import load_json
from arena_util import write_json


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder ,self).__init__()
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
    return tf.reduce_mean(tf.square(tf.substract(preds, real)))


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        preds = model(original)
        reconstruction_error = loss(preds, original)
    gradients = tape.gradient(reconstruction_error, model.trainable_variable)
    gradient_variables = zip(gradients, model.trainable_variable)
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


def run(song_meta_fname, train_fname, question_fname):
    print("Loading song meta...")
    song_meta_json = load_json(song_meta_fname)

    print("Loading train file...")
    train_data = load_json(train_fname)
    x_train = [tf.concat([tf.one_hot(ply['songs'], 707989), tf.one_hot(ply['tags'], 30653)], 1) for ply in train_data]
    print("Loading question file...")
    questions = load_json(question_fname)
    x_test = [tf.concat([tf.one_hot(ply['songs'], 707989), tf.one_hot(ply['tags'], 30653)], 1) for ply in questions]

    # print("Writing answers...")
    # answers = self._generate_answers(song_meta_json, train_data, questions)
    # write_json(answers, "results/results.json")

    training_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(256)

    model = AutoEncoder(intermediate_dim=128 , original_dim=707989+30653)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)

    train_loop(model, opt, loss, training_dataset, 20)

    preds = model(x_test)

    pred_songs = preds[:, :707989]
    pred_tags = preds[:, 707989:]

    print(pred_songs)
    print(pred_tags)


if __name__ == "__main__":
    fire.Fire()