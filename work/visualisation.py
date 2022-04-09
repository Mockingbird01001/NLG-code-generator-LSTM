#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 03:11:25 2022

@author: mockingbird
"""

from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

import functions as f
from Text import *
from LSTM_class import *


class Visu:
    def __init__(self):
        self.max_len = 2 # longeur des sequence
        self.step = 1 # le pas
        self.layer_size = 64 # nombre de neuronnes
        # lecture des fichiers
        self.input_train = f.read_dir()
        # tokeniser le text
        self.text_train = Text(self.input_train)
        # creation des sequences a partir des tokens
        self.seq_train = Sequences(self.text_train, self.max_len, self.step)
        # load un ancien model
        self.model = self.lstm_model(sequence_length=self.max_len, vocab_size=len(self.text_train), layer_size=self.layer_size)
        self.model = models.load_model('data/out/lstm_model_simple')
        # init les equivalent du TF IDF
        self.token2ind, self.ind2token = self.text_train.token2ind, self.text_train.ind2token
        
        
    # def pour la creation de notre model
    def lstm_model(self, sequence_length, vocab_size, layer_size, embedding=False):
        model = models.Sequential()
        if embedding:
            model.add(layers.Embedding(vocab_size, layer_size))
            model.add(layers.Bidirectional(layers.LSTM(layer_size)))
            model.add(layers.Dropout(0.5))
        else:
            model.add(layers.LSTM(layer_size, input_shape=(sequence_length, vocab_size)))
            model.add(layers.Dropout(0.5))
        model.add(layers.Dense(vocab_size, activation='relu'))
        return model
        
    def prediction_lstm(self, input_prefix, nb_seq):
        result = []
        # tokenization de la sequence initiale
        text_prefix = Text(input_prefix, self.token2ind, self.ind2token)
        # prediction a partir d'une sequence
        pred = ModelPredict(self.model, text_prefix, self.token2ind, self.ind2token, self.max_len)
        for temperature in [1, 0.7, 0.4, 0.1]:
            result.append(pred.generate_sequence(nb_seq, temperature=temperature))
        return result