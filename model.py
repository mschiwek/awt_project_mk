import numpy as np
import keras
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras import backend as K
import tensorflow as tf


def lstm_model(activation='tanh', feature_dim=1, hidden_layers=1, loss=None,
               optimizer=keras.optimizers.RMSprop(lr=0.01, clipnorm=1.0),
               units=144, output_layer_units=48, dropout=0.5):
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=(None, feature_dim), return_sequences=hidden_layers > 1,
                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2), recurrent_dropout=dropout,
                   dropout=dropout))
    hidden_layers = hidden_layers - 1
    for i in range(hidden_layers):
        model.add(LSTM(units, activation=activation, return_sequences=hidden_layers > 1,
                       kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2),
                       recurrent_dropout=dropout,
                       dropout=dropout))
        hidden_layers = hidden_layers - 1
    model.add(Dense(output_layer_units, activation='linear',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3)))

    model.compile(loss=loss, optimizer=optimizer)

    return model


def conv_lstm_model(activation='tanh', feature_dim=1, hidden_layers=1, loss=None,
                    optimizer=keras.optimizers.RMSprop(lr=0.01, clipnorm=1.0),
                    units=144, output_layer_units=48):
    model = Sequential()

    model.add(Conv1D(filters=units, kernel_size=24, strides=1, activation="relu", input_shape=(None, feature_dim)))
    model.add(Conv1D(filters=units, kernel_size=24, strides=12, activation="relu"))
    model.add(LSTM(units, activation=activation, return_sequences=hidden_layers > 1,
                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))
    hidden_layers = hidden_layers - 1
    for i in range(hidden_layers):
        model.add(LSTM(units, activation=activation, return_sequences=hidden_layers > 1,
                       kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))
        hidden_layers = hidden_layers - 1
    model.add(Dense(output_layer_units, activation='linear',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3)))

    model.compile(loss=loss, optimizer=optimizer)

    return model
