from loss_function import *
from data_getter import Data_Getter
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import os, os.path
import errno
import datetime
import json
import keras


# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def train_model(model, trainX, trainY, epochs=3, batch_size=10, verbose=1, plot_name=None):
    model_hist = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)

    if plot_name is not None:
        plt.clf()
        plt.figure(1, figsize=[8, 8])
        plt.rcParams.update({'font.size': 16})
        plt.plot(model_hist.history['loss'], label='Training Loss')
        plt.tight_layout()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(plot_name + ".png", bbox_inches="tight")


config = {
    "n_": 256,
    "seq_len": 700,
    "look_back": 192,
    "horizon": 48,
    "hop": 5,
    "feature_dim": 12,
    "tau": 0.48,
    "epochs": 15,
    "point_loss": "Pinball",
    "lube_loss": "MSIS",
    "model": "lstm_model",
    "hidden_layers": 1,
    "hidden_units": 64,
    "optimizer": "keras.optimizers.RMSprop(lr=0.01, clipnorm= 1.0)",
    "activation": "tanh",
    "dropout": 0.1,
}

# loading data
hourly_df = pd.read_csv('Train/Hourly-train.csv', header=0, index_col=0)
train_df = pd.DataFrame()

for index, row in hourly_df.iterrows():
    train_df[index] = row.dropna()[-config["seq_len"]:].values
train_df = train_df.transpose()

# data preperation
data = np.array(train_df[train_df.columns[:]])

data_gen = Data_Getter(data)
trainX, trainY = data_gen.get_data(lookback=config["look_back"], horizon=config["horizon"],
                                   hop=config["hop"],
                                   feature_dim=config["feature_dim"], split=331)

# create path to save results and modell
timestampStr = datetime.datetime.now().strftime("%d_%b_%Y_(%H_%M_%S_%f)")
path = "results/new_models/small_lstm_model_" + timestampStr
path1 = "results/new_models/small_lstm_model_" + timestampStr + "/Holdout"
path2 = "results/new_models/small_lstm_model_" + timestampStr + "/Test"
mkdir_p(path1)
mkdir_p(path2)

# json config for recreation purposes
with open(path + '/config.json', 'w') as fp:
    json.dump(config, fp)

# train model

model = lstm_model(activation=config["activation"], feature_dim=config["feature_dim"],
                   loss=partial(pinball_loss, tau= config["tau"]),
                   hidden_layers=config["hidden_layers"],
                   units=config["hidden_units"], output_layer_units=48,
                   optimizer=keras.optimizers.RMSprop(lr=0.01, clipnorm=1.0), dropout=config["dropout"])

train_model(model, trainX, trainY, epochs=config["epochs"], batch_size=config["n_"],
            plot_name=path + "\point")

# evaluate model
predictedY = model.predict(data_gen.scaling(data, feature_dim=config["feature_dim"]))
predictedY = data_gen.inverse_scaling(predictedY)
predictedY = predictedY.reshape(-1, config["horizon"])

np.savetxt(path1 + "/point.csv", predictedY[331:], delimiter=",")
np.savetxt(path2 + "/point.csv", predictedY[:331], delimiter=",")

point_labels = predictedY
model = lstm_model(activation=config["activation"], feature_dim=config["feature_dim"],
                   loss=msis_no_normalization,
                   hidden_layers=config["hidden_layers"],
                   units=config["hidden_units"], output_layer_units=96,
                   optimizer=keras.optimizers.RMSprop(lr=0.01, clipnorm=1.0), dropout=config["dropout"])

train_model(model, trainX, trainY, epochs=config["epochs"], batch_size=config["n_"],
            plot_name=path + "\lube")

# evaluate model
predictedY = model.predict(data_gen.scaling(data, feature_dim=config["feature_dim"]))
print(predictedY.shape)
print(predictedY[0])
predictedY = data_gen.inverse_scaling(predictedY)
print(predictedY[0])
print(predictedY.shape)
predictedY = predictedY.reshape(-1, config["horizon"] * 2)
print(predictedY.shape)

np.savetxt(path1 + "/lower.csv", predictedY[331:, 1::2], delimiter=",")
np.savetxt(path2 + "/lower.csv", predictedY[:331, 1::2], delimiter=",")

np.savetxt(path1 + "/upper.csv", predictedY[331:, ::2], delimiter=",")
np.savetxt(path2 + "/upper.csv", predictedY[:331, ::2], delimiter=",")

lower_labels = predictedY[:, 1::2]
upper_labels = predictedY[:, ::2]

# plotting
results = np.array(pd.read_csv('Test/Hourly-test.csv', header=0, index_col=0))
valX = np.array(train_df[train_df.columns[:]])
trueY = results.reshape(414, 48)

X_input = np.linspace(0, 699, 700)
X_output = np.linspace(700, 747, 48)

plt.clf()
plt.figure(figsize=[10, 8])
plt.rcParams.update({'font.size': 16})
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.plot(X_input[-150:], valX[0, -150:], label='Input')
plt.plot(X_output, trueY[0], label='Actual Output', color="yellow")
plt.axvline(x=700, color='r')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(path + "/correct.png", bbox_inches='tight')

plt.clf()
plt.figure(figsize=[10, 8])
plt.rcParams.update({'font.size': 16})
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.plot(X_input[-150:], valX[0, -150:], label='Input')
plt.plot(X_output, point_labels[0], label='Point Prediction', color="green")
plt.axvline(x=700, color='r')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(path + "/point_prediction.png", bbox_inches='tight')

plt.clf()
plt.figure(figsize=[10, 8])
plt.rcParams.update({'font.size': 16})
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.plot(X_input[-150:], valX[0, -150:], label='Input')
plt.plot(X_output, point_labels[0], label='Point Prediction', color="green")
plt.plot(X_output, upper_labels[0], label='Upper Bound Prediction', color="red")
plt.plot(X_output, lower_labels[0], label='Lower Bound Prediction', color="orange")
plt.plot(X_output, trueY[0], label='Actual Output', color="yellow")
plt.axvline(x=700, color='r')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(path + "/result.png", bbox_inches='tight')
