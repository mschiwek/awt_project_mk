import numpy as np
from itertools import islice
from sklearn.preprocessing import StandardScaler


# https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
def window(seq, n=652):
    # Returns a sliding window (of width n) over data from the iterable"
    #   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class Data_Getter():

    def __init__(self, data):
        self.data = data
        self.scalers = [StandardScaler() for _ in data]
        self.scaled_data = []

        for i, value in enumerate(data):
            self.scaled_data.append(self.scalers[i].fit_transform(np.array(data[i]).reshape(-1, 1)))

    # create sliding window of input data
    def get_data(self, lookback=652, horizon=48, split=331, hop=1, feature_dim=1):
        length = lookback + horizon

        sliding_window = np.array([i for seq in self.scaled_data[:split] for i in window(seq, length)])

        trainX = np.array(sliding_window[::hop, :lookback])

        # reshape feature_dim if needed
        if feature_dim > 1:
            shape = trainX.shape
            seasons = shape[1] // feature_dim
            trainX = trainX[:, -seasons * feature_dim:, :]
            trainX = trainX.reshape(shape[0], -1, feature_dim)

        trainY = np.array(sliding_window[::hop, lookback:]).reshape(-1, horizon)

        return trainX, trainY

    # inverse scaling of data
    def inverse_scaling(self, data):
        inverse_data = []

        for i, value in enumerate(data):
            inverse_data.append(self.scalers[i].inverse_transform(data[i].reshape(-1, 1)))

        return np.array(inverse_data)

    # scaling of data
    def scaling(self, data, feature_dim=1):
        scaled_data = []

        for i, value in enumerate(data):
            scaled_data.append(self.scalers[i].transform(data[i].reshape(-1, 1)))

        scaled_data = np.array(scaled_data)

        # reshape feature_dim if needed
        if feature_dim > 1:
            shape = scaled_data.shape
            seasons = shape[1] // feature_dim
            scaled_data = scaled_data[:, -seasons * feature_dim:, :]
            scaled_data = scaled_data.reshape(shape[0], -1, feature_dim)

        return scaled_data
