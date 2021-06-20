# data download
import yfinance as yf
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

from sklearn.preprocessing import MinMaxScaler


TICKERS = ['AAPL', 'GOOG', 'FB', 'AMZN']


def download():
    data = yf.download("AAPL GOOG FB AMZN", start="2017-01-01", end="2021-05-01")

    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)


# split a multivariate sequence into samples
# 1. Select last n columns # df1 = df.iloc[:,-n:]
# 2. Exclude last n columns # df1 = df.iloc[:,:-n]
def split_sequences(sequences, n_steps, n_y_cols):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-n_y_cols], sequences[i:i + 1, -n_y_cols:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # download()
    # with open('data.pickle', 'rb') as f:
    #     data = pickle.load(f)
    #
    # df = data['Adj Close']

    # input_s1 = df['AAPL'].to_numpy()
    # input_s2 = df['GOOG'].to_numpy()
    # output_s1 = df['AAPL'].shift(-n_timesteps).to_numpy()
    # output_s2 = df['GOOG'].shift(-n_timesteps).to_numpy()
    n = 1000
    n_timesteps = 5
    n_features = 2

    input_s1 = np.array([i / 100 + np.random.uniform(-1, 3) for i in range(n)])
    input_s2 = np.array([i / 100 + np.random.uniform(-3, 5) + 2 for i in range(n)])
    output_s1 = np.roll(input_s1, -n_timesteps)
    output_s2 = np.roll(input_s2, -n_timesteps)

    # convert to [row, col] structure, num of elements rows x 1 col
    input_s1 = input_s1.reshape(len(input_s1), 1)
    input_s2 = input_s2.reshape(len(input_s2), 1)

    output_s1 = output_s1.reshape(len(output_s1), 1)
    output_s2 = output_s2.reshape(len(output_s2), 1)


    # scaler1 = MinMaxScaler(feature_range=(0, 1))
    # input_scaled_s1 = scaler1.fit_transform(input_s1)
    # output_scaled_s1 = scaler1.transform(output_s1)
    #
    # scaler2 = MinMaxScaler(feature_range=(0, 1))
    # input_scaled_s2 = scaler2.fit_transform(input_s2)
    # output_scaled_s2 = scaler2.transform(output_s2)

    #dataset = np.hstack((input_scaled_s1, input_scaled_s2, output_scaled_s1, output_scaled_s2))
    dataset = np.hstack((input_s1, input_s2, output_s1, output_s2))

    dataset = dataset[~np.isnan(dataset).any(axis=1)]

    print(dataset)

    x, y = split_sequences(dataset, n_timesteps, n_y_cols=2)

    print(x.shape, y.shape)

    for i in range(3):
        print(x[i], y[i])

    # Create model .

    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(x, y, epochs=400, verbose=1)
    # demonstrate prediction
    x_input = np.array([[27.52194405, 794.02001953],
                        [27.82876396, 806.15002441],
                        [28.08366013, 806.65002441],
                        [28.11197853, 804.78997803],
                        [28.2630291, 807.90997314]])
    x_input = x_input.reshape((1, n_timesteps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)
