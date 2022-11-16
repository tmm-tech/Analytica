import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import LSTM
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


def predict_value(ticker):
    start_date = '2009-01-01'  # as strings
    end_date = '2018-12-31'  # as strings
    data = yf.download(ticker, start_date, end_date)
    # type(data)
    # initialize
    opn = data[['Open']]
    ds = opn.values
    # Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0, 1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))
    # length of the ds_scaled amd ds
    # len(ds_scaled), len(ds)
    # Selecting the Open column that we’ll be used in our modeling
    dataset_train = data.iloc[:, 1:2].values
    # notice ==>> df.iloc[:, 1:2] returns a dataframe whereas df.iloc[:, 1] returns a series notice ==>"Open" column is
    # the starting price while the Close column is the final price of a stock on a particular trading day.
    # Normalization of the training set - Transform features by scaling each feature to a given range.
    normalizer = MinMaxScaler(feature_range=(0, 1))
    train_set_scaled = normalizer.fit_transform(dataset_train)
    # Creating data with timesteps for the train_set_scaled
    # timesteps ==>> to train the algorithm, I will check 45 rows on each step
    x_train = []
    y_train = []

    for i in range(45, len(ds_scaled)):
        x_train.append(train_set_scaled[i - 45:i, 0])
        # in other words => first iteration: [45-45:45, collumn index: 0] (from zero until 44)
        # => segunda iteration: [46-45:46, collumn index: 0] (from 1 until 45)
        # => segunda iteration: [47-45:47, collumn index: 0] (from 2 until 46)
        y_train.append(train_set_scaled[i, 0])  # in other words => stores the index that it wants to predict

    x_train, y_train = np.array(x_train), np.array(y_train)  # I want the data to be in an numpy array
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Reshape method will create dimensions (it will have 3 dimensions: the two we had and one more)
    # NOTICE == >> we have to reshape our data to 3D because tensorflow requires it to run
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(LSTM(units=50))  # here i don't need the history cos I only need the result

    # we add the Dense layer that specifies the output of 1 unit, the output is 1
    regressor.add(Dense(units=1, activation='linear'))
    # last layer is a dense layer with linear because I have continuous values
    # we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error.
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    regressor.fit(x_train, y_train, epochs=100, batch_size=64)
    start_date = '2019-01-01'  # as strings
    end_date = '2022-11-11'  # as strings
    dataset_test = yf.download(ticker, start_date, end_date)
    # Selecting the Open column that we’ll use in our modeling
    real_stock_price = dataset_test.iloc[:, 1:2].values
    # dataset_test
    pd.DataFrame({'Open': dataset_train[0]})

    pd.DataFrame(dataset_train)

    dataset_totality = pd.concat((pd.DataFrame(dataset_train), dataset_test['Open']), axis=0)

    inputs = dataset_totality[len(dataset_totality) - len(dataset_test) - 45:].values
    # No meu dataset total eu quero garantir q pego só as linhas do dataset treino.
    inputs = inputs.reshape(-1, 1)  # reshape to turn it into a vector

    inputs = normalizer.transform(inputs)  # AFINAL = usar aqui so transform ou fit_transform
    x_test = []
    for i in range(45, len(dataset_test)):
        x_test.append(inputs[i - 45:i, 0])
        # in other words => [45-45:45, collumn index: 0]

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # NOTICE == >> we have to reshape our data to 3D DataFrame (panel) A data frame is a two-dimensional data structure,
    # that is, the data is aligned in rows and columns in a table, a three-dimensional data structure is called panel.

    # Making the predictions
    predicted_stock_price = regressor.predict(x_test)
    # Get back the stock prices in normal readable format
    predicted_stock_price = normalizer.inverse_transform(predicted_stock_price)
    len(dataset_test)

    # Getting the last 100 days records
    fut_inp = dataset_test[151:]
    fut_inp = fut_inp.values.reshape(1, -1)
    print(fut_inp.shape)
    len(train_set_scaled)
    ds_new = train_set_scaled.tolist()

    len(ds_new)

    # Entends helps us to fill the missing value with approx value
    # Creating final data for plotting
    final_graph = normalizer.inverse_transform(ds_new).tolist()
    # 30 days stock prediction
    answer = format(round(float(*final_graph[len(final_graph) - 1]), 2))
    print("Stock prediction in the next 30 days: " + answer)
    # This will compute the mean of the squared errors.
    # mean_squared_error = measures the average of the squares of the errors /
    mlp_mse = mean_squared_error(real_stock_price[0:983], predicted_stock_price)
    print("Accuracy with MSE: ", mlp_mse)
    # Notice - if one of the errors is too big, it will impact the results a lot.
    # This is why "mean absolute percentage error" is used as a great solution to calculate the error.
    mlp_mae = mean_absolute_error(real_stock_price[0:983], predicted_stock_price)
    print("Accuracy with MAE: ", mlp_mae)

    return str(answer)
