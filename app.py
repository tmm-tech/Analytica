from flask import Flask
import yfinance as yf

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    print(__name__)
    return 'Analytica'


def predict_value(ticker):
    data = yf.download(tickers=ticker, period='13y', interval='1d')

    type(data)

    # data.head()

    # data.tail()

    opn = data[['Open']]

    # opn.plot()

    ds = opn.values

    # ds

    # plt.plot(ds)

    import numpy as np

    from sklearn.preprocessing import MinMaxScaler

    # Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0, 1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))

    len(ds_scaled), len(ds)

    # Defining test and train data sizes
    train_size = int(len(ds_scaled) * 0.75)
    test_size = len(ds_scaled) - train_size

    train_size, test_size

    # Splitting data between train and test
    ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]

    len(ds_train), len(ds_test)

    # creating dataset in time series for LSTM model
    # X[100,120,140,160,180] : Y[200]
    def create_ds(dataset, step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset) - step - 1):
            a = dataset[i:(i + step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    # Taking 100 days price as one record for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train, time_stamp)
    X_test, y_test = create_ds(ds_test, time_stamp)

    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    # Creating LSTM model using keras
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1, activation='linear'))
    model.summary()

    # Training model with adam optimizer and mean squared error loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

    # PLotting loss, it shows that loss has decreased significantly and model trained well
    # loss = model.history.history['loss']
    # plt.plot(loss)

    # Predicitng on train and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform to get actual value
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)
    # #Comparing using visuals
    # plt.plot(normalizer.inverse_transform(ds_scaled))
    # plt.plot(train_predict)
    # plt.plot(test_predict)

    type(train_predict)

    test = np.vstack((train_predict, test_predict))

    # #Combining the predited data to create uniform data visualization
    # plt.plot(normalizer.inverse_transform(ds_scaled))
    # plt.plot(test)

    len(ds_test)

    # Getting the last 100 days records
    fut_inp = ds_test[271:]

    fut_inp = fut_inp.reshape(1, -1)
    tmp_inp = list(fut_inp)
    fut_inp.shape
    print(fut_inp.shape)

    # Predicting next 30 days price suing the current data
    # It will predict in sliding window manner (algorithm) with stride 1
    lst_output = []
    n_steps = len(ds_test) - 271
    i = 0
    while i < 30:

        if len(tmp_inp) > n_steps:
            fut_inp = np.array(tmp_inp[1:])
            fut_inp = fut_inp.reshape(1, -1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    print(lst_output)

    len(ds_scaled)

    # # Creating a dummy plane to plot graph one after another
    # # Creating a dummy plane to plot graph one after another
    # plot_new = np.arange(1, 101)
    # plot_pred = np.arange(101, 131)

    # Creating list of the last 100 data
    ds_new = ds_scaled.tolist()

    len(ds_new)

    # Entends helps us to fill the missing value with approx value
    ds_new.extend(lst_output)
    # plt.plot(ds_new[1200:])

    # Creating final data for plotting
    final_graph = normalizer.inverse_transform(ds_new).tolist()

    # #Plotting final results with predicted value after 30 Days
    # plt.plot(final_graph)
    # plt.ylabel("Price")
    # plt.xlabel("Time")
    # plt.title("{0} prediction of next month open".format(stock_symbol))
    # plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
    # plt.legend()

    answer = format(round(float(*final_graph[len(final_graph) - 1]), 2))
    return str(answer)


@app.route('/prediction/<stock>', methods=['POST', 'GET'])
def predict(stock):
    try:
        result = predict_value(stock)
        return str(result)
    except EOFError as e:
        print("as: " + str(e))


if __name__ == '__main__':
    app.run()
