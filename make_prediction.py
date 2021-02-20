# imports
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from scipy.stats import linregress
from alpha_vantage.timeseries import TimeSeries

# constants
PARAMS = {'IBM': (2,10)}  # the window size and number of layers for the model of each stock


def get_min(date):
    """
    This function computes the minute in the trading day (i.e. 9:30 == 0).
    :param date: a datetime.time object of the minute and the hour.
    :return: the minute
    """
    time = (int(date.hour) * 60) + int(date.minute)
    return time - 570


def get_slope_train(data):
    """
    This function calculates the linear regression slope of a set of points.
    :param data: the set of points.
    :return: the slope.
    """
    return linregress(data['MINUTE'].to_numpy(), data['CLOSE'].to_numpy())[0]


def convert_date_train(s):
    """
    Converts the string to date in the MM/DD/YYYY format.
    :param s: the string representing the date.
    :return: the date
    """
    return datetime.datetime.strptime(s, "%m/%d/%Y")


def convert_time_train(s):
    """
    Converts the string to time in the HH:MM format.
    :param s: the string representing the time.
    :return: the time
    """
    return datetime.datetime.strptime(s, "%H:%M").time()


def convert_date(s):
    """
    This function converts a string to a date in the YYYY-MM-DD HH:MM:SS format
    :return: the date
    """
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').date()


def convert_time(s):
    """
    This function converts a string to a time in the YYYY-MM-DD HH:MM:SS format
    :return: the time
    """
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').time()


def get_slope(data):
    """
    This function calculates the linear regression slope of a set of points.
    :param data: the set of points.
    :return: the slope.
    """
    return linregress(data['min'].to_numpy(), data['close'].to_numpy())[0]


def save_data(plr, slope, std, last_avg, stock_name):
    """
    Saves the data as npy files.
    :param plr: the pseudo log-return array
    :param slope: the linear regression slope array
    :param std: the standard deviation array
    :param last_avg: the last price average
    :param stock_name: The stock symbol (capitalised)
    """
    np.save("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\plr.npy", plr)
    np.save("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\slope.npy", slope)
    np.save("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\std.npy", std)
    np.save("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\last_avg.npy", np.array([last_avg]))


def load_data(stock_name):
    """
    loads the data and returns it.
    :param stock_name: The stock symbol (capitalised)
    :return: the loaded data
    """
    plr = np.load("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\plr.npy")
    slope = np.load("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\slope.npy")
    std = np.load("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\std.npy")
    last_avg = np.load("D:\BACKUP\\user\\financenn\data\\vals\\" + stock_name + "\\last_avg.npy")[0]
    return plr, slope, std, last_avg


class FirstRun:
    """
    This is a class for running the initial training run.
    """
    def __init__(self, stock_name):
        """
        :param stock_name: the stock to predict for.
        """
        self.initial_train_filepath = "D:\BACKUP\\user\\financenn\data\initial_data\\" + stock_name + "\\train.csv"
        self.initial_test_filepath = "D:\BACKUP\\user\\financenn\data\initial_data\\" + stock_name + "\\test.csv"
        self.stock = stock_name
        self.plr = None
        self.slope = None
        self.std = None
        self.last_avg = None

    def preprocess_train(self):
        """
        Prepossesses the initial train data and calculates three np arrays representing the pseudo
        log-return of each day, the slope of each day and the std of each day.
        """
        df = pd.read_csv(self.initial_train_filepath, names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
        df['DATE'] = df['DATE'].apply(convert_date_train)
        df['TIME'] = df['TIME'].apply(convert_time_train)
        df = df.loc[df['TIME'] > datetime.time(9, 29)]
        df = df.loc[df['TIME'] < datetime.time(16, 0)]
        df['MINUTE'] = df['TIME'].apply(get_min)
        df = df.drop(columns=['TIME', 'OPEN', 'HIGH', 'LOW', 'VOLUME'])
        df = df.loc[df["DATE"] < datetime.datetime(2021, 2, 1)]
        average = df.groupby('DATE').mean()['CLOSE']
        slope = df.groupby('DATE').apply(get_slope_train).to_numpy()[1:]
        plr = np.log(average / average.shift(1)).to_numpy()[1:]
        std = df.groupby('DATE').std()['CLOSE'].to_numpy()[1:]
        self.plr, self.slope, self.std = plr, slope, std
        self.last_avg = average[-1]

    def preprocess_test(self):
        """
        Prepossesses the initial test data and returns three np arrays representing the pseudo log-return of
        each day, the slope of each day and the std of each day.
        """
        test = pd.read_csv(self.initial_test_filepath)
        test['date'] = test['timestamp'].apply(convert_date)
        test['time'] = test['timestamp'].apply(convert_time)
        test = test.loc[test['time'] > datetime.time(9, 29)]
        test = test.loc[test['time'] < datetime.time(16, 0)]
        test['min'] = test['time'].apply(get_min)
        test = test.drop(columns=['timestamp', 'open', 'high', 'low', 'volume'])
        test_slope = test.groupby('date').apply(get_slope).to_numpy()
        test_std = test.groupby('date').std()['close'].to_numpy()
        test_avg = test.groupby('date').mean()['close'].to_numpy()
        test_avg = np.insert(test_avg, 0, self.last_avg)
        test_plr = np.log(test_avg[1:] / test_avg[:len(test_avg) - 1])
        self.plr = np.hstack((self.plr,test_plr))
        self.slope = np.hstack((self.slope,test_slope))
        self.std = np.hstack((self.std,test_std))
        self.last_avg = test_avg[-1]

    def first_run(self):
        """
        Creates and saves the initial arrays for learning.
        """
        self.preprocess_train()
        self.preprocess_test()
        save_data(self.plr, self.slope, self.std, self.last_avg, self.stock)


def sliding_window(arr, window_size):
    """
    This function creates a sliding window matrix from the given array and window size
    :param arr: the array to create a window for
    :param window_size: the sliding window size
    :return: the 2-d array of windows
    """
    idx = np.arange(window_size)[None,:] + np.arange(len(arr) - window_size)[:,None]
    return arr[idx]


def create_train_data(plr, slope, std, window_size):
    """
    This function creates the training data for the model.
    :param plr: the pseudo log-return array
    :param slope: the linear regression slope array
    :param std: the standard deviation array
    :param window_size: the sliding window size
    :return: the training samples and values
    """
    std_slid = sliding_window(std, window_size)
    plr_slid = sliding_window(plr, window_size)
    slope_slid = sliding_window(slope, window_size)

    X = np.hstack((std_slid,plr_slid,slope_slid))
    y = plr[window_size:,]

    return X, y


def build_nn_model(num_layers, input_size):
    """
    This function returns a DNN model built using the architecture in the research paper.
    :param num_layers: the number of hidden layers
    :param input_size: the size of the input layer
    :return: the DNN model
    """
    input = Input(shape=(input_size))
    for i in range(num_layers):
        if i == 0:
            layer = Dense(input_size, activation='tanh')(input)
        else:
            layer = Dense(math.ceil(input_size * ((num_layers - i) / num_layers)), activation='tanh')(layer)
    out = Dense(1)(layer)
    return Model(inputs=input, outputs=out)


def train_model(X_train, y_train, window_size, num_layers):
    """
    This function trains the model with the training data.
    :param X_train: the training samples
    :param y_train: the training values
    :param window_size: the sliding window size
    :param num_layers: the number of hidden layers
    :return: a trained model
    """
    model = build_nn_model(num_layers, 3 * window_size)
    model.compile(loss='mean_squared_error', optimizer=Adadelta(learning_rate=0.005, rho=0.9999, epsilon=1e-10))
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model


def train(plr, slope, std, window_size, num_layers):
    """
    This function performs the whole training process.
    :param plr: the pseudo log-return array
    :param slope: the linear regression slope array
    :param std: the standard deviation array
    :param window_size: the sliding window size
    :param num_layers: the number of hidden layers
    :return: a trained model
    """
    X_train, y_train = create_train_data(plr, slope, std, window_size)
    model = train_model(X_train, y_train, window_size, num_layers)
    return model


def get_last_data(day, month, year, api_key, plr, slope, std, last_avg, stock_name):
    """
    Returns the updated data for learning including the last trading day's data.
    :param day: The day of the month of the last trading day
    :param month: The month of the last trading day
    :param year: The year of the last trading day
    :param api_key: the Alpha Vantage API key
    :param plr: the pseudo log-return array
    :param slope: the linear regression slope array
    :param std: the standard deviation array
    :param last_avg: the last price average
    :param stock_name: The stock symbol (capitalised)
    :return: the updated data
    """
    ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='integer')
    next, meta_data = ts.get_intraday(symbol=stock_name, interval='1min', outputsize='full')
    next.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    next['date'] = next['timestamp'].apply(convert_date)
    next['time'] = next['timestamp'].apply(convert_time)
    next = next.loc[next["date"] == datetime.date(year, month, day)]
    next = next.loc[next['time'] > datetime.time(9, 29)]
    next = next.loc[next['time'] < datetime.time(16, 0)]
    next['min'] = next['time'].apply(get_min)
    next = next.drop(columns=['timestamp', 'open', 'high', 'low', 'volume'])
    next_avg = next.groupby('date').mean()['close'].to_numpy()
    last_slope = next.groupby('date').apply(get_slope).to_numpy()[0]
    last_std = next.groupby('date').std()['close'].to_numpy()[0]
    last_plr = np.log(next_avg[0] / last_avg)
    plr = np.hstack((plr, last_plr))
    slope = np.hstack((slope, last_slope))
    std = np.hstack((std, last_std))
    last_avg = next_avg[0]
    return plr, slope, std, last_avg


def update_data(initial_run, day, month, year, api_key, stock_name):
    """
    Updates the train data and saves it.
    :param initial_run: True if this is the initial run for this stock
    :param day: The day of the month of the last trading day
    :param month: The month of the last trading day
    :param year: The year of the last trading day
    :param api_key: the Alpha Vantage API key
    :param stock_name: The stock symbol (capitalised)
    :return:
    """
    plr, slope, std, last_avg = load_data(stock_name)

    if initial_run:
        return plr, slope, std, last_avg

    plr, slope, std, last_avg =  get_last_data(day, month, year, api_key, plr, slope, std, last_avg, stock_name)
    save_data(plr, slope, std, last_avg, stock_name)
    return plr, slope, std, last_avg


def predict(initial_run, day, month, year, stock_name, api_key):
    """
    This function predicts and prints the next trading day's average price.
    :param initial_run: True if this is the initial run for this stock
    :param day: The day of the month of the last trading day
    :param month: The month of the last trading day
    :param year: The year of the last trading day
    :param stock_name: The stock symbol (capitalised)
    :param api_key: the Alpha Vantage API key
    :return:
    """
    plr, slope, std, last_avg = update_data(initial_run, day, month, year, api_key, stock_name)
    window_size, num_layers = PARAMS[stock_name]
    model = train(plr, slope, std, window_size, num_layers)
    x_test = np.hstack((std[-window_size:], plr[-window_size:], slope[-window_size:]))[None, :]
    predicted_plr = model.predict(x_test)[0][0]
    prediction = np.exp(predicted_plr) * last_avg
    print("The predicted average price for the next trading day is: " + str(prediction))


def main():
    """
    The main function to read input and run the program.
    """
    initial_run = bool(int(input("Is this the first run ever for this stock? (0 for No, 1 for Yes) : ")))
    day = int(input("Please enter day of the month of the last trading day: "))
    month = int(input("Please enter the month of the last trading day: "))
    year = int(input("Please enter the year of the last trading day: "))
    stock_name = input("Please enter the stock symbol in capital letters: ")
    api_key = input("Please enter your Alpha Vantage API key: ")

    if initial_run:
        initial = FirstRun(stock_name)
        initial.first_run()

    predict(initial_run, day, month, year, stock_name, api_key)


if __name__ == "__main__":
    main()
