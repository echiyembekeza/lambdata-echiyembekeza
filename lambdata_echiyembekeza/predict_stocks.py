import quandl
import pandas as pd

# Choose the stock
# symbol = 'EOD/AAPL'
# start = '1988-01-01'
# end = '2015-12-31'
# data_raw = get_data_quandl(symbol, start, end)
# data = generate_features(data_raw)

data_raw = get_data_quandl(symbol, start, end)
data = generate_features(data_raw)

# Token to use the Quandl API
authtoken = 'CYmZJQTZk9cM-5LUZxD_'
def get_data_quandl(symbol, start_date, end_date):
    data = quandl.get(symbol, start_date=start_date,
                      end_date=end_date, authtoken=authtoken)
    return data

# Feature engineering
def generate_features(df):
    """Generate features for a stock/index based on historical price and performance
    Args:
        df(dataframe with columns "Open", "Close", "High", "Low", "Volume", "Adjusted Close")
    Returns:
        dataframe, data set with new features
        """
    df_new = pd.DataFrame()
    # 6 original features
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    # Shift index by 1, in order to take the value of previous day. For example, [1, 3, 4, 2] -> [N/A, 1, 3, 4]
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)
    # 31 original features
    # average price
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    # average volume
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    # standard deviation of prices
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    # rolling_mean calculates the moving standard deviation given a window
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    # standard deviation of volumes
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    # return
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean()
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean()
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean()
    # the target
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    # This will drop rows with any N/A value, which is by-product of moving average/std.
    return df_new

import datetime
start_train = datetime.datetime(1988, 1, 1, 0, 0)
end_train = datetime.datetime(2014, 12, 31, 0, 0)
data_train = data.loc[start_train:end_train]

X_columns = list(data.drop(['close'], axis=1).columns)
y_column = 'close'
X_train = data_train[X_columns]
y_train = data_train[y_column]

print(X_train.shape)
y_train.shape

start_test = datetime.datetime(2015, 1, 1, 0, 0)
end_test = datetime.datetime(2015, 12, 31, 0, 0)
data_test = data.loc[start_test:end_test]
X_test = data_test[X_columns]
y_test = data_test[y_column]

print(X_test.shape)
y_test.shape

def compute_prediction(X, weights):
    predictions = np.dot(X, weights)
    return predictions

def update_weights_gd(X_train, y_train, weights, learning_rate):
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X, y, weights):
    predictions = compute_prediction(X, weights)
    cost = np.mean((predictions - y) ** 2 / 2.0)
    return cost

def train_linear_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
        weights = np.zeros(X_train.shape[1])
        for iteration in range(max_iter):
            weights = update_weights_gd(X_train, y_train, weights, learning_rate)
            if iteration % 100 == 0:
                print(compute_cost(X_train, y_train, weights))
    return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)




train_linear_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False)
predict(data, weights)
