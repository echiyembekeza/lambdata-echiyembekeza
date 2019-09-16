import quandl
import pandas as pd

# Choose the stock
# symbol = 'EOD/AAPL'
# start = '1988-01-01'
# end = '2015-12-31'
# data_raw = get_data_quandl(symbol, start, end)
# data = generate_features(data_raw)


class Choose_Stock:
    """Select the stock of choice"""
    def __init__(self, symbol='EOD/AAPL', start='1988-01-01', end='2015-12-31'):
        self.symbol = symbol
        self.start = start
        self.end = end
