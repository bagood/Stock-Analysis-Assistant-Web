import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class prepare_dataset_functions:
    def __init__(self):
        return
    
    def scrape_stock_price(self, emiten, start=datetime(2022, 1, 1), end=datetime.now()):
        """collects relevant information on an emitten's stock
        """
        stock_data = yf.download(emiten.upper() + '.JK', start, end)

        return stock_data

    def _high_minus_low_price(self, data):
        """substract the highest price with the lowest stock price on each respcetive date
        """
        return data['High'].values - data['Low'].values

    def _close_minus_open_price(self, data):
        """substract the closing price with the opeing price on each respective date
        """
        return data['Close'].values - data['Open'].values

    def _price_moving_average(self, data, window_size):
        """calculate the moving average with a certain lag for several measurements
        """
        data_rolling_mean = data[['Open', 'High', 'Low', 'Close', 'Adj Close']].rolling(window_size).mean()
        moving_average = data_rolling_mean.mean(axis=1).values
        
        return moving_average

    def _price_moving_std(self, data, window_size):
        """calculate the moving standard deviation with a certain lag for several measurements
        """
        data_rolling_std = data[['Open', 'High', 'Low', 'Close', 'Adj Close']].rolling(window_size).std()
        moving_std = data_rolling_std.mean(axis=1).values
        
        return moving_std
    
    def _scale_data(self, data, columns):
        """scale a columns on the data using a StandardScaler
        """
        store_scalers = {}
        for col in columns:
            # standardize the data
            scaler = StandardScaler()
            data_to_scale = data[col].values.reshape(-1, 1)
            scaler.fit(data_to_scale)
            data[col] = scaler.transform(data_to_scale).T[0]
            # store the scaler into a dictionary for further use
            store_scalers[col] = scaler
        
        return (data, store_scalers)

    def create_feature_data(self, data):
        """creates the feature data to train the forecasting model
        """
        # prepares dataframe for the feature data
        feature_data = pd.DataFrame()
        feature_data['Date'] = data.index
        feature_data.set_index('Date', inplace=True)

        # compute the values for the feature data
        feature_data['High - Low'] = self._high_minus_low_price(data)
        feature_data['Close - Open'] = self._close_minus_open_price(data)
        feature_data['7 Days MA'] =  self._price_moving_average(data, 7)
        feature_data['14 Days MA'] = self._price_moving_average(data, 14)
        feature_data['21 Days MA'] =  self._price_moving_average(data, 21)
        feature_data['7 Days STD'] =  self._price_moving_std(data, 7)
        
        # drop missing values and scale the feature data
        feature_data = feature_data.dropna()
        feature_data, store_scalers = self._scale_data(feature_data, feature_data.columns)

        return feature_data
    
    def create_target_data(self, data, feature_data):
        """creates the target data to train the forecasting model
        """
        len_feature_data = len(feature_data)
        target_data = data[['Close']][-len_feature_data:]
        scaled_target_data, store_scalers = self._scale_data(target_data, target_data.columns)

        return scaled_target_data