import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class prepare_dataset_functions:
    def __init__(self):
        return
    
    def scrape_stock_price(self, emiten, start=datetime(2022, 1, 1), end=datetime.now()):
        stock_data = yf.download(emiten.upper() + '.JK', start, end).reset_index()

        return stock_data

    def _high_minus_low_price(self, data):
        return data['High'].values - data['Low'].values

    def _close_minus_open_price(self, data):
        return data['Close'].values - data['Open'].values

    def _price_moving_average(self, data, window_size):
        data_rolling_mean = data[['Open', 'High', 'Low', 'Close', 'Adj Close']].rolling(window_size).mean()
        moving_average = data_rolling_mean.mean(axis=1).values
        
        return moving_average

    def _price_moving_std(self, data, window_size):
        data_rolling_std = data[['Open', 'High', 'Low', 'Close', 'Adj Close']].rolling(window_size).std()
        moving_std = data_rolling_std.mean(axis=1).values
        
        return moving_std
    
    def _scale_data(self, data, columns):
        scaled_data = pd.DataFrame()
        store_scalers = {}
        for col in columns:
            scaler = StandardScaler()
            data_to_scale = data[col].values.reshape(-1, 1)
            scaler.fit(data_to_scale)
            scaled_data[col] = scaler.transform(data_to_scale).T[0]
            store_scalers[col] = scaler
        
        return (scaled_data, store_scalers)

    def create_training_data(self, data):
        training_data = pd.DataFrame()
        training_data['High - Low'] = self._high_minus_low_price(data)
        training_data['Close - Open'] = self._close_minus_open_price(data)
        training_data['7 Days MA'] =  self._price_moving_average(data, 7)
        training_data['14 Days MA'] = self._price_moving_average(data, 14)
        training_data['21 Days MA'] =  self._price_moving_average(data, 21)
        training_data['7 Days STD'] =  self._price_moving_std(data, 7)
        training_data = training_data.dropna()
        scaled_training_data, store_scalers = self._scale_data(training_data, training_data.columns)

        return scaled_training_data
    
    def create_target_data(self, data, training_data):
        len_training_data = len(training_data)
        target_data = data[['Close']][-len_training_data:]
        scaled_target_data, store_scalers = self._scale_data(target_data, target_data.columns)

        return scaled_target_data