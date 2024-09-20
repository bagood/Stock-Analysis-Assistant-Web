import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class PD_5Days:
    def __init__(self, emiten_data):
        self.emiten_data = emiten_data

        return

    def _high_minus_low_price(self):
        """substract the highest price with the lowest stock price on each respcetive date
        """
        return self.emiten_data['High'].values - self.emiten_data['Low'].values

    def _close_minus_open_price(self):
        """substract the closing price with the opeing price on each respective date
        """
        return self.emiten_data['Close'].values - self.emiten_data['Open'].values

    def _price_moving_average(self, window_size):
        """calculate the moving average with a certain lag for several measurements
        """
        return self.emiten_data[['Open', 'High', 'Low', 'Close', 'Adj Close']] \
                        .rolling(window_size) \
                        .mean() \
                        .mean(axis=1) \
                        .values
        

    def _price_moving_std(self, window_size):
        """calculate the moving standard deviation with a certain lag for several measurements
        """
        return self.emiten_data[['Open', 'High', 'Low', 'Close', 'Adj Close']] \
                                .rolling(window_size) \
                                .mean() \
                                .mean(axis=1) \
                                .values
    
    def _volume_moving_average(self, window_size):
        """calculate the moving standard deviation with a certain lag for several measurements
        """
        return self.emiten_data['Volume'] \
                                .rolling(window_size) \
                                .mean() \
                                .values
    
    def _scale_emiten_data(self, data, columns):
        """scale a columns on the self.emiten_data using a StandardScaler
        """
        store_scalers = {}
        for col in columns:
            # standardize the self.emiten_data
            scaler = MinMaxScaler()
            emiten_data_to_scale = data[col].values.reshape(-1, 1)
            scaler.fit(emiten_data_to_scale)
            data[col] = scaler.transform(emiten_data_to_scale).T[0]

            # store the scaler into a dictionary for further use
            store_scalers[col] = scaler
        
        return (data, store_scalers)

    def create_feature_data(self):
        """creates the feature self.emiten_data to train the forecasting model
        """
        # prepares self.emiten_dataframe for the feature self.emiten_data
        self.feature_data = pd.DataFrame()
        self.feature_data['Date'] = self.emiten_data.index
        self.feature_data.set_index('Date', inplace=True)

        # compute the values for the feature self.emiten_data
        self.feature_data['5 Days Volume MA'] = self._close_minus_open_price()
        self.feature_data['10 Days Volume MA'] = self._close_minus_open_price()
        self.feature_data['5 Days MA'] =  self._price_moving_average(5)
        self.feature_data['10 Days MA'] = self._price_moving_average(10)
        self.feature_data['20 Days MA'] =  self._price_moving_average(20)
        self.feature_data['10 Days STD'] =  self._price_moving_std(10)
        
        # drop missing values and scale the feature self.emiten_data
        self.feature_data = self.feature_data.dropna()
        self.feature_data, store_scalers = self._scale_emiten_data(self.feature_data, self.feature_data.columns)

        return self.feature_data[:-4]
    
    def create_target_data(self):
        """creates the target emiten_data to train the forecasting model
        """
        self.target_data = pd.DataFrame({'Date':self.emiten_data.index[:-4]}) \
                            .set_index('Date')

        X = np.arange(0, 5, 1).reshape(-1, 1)
        reg_coef = [LinearRegression().fit(X, self.emiten_data[i:i+5][['Open', 'High', 'Low', 'Close', 'Adj Close']].mean(axis=1).values.reshape(-1, 1)).coef_[0, 0]
                        for i in range(len(self.emiten_data)-4)]

        self.target_data['Coef'] = reg_coef    
        self.target_data, store_scalers = self._scale_emiten_data(self.target_data, self.target_data.columns)

        return self.target_data[19:]