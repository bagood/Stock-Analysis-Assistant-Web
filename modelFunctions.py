import pytz
import keras
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

class model_functions:
    def __init__(self):
        jkt = pytz.timezone('Asia/Jakarta')
        current_hour = datetime.now(jkt).hour
        self.forecast_today = current_hour < 15

    def slice_dataset(self, feature_data, target_data, split_index):
        train_feature = feature_data.values[:split_index]
        train_target = target_data.values[:split_index]
        test_feature = feature_data.values[split_index:]
        test_target = target_data.values[split_index:]

        return (train_feature, train_target, test_feature, test_target)
    
    def window_feature_data(self, window_size, data, test_bool=False):
        windowed_data = []
        start_index = (len(data) // window_size) * window_size
        data = data[-start_index:]
        for index in range(len(data)-window_size):
            temp_windowed_data = data[index: index+window_size]
            windowed_data.append(temp_windowed_data)
        if test_bool == False:
            windowed_data = np.array(windowed_data[:-1])
        elif test_bool and self.forecast_today:
            # today's forecast
            windowed_data = np.array(windowed_data[:-1])
        elif test_bool and self.forecast_today == False:
            # tomorrow's forecast
            windowed_data = np.array(windowed_data)
        
        return windowed_data

    def adjust_target_data(self, window_size, data):
        start_index = (len(data) // window_size) * window_size
        data = data[-start_index + window_size + 1:]

        return data
    
    def create_model(self, window_size):
        """Generates the model using recurrenct neural network
        Args:
        window_size (int) - the number of data contained within a list used for fitting

        Returns:
        model (TF Keras Model) - the generated reccurrent neural network
        """
        model = tf.keras.models.Sequential([
                tf.keras.layers.SimpleRNN(40, return_sequences=True, input_shape=(window_size, 6)),
                tf.keras.layers.SimpleRNN(40),
                tf.keras.layers.Dense(8),
                tf.keras.layers.Dense(1)
                ])
        model.summary()

        return model
    
    def find_best_learningrate(self, model, train_feature, train_target):
        """Used to find the best learningrate for the generated recurrenct neural network
        Args:
        model (TF Keras Model) - the generated reccurrent neural network
        train_target (array of float) - contains the data used for training the model
        """
        init_weights = model.get_weights()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-2 * 10**(epoch / 80))
        optimizer = tf.keras.optimizers.SGD(momentum=0.9)
        model.compile(
            loss=tf.keras.losses.Huber(), 
            optimizer=optimizer)
        history = model.fit(
            train_feature,
            train_target, 
            epochs=100, 
            callbacks=[lr_schedule])
        lrs = 1e-2 * (10 ** (np.arange(100) / 80))
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, history.history["loss"])
        plt.show()
        tf.keras.backend.clear_session()
        model.set_weights(init_weights)

        return 

    def model_fitting(self, model, learning_rate, epochs, train_feature, train_target):
        """Fits the model using the training data
        Args:
        model (TF Keras Model) - the generated reccurrent neural network
        learning_rate (float) - the learning rate used for fitting the model
        epochs (int) - the number of epoch used to train the model
        train_target (array of float) - contains the data used for training the model

        Returns:
        model (TF Keras Model) - the fitted model
        """
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=["mae"])
        _ = model.fit(            
            train_feature,
            train_target,  
            epochs=epochs)
        
        return model

    def model_forecast(self, emiten, test_feature, test_target):
        model = keras.models.load_model(f'Models/{emiten}.h5')
        forecast = model.predict(test_feature).T[0]
        actual = test_target.T[0]
        
        return (forecast, actual)