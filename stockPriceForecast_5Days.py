import pytz
import keras
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

class SPF_5Days:
    def __init__(self, feature_data, target_data):
        jkt = pytz.timezone('Asia/Jakarta')
        current_hour = datetime.now(jkt).hour
        current_day = datetime.now(jkt).strftime("%A")

        # forecast for today's price if it is still under 3 pm and a weekday
        self.forecast_today = (current_hour < 15) & (current_day not in ['Saturday', 'Sunday'])

        self.feature_data = feature_data
        self.target_data = target_data

    def _calculate_split_index(self, data, training_perc):
        return np.floor(len(data) * training_perc).astype('int')

    def slice_training_testing_dataset(self, training_perc):
        """slice the dataset to create training and testing dataset
        """
        if self.forecast_today:
            self.feature_data = self.feature_data[:-2] 
            self.target_data = self.target_data[1:-1]
        else:
            self.feature_data = self.feature_data[:-1] 
            self.target_data = self.target_data[1:]

        split_index = self._calculate_split_index(self.feature_data, training_perc)
        
        self.train_feature = self.feature_data.values[:split_index]
        self.train_target = self.target_data.values[:split_index]
        self.test_feature = self.feature_data.values[split_index:]
        self.test_target = self.target_data.values[split_index:]
        
        return
    
    def slice_forecast_dataset(self):
        """slice the dataset to create feature data for forecast
        """
        if self.forecast_today:
            self.forecast_feature = self.feature_data.tail(3).head(2).values
            return
        
        self.forecast_feature = self.feature_data.tail(2).values
        return
    
    def create_model(self):
        """Generates the model using recurrenct neural network
        Args:
        n_input (int) - the number of data contained within a list used for fitting

        Returns:
        model (TF Keras Model) - the generated reccurrent neural network
        """
        model = tf.keras.models.Sequential([
                tf.keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(self.feature_data.shape[1], 1)),
                tf.keras.layers.SimpleRNN(40, return_sequences=True),
                tf.keras.layers.SimpleRNN(24),
                tf.keras.layers.Dense(8),
                tf.keras.layers.Dense(1)
                ])
        model.summary()

        return model
    
    def find_best_learningrate(self, model, start_lr):
        """Used to find the best learningrate for the generated recurrenct neural network
        Args:
        model (TF Keras Model) - the generated reccurrent neural network
        self.train_target (array of float) - contains the data used for training the model
        """
        init_weights = model.get_weights()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: start_lr * 10**(epoch / 25))
        optimizer = tf.keras.optimizers.SGD(momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(), 
                        optimizer=optimizer)
        history = model.fit(self.train_feature,
                            self.train_target, 
                            epochs=100, 
                            callbacks=[lr_schedule])

        lrs = start_lr * (10 ** (np.arange(100) / 25))
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, history.history["loss"])
        plt.show()

        tf.keras.backend.clear_session()
        model.set_weights(init_weights)
        
        return 

    def model_fitting(self, model, learning_rate, epochs):
        """Fits the model using the training data
        Args:
        model (TF Keras Model) - the generated reccurrent neural network
        learning_rate (float) - the learning rate used for fitting the model
        epochs (int) - the number of epoch used to train the model
        self.train_target (array of float) - contains the data used for training the model

        Returns:
        model (TF Keras Model) - the fitted model
        """
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, 
                                                momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(),
                        optimizer=optimizer,
                        metrics=["mae"])
        _ = model.fit(self.train_feature,
                        self.train_target,  
                        epochs=epochs)
    
        return model
    
    def save_model_in_keras(self, model, emiten):
        """saves the model in keras file format
        """
        model.save(f'DeepLearningModels/{emiten}.keras')

        return

    def model_forecast(self, data, emiten):
        """create forecasts utilizing the developed model and the feature data
        """
        model = keras.models.load_model(f'DeepLearningModels/{emiten}.keras')
        forecast = model.predict(data)
        
        return forecast