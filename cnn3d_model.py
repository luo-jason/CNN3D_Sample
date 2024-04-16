import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class Conv3DRegressionModel:
    def __init__(self, input_shape, units=32):
        self.input_shape = input_shape
        self.units = units
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        nn = self.units
        # Add a 3D convolutional layer
        model.add(Conv3D(filters=nn, kernel_size=(3, 3, 3), activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=nn*2, kernel_size=(3, 3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=nn*4, kernel_size=(3, 3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=nn*8, kernel_size=(3, 3, 3), activation='relu'))
        model.add(BatchNormalization())

        # Flatten the output of the convolutional layer
        model.add(Flatten())

        # Add fully connected layers with dropout for regularization
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=1, activation=None))  # Regression, no activation function

        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X_train, y_train, validation_data=None, checkpoint_dir=None, epochs=100, batch_size=32):
        """
        validation_data: tuple -> (x_valid, y_valid)
        """
        callbacks = []

        ## model checkpoint
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, "model_epoch{epoch:02d}_val_loss{val_loss:.3f}.h5")
            checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
            callbacks.append(checkpoint_callback)

        ## early stopping
        early_stopping_callback = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=1)
        callbacks.append(early_stopping_callback)

        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=validation_data)

    def score(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        return self.model.summary()

    def save_weights(self, filepath):
        """only save the model weights to an HDF5 file."""
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        """Load the model weights from an HDF5 file."""
        return self.model.load_weights(filepath)
