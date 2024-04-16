import numpy as np
import os
from cnn3d_model import Conv3DRegressionModel

## 3D input shape + channels
## Here, I assume the input is 10x10x10 cubic and the number of weather variables is 5(e.g. temperature, pressure, u, v, wind, rain)
## You can change the input_shape to fit in your needs
input_shape = (10, 10, 10, 5)  

## init the model
model = Conv3DRegressionModel(input_shape)
## compile the model before training
## This function confirm the loss & optimizer(defulat is mean squared error & Adam optim)
model.compile_model()
model.summary()

# Sample training and testing
## The dimension should be (samples, length, width, height, weather_features)
## Here, I assume there are 100 training samples and 20 validation samples
## So, the training/validation data's dim is (100, 10, 10, 10, 5) / (20, 10, 10, 10, 5)
## and the training/validation label's dim is (100, 1) / (20, 1)
X_train = np.random.normal(size=(100, *input_shape))
y_train = np.random.normal(size=(100, 1))
X_valid = np.random.normal(size=(20, *input_shape))
y_valid = np.random.normal(size=(20, 1))

## Train the model
model.fit(X_train, y_train, 
          validation_data=(X_valid, y_valid),
          ## you can pass a directory path, the good model weights will be saved during training
          ## checkpoint_dir="./model_check/"
          checkpoint_dir=None, 
          epochs=100)

loss = model.score(X_valid, y_valid)
print("Test Loss:", loss)

print("Save model weights")
os.makedirs("model_weights", exist_ok=True)
model.save_weights("model_weights/cnn_3d_weights.h5")
