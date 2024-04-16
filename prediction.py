import numpy as np
from cnn3d_model import Conv3DRegressionModel

## Same as training
input_shape = (10, 10, 10, 5)

model = Conv3DRegressionModel(input_shape)
model.load_weights("model_weights/cnn_3d_weights.h5")

## say you have 20 samples to test
X_test = np.random.normal(size=(20, *input_shape))

predictions = model.predict(X_test)

print("prediction output shape:", predictions.shape)
