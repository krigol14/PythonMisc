import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# scale the dataset - scaling is a technique that improves the machine learning process
# after scaling all the values are between 0 and 1
x_train = x_train/ 255
x_test = x_test/ 255

print(y_train)

# flatten the 2d arrays into 1d arrays
x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)

# create a simple NN with only two layers
model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,), activation = 'sigmoid')
])

# optimizers allow you to train the NN more efficiently
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train_flat, y_train, epochs = 5)

# evaluate the accuracy
model.evaluate(x_test_flat, y_test)

# predict all the numbers
y_predicted = model.predict(x_test_flat)
y_predicted[0]      # first prediction should be a 7 because x_test[0] represents the number 7

print(np.argmax(y_predicted[0]))
