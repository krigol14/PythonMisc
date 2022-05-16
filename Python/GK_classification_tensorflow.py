import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt 
import numpy as np
import random 
from random import randint


def print_letter(letter):
    """
    Function that prints one of the letters of the dataset - helps to visually check what number the 2d array represents
    """
    plt.matshow(letter, cmap = 'Greys', interpolation = "nearest")
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, 9, 1))
    ax.set_yticks(np.arange(-.5, 9, 1))
    ax.set_xticklabels(np.arange(0, 10, 1))
    ax.set_yticklabels(np.arange(0, 10, 1))
    ax.grid(color = 'red', linestyle = '-', linewidth = 1)
    plt.show() 


def print_K_data():
    """
    Function that prints the sample K letters created for the training dataset
    """
    f, axarr = plt.subplots(3, 2)

    axarr[0][0].imshow(x_train[0])
    axarr[0][1].imshow(x_train[1])
    axarr[1][0].imshow(x_train[2])
    axarr[1][1].imshow(x_train[3])
    axarr[2][0].imshow(x_train[4])
    axarr[2][1].imshow(x_train[5])  

    plt.show()


def print_G_data():
    """
    Function that prints the sample G letters created for the training dataset
    """
    f, axarr = plt.subplots(3, 2)

    axarr[0][0].imshow(x_train[6])
    axarr[0][1].imshow(x_train[7])
    axarr[1][0].imshow(x_train[8])
    axarr[1][1].imshow(x_train[9])
    axarr[2][0].imshow(x_train[10])
    axarr[2][1].imshow(x_train[11])

    plt.show()


def predict(index):
    # create a test dataset to test the predictions of the NN after the training is completed - now we just replicate the training set 
    x_test_flat = x_train_flat
    y_test = y_train

    y_predicted = model.predict(x_test_flat)
    y_predicted[index]

    print_letter(x_train[index])

    result = np.argmax(y_predicted[index])

    if result == 0:
        print("\nTHE PREDICTED LETTER IS K!")
    elif result == 1: 
        print("\nTHE PREDICTED LETTER IS G!")


# initialize the list that will store the training dataset
x_train = []

#-----LETTER K SAMPLES-----
rows, cols = (10, 10)       # the letters are being digitized in a 10*10 grid

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[2][3] = letter[2][7] = letter[3][3] = letter[3][6] = letter[4][3] = letter[4][5] = letter[5][3] = letter[5][4] = letter[6][3] = letter[7][3] = letter[8][3] = letter[6][5] = letter[7][6] = letter[8][7] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[2][3] = letter[2][6] = letter[3][3] = letter[3][6] = letter[4][3] = letter[4][5] = letter[5][3] = letter[5][4] = letter[6][3] = letter[7][3] = letter[8][3] = letter[6][5] = letter[7][6] = letter[8][6] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[2][3] = letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[8][3] = letter[2][6] = letter[3][5] = letter[4][4] = letter[5][4] = letter[6][5] = letter[7][6] = letter[8][7] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[2][3] = letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[8][3] = letter[2][6] = letter[3][5] = letter[4][4] = letter[5][5] = letter[6][6] = letter[7][6] = letter[8][6] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[0][3] = letter[1][3] = letter[2][3] = letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[8][3] = letter[9][3]  = letter[2][6] = letter[1][7] = letter[0][8] = letter[3][5] = letter[4][4] = letter[5][5] = letter[6][6] = letter[7][7] = letter[8][8] = letter[9][9] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[1][2] = letter[1][3] = letter[2][3] = letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[8][3] = letter[8][2]  = letter[2][6] = letter[1][7] = letter[1][6] = letter[3][5] = letter[4][4] = letter[5][5] = letter[6][6] = letter[7][6] = letter[8][6] = letter[8][7] = 1
x_train.append(letter)


#-----LETTER G SAMPLES-----
letter = [[0 for i in range(cols)] for j in range(rows)]
letter[1][4] = letter[2][4] = letter[3][4] = letter[4][4] = letter[5][4] = letter[6][4] = letter[7][4] = letter[8][4] = letter[1][5] = letter[1][6] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[1][3] = letter[2][3] = letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[8][3] = letter[1][4] = letter[1][5] = letter[1][6] = letter[1][7] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[1][3] = letter[2][3] = letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[8][3] = letter[1][4] = letter[1][5] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[8][3] = letter[2][6] = letter[2][5] = letter[2][4] = 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[1][2] = letter[2][2] = letter[3][2] = letter[4][2] = letter[5][2] = letter[6][2] = letter[7][2] = letter[8][2] = letter[8][1] = letter[8][3] = letter[1][3] = letter[1][4] = letter[1][5] = letter[1][6] = letter[2][6]= 1
x_train.append(letter)

letter = [[0 for i in range(cols)] for j in range(rows)]
letter[3][3] = letter[4][3] = letter[5][3] = letter[6][3] = letter[7][3] = letter[3][4] = letter[3][5] = 1
x_train.append(letter)


# y_train contains the actual value that each letter in the x_train represents
# 0 -> THE LETTER IS K
# 1 -> THE LETTER IS G
y_train = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# convert the list to numpy array so that we can use the reshape function on it
x_train_np = np.array(x_train)
# flatten the 2d array into 1d array so that we can pass the values in the input layer neurons of the NN
x_train_flat = x_train_np.reshape(len(x_train), 10*10)

# create a simple NN with only two layers - input and output layer
model = keras.Sequential([
    keras.layers.Dense(2, input_shape = (100,), activation = 'sigmoid')
])

# optimizers allow you to train the NN more efficiently
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train_flat, y_train, epochs = 100)

rnd = random.randint(0, 11)
predict(rnd)

print_K_data()
print_G_data()
