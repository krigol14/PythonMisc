import numpy as np 

# training input data
inputs = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 
                   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 
                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                   0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# training output data
# letter K is represented by 0's, while letter G is represented by 1's 
outputs = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])

# create a NN class
class NeuralNetwork:

    def __init__(self, inputs, outputs):
        """
        Initialize the variables of the class and create a matrix that holds initially random weights.
        """
        self.inputs = inputs
        self.outputs = outputs         
        self.weights = 2 * np.random.random((100, 1)) - 1

    
    def sigmoid(self, x):
        """
        Sigmoid function used as activation function for the neural network.
        """
        return 1 / (1 + np.exp(-x))


    def sigmoid_der(self, x):
        """
        Function that finds the derivative of the sigmoid function for the input given - used for adjusting the weights during backpropagation
        """
        return x * (1 - x)


    def feed_forward(self):
        """
        Function that feeds forward the data of the input layer.
        """
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))


    def backpropagation(self):
        """
        Function that performs the backpropagation by adjusting the weights matrix based on the error rate.
        """
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid_der(self.hidden)
        self.weights += np.dot(self.inputs.T, delta)


    def train(self, iterations = 25000):
        """
        Function used to train the neural network for 25000 iterations.
        """
        for iteration in range(iterations):
            # feed forward and produce an output
            self.feed_forward()
            # adjust the weights using the backpropagation function
            self.backpropagation()    
            
                            
    def predict(self, new_input):
        """
        Function that predicts the output of new data given to the neural network.
        """
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction    

# initialize neural network
neural_network = NeuralNetwork(inputs, outputs)
# train the neural network 
neural_network.train()

# create two new examples to predict 
example1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                      0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])   # letter K example

example2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
                      0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                      0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])   # letter G example 

example3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])   # letter G example

prediction1 = neural_network.predict(example1)
prediction2 = neural_network.predict(example2)
prediction3 = neural_network.predict(example3)

if prediction1 > 0.5:
    print("The first letter is a G!")
else:
    print("The first letter is a K!")

if prediction2 > 0.5:
    print("The second letter is a G!")
else:
    print("The second letter is a K!")

if prediction2 > 0.5:
    print("The third letter is a G!")
else:
    print("The third letter is a K!")
