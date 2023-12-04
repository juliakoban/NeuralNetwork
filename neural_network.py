import numpy as np
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, neurons_in_layers):
        self.number_of_layers = len(neurons_in_layers)
        self.neurons_in_layers = neurons_in_layers
        # randn() - numbers from a standard normal distribution (mean=0, variance=1)
        # randn(y, x) - y number of rows, x - number of columnss in the returned array of random numbers
        # [start:stop:end] - slicing the array
        # zip() function takes iterables, aggregates them in a tuple, and returns it
        self.biases = [np.random.randn(y, 1) for y in neurons_in_layers[1:]] # matrix  
        self.weights = [np.random.randn(y, x) for x, y in zip(neurons_in_layers[:-1], neurons_in_layers[1:])] # matrix
        # print(self.weights[0], self.weights[1], self.biases[0], self.biases[1]) # weights array from second to third layer
        # print(self.biases[0][0][0])
    
    def train(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        # train the neural network using mini-batch stochastic gradient descent
        # "epoch" refers to one complete pass through the entire training dataset
        # during each epoch, the neural network's weights and biases are updated
        # based on the training data to minimize the cost function

        for _ in range(epochs):
            np.random.shuffle(training_data)
            # distributing training data(pictures) as mini-batches of mini_batch_size pictures
            mini_batches = [training_data[i : i + mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(_, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(_))

    def update_mini_batch(self, mini_batch, eta):
        # update the network's weights and biases by applying a single step gradient descent
        # using backpropagation to a single mini batch 
        # mini-batch - set of pictures(training examples) 
        # mini_batch is a list of tuples (training_input(picture), desired_output)

        nabla_bias = [np.zeros(bias_array.shape) for bias_array in self.biases] # dC/db
        nabla_weight = [np.zeros(weight_array.shape) for weight_array in self.weights] # dC/dw

        for training_input, desired_output in mini_batch:
            # dC/db, dC/dw for one picture in mini-batch
            delta_nabla_bias, delta_nabla_weight = self.backward_propagation(training_input, desired_output)
            nabla_bias = [nb + dnb for nb, dnb in zip(nabla_bias, delta_nabla_bias)] 
            nabla_weight = [nw + dnw for nw, dnw in zip(nabla_weight, delta_nabla_weight)] 

        self.biases = [b -  (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_bias)]
        self.weights = [w -  (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_weight)]

        # print(self.biases)
        # print(self.weights)
    
    def propagation(self, a):
        # a - vector of activations of the layer of neurons
        for bias_array, weight_array in zip(self.biases, self.weights): 
            a = sigmoid(np.dot(weight_array, a) + bias_array)
        
        return a # returns vector of activations of the output layer of neurons

    def backward_propagation(self, training_input, desired_output):
        # training_input - vector of activations of the layer of neurons(pixel grayscale number)
        # computing gradient of the cost function in reference to biases and weights for a single mini-batch

        # creating empty biases and weights matricies(same shape)
        nabla_bias = [np.zeros(bias_array.shape) for bias_array in self.biases] # dC/db
        nabla_weight = [np.zeros(weight_array.shape) for weight_array in self.weights] # dC/dw

        # PROPAGATION
        activations_vector = training_input
        activations_vectors = [activations_vector] # list to store all the activations vectors, layer by layer
        z_vectors = [] # list to store all the z vectors, layer by layer

        for bias_array, weight_array in zip(self.biases, self.weights): 
            z = np.dot(weight_array, activations_vector) + bias_array
            z_vectors.append(z)
            activations_vector = sigmoid(z)
            activations_vectors.append(activations_vector)
            
        # print(activations_vectors)
        # print(z_vectors)

        # BACKWARD PASS
        # activations_vectors[-1] - output activations of neurons
        
        delta = self.cost_derivative(activations_vectors[-1], desired_output) * sigmoid_derivative(z_vectors[-1])
        nabla_bias[-1] = delta
        nabla_weight[-1] = np.dot(delta, activations_vectors[-2].transpose())
        
        for layer in range(2, self.number_of_layers):
            z = z_vectors[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_derivative(z) 
            nabla_bias[-layer] = delta
            nabla_weight[-layer] = np.dot(delta, activations_vectors[-layer - 1].transpose())

        # print("biases: ", self.biases)
        # print("weights: ", self.weights)
        # print("nabla biases: ", nabla_bias)
        # print("nabla weights: ",nabla_weight)

        return (nabla_bias, nabla_weight)

    def cost_derivative(self, output_activations_vector, desired_output):
        # dC_x / da
        return (output_activations_vector - desired_output)
    
    def evaluate(self, test_data):
        # return the number of test inputs for which the neural
        # network outputs the correct result
        test_results = [(np.argmax(self.propagation(training_input)), desired_output) for (training_input, desired_output) in test_data]
        return sum(int(result == desired_output) for (result, desired_output) in test_results)

        
network = NeuralNetwork([784, 30, 10])
network.train(training_data, 30, 10, 3.0, test_data = test_data)

