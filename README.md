# NeuralNetwork
## General Info
This project is a simple neural network that learns how to recognize handwritten digits using [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) and [the MNIST training data](http://yann.lecun.com/exdb/mnist/). Created as my introductory project to Neural Networks and Deep Learning based on [Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/index.html).
## Examples of use
Let's create a Neural Network consisting of 30 hidden neurons and train it for 30 epochs with 10-pictures mini-batch and 3.0 learning rate:
```
network = NeuralNetwork([784, 30, 10])
network.train(training_data, 30, 10, 3.0, test_data = test_data)
```
The trained network gives us a classification rate of about 95.23% at its peak ("Epoch 28"):
```
Epoch 0: 9078 / 10000
Epoch 1: 9255 / 10000
Epoch 2: 9296 / 10000
...
Epoch 27: 9502 / 10000
Epoch 28: 9523 / 10000
Epoch 29: 9520 / 10000
```
Experimenting with the input may give slightly better results, e.g. changing the number of hidden neurons to 100 allowed me to achieve a digit recognition accuracy of 96.47%.