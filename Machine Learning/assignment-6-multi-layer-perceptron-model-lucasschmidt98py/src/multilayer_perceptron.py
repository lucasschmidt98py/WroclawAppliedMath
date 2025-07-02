import numpy as np
import mnist_loader
def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    def __init__(self, sizes, learning_rate=0.01):
        """
        Initialize the Neural Network with random weights, biases, and given hyperparameters.
        Args:
            sizes: List containing the sizes of each layer [n_input, n_neurons_1, n_neurons_2, n_output, n_epochs].
            learning_rate: The learning rate for gradient descent (default: 0.01).
        """
        n_input = sizes[0]
        n_neurons_1 = sizes[1]
        n_neurons_2 = sizes[2]
        n_output = sizes[3]
        n_epochs = sizes[4]

        # Number of training epochs
        self.epochs = n_epochs

        # Initialize weights and biases for each layer
        self.w1 = np.random.randn(n_neurons_1, n_input)
        self.w2 = np.random.randn(n_neurons_2, n_neurons_1)
        self.w3 = np.random.randn(n_output, n_neurons_2)
        self.b1 = np.zeros((n_neurons_1, 1))
        self.b2 = np.zeros((n_neurons_2, 1))
        self.b3 = np.zeros((n_output, 1))

        # Learning rate for the network
        self.learning_rate = learning_rate

        # Track loss for each epoch
        self.Loss = np.zeros(n_epochs)

    def Loss_L2(self, y, y_hat):
        """
        Compute the L2 loss (squared error) between the predicted output and the actual output.
        Args:
            y: Actual output (ground truth).
            y_hat: Predicted output.

        Returns:
            The L2 loss value.
        """
        return 0.5 * np.sum((y_hat - y) ** 2)

    def Train_2Layers(self, training_data):
        """
        Train the neural network with two hidden layers using forward and backward propagation.
        Args:
            training_data: A list of tuples (x, y) where x is the input and y is the target output.
        """
        for epoch in range(self.epochs):
            for x, y in training_data:
                # Forward pass
                z1 = np.dot(self.w1, x) + self.b1
                a1 = sigmoid(z1)

                z2 = np.dot(self.w2, a1) + self.b2
                a2 = sigmoid(z2)

                z3 = np.dot(self.w3, a2) + self.b3
                y_hat = sigmoid(z3)

                # Calculate the Loss
                self.Loss[epoch] += self.Loss_L2(y, y_hat) / len(training_data)

                # Backpropagation
                # Output layer gradients
                delta3 = (y_hat - y) * sigmoid_derivative(z3)
                dLdW3 = np.dot(delta3, a2.T)
                dLdB3 = delta3

                # Hidden Layer 2 gradients
                delta2 = np.dot(self.w3.T, delta3) * sigmoid_derivative(z2)
                dLdW2 = np.dot(delta2, a1.T)
                dLdB2 = delta2

                # Hidden Layer 1 gradients
                delta1 = np.dot(self.w2.T, delta2) * sigmoid_derivative(z1)
                dLdW1 = np.dot(delta1, x.T)
                dLdB1 = delta1

                # Gradient descent updates
                self.w3 -= self.learning_rate * dLdW3
                self.w2 -= self.learning_rate * dLdW2
                self.w1 -= self.learning_rate * dLdW1

                self.b3 -= self.learning_rate * np.sum(dLdB3, axis=1, keepdims=True)
                self.b2 -= self.learning_rate * np.sum(dLdB2, axis=1, keepdims=True)
                self.b1 -= self.learning_rate * np.sum(dLdB1, axis=1, keepdims=True)

    def Y_hat(self, x):
        """
        Compute the forward pass for a given input to predict the output.
        Args:
            x: Input data.

        Returns:
            Predicted output after forward pass.
        """
        z1 = np.dot(self.w1, x) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = sigmoid(z2)
        z3 = np.dot(self.w3, a2) + self.b3
        y_hat = sigmoid(z3)
        return y_hat

    def Validate_2Layer(self, validation_data):
        """
        Validate the trained model on a validation dataset.
        Args:
            validation_data: A list of tuples (x, y) where x is the input and y is the target output.

        Returns:
            total_loss: Total L2 loss on the validation dataset.
            accuracy: Accuracy of the model on the validation dataset.
        """
        total_loss = 0
        accuracy = 0
        for x, y in validation_data:
            y_hat = self.Y_hat(x)
            total_loss += self.Loss_L2(y, y_hat) / len(validation_data)
            if np.argmax(y_hat) == np.argmax(y):
                accuracy += 1 / len(validation_data)
        return total_loss, accuracy

    def Test_2Layer(self, test_data):
        """
        Test the trained model on a test dataset.
        Args:
            test_data: A list of tuples (x, y) where x is the input and y is the target output.

        Returns:
            Accuracy of the model on the test dataset.
        """
        count = 0
        for x, y in test_data:
            y_hat = self.Y_hat(x)
            if np.argmax(y_hat) == np.argmax(y):
                count += 1 / len(test_data)
        return count