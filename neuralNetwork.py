import numpy as np
import scipy.special


# neural network class definition
class neuralNetwork:

    # initialzie the neural network
    def __init__(
        self,
        inputnodes: list,
        hiddennodes: list,
        outputnodes: list,
        learningrate: float,
    ):
        """Function to initialize the neural network

        Args:
            inputnodes (list): A matrix of input weights
            hiddennodes (list): A matrix of hidden link weights
            outputnodes (list): A matrix of output weights
            learningrate (float): Controls the step of weight updates
        """
        # set number of nodes in each input, hidden, output layers
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside array are w_i_j where link is from node i to node j in the next layer
        self.wih = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = np.random.normal(
            0.0, pow(self.onodes, -0.5), (self.hnodes, self.inodes)
        )

        # learning rate
        self.lr = learningrate

        # activation function: sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, input_list: list, target_list: list) -> None:
        """This method helps in training the neural network, does a forward pass and backpropagation too

        Args:
            input_list (list): _description_
            target_list (list): _description_
        """
        # convert input list to 2d array: converts a list into a list of lists and transposes it so this gives 3x1 matrix if input_list = [1.2, 2.3, 3.2] for example
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate signals emerging out of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate signals emerging out of output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target - predicted)
        output_errors = targets - final_outputs

        # hidden layer error is the output errors split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the hidden weights for links between hidden and output layers
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1 - final_outputs)),
            np.transpose(hidden_outputs),
        )

        # update the weights for the links between input and hidden layers
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)),
            np.transpose(inputs),
        )

        pass

    # query the neural network
    def query(self, input_list: list) -> None:
        """This takes in a input list, performs a forward pass by calculating dot product and activation applied which acts as input to the next node

        Args:
            input_list (list): random input weight initially passed
        """
        # convert input list to 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate signals emerging from hidden layers
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into output layers
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signal emberging from final output layers
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
