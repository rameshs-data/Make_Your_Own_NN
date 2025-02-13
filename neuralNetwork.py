import numpy
import scipy.special

# neural network class definition
class neuralNetwork:

    # initialzie the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """Function to initialize the neural network

        Args:
            inputnodes (_type_): _description_
            hiddennodes (_type_): _description_
            outputnodes (_type_): _description_
            learningrate (_type_): _description_
        """
        # set number of nodes in each input, hidden, output layers
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside array are w_i_j where link is from node i to node j in the next layer
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.hnodes, self.inodes))
        
        # learning rate
        self.lr = learningrate

        # activation function: sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

        pass  
    

    # train the neural network
    def train():
        pass


    # query the neural network
    def query(self, input_list):
        """This takes in a input list, performs a forward pass by calculating dot product and activation applied which acts as input to the next node

        Args:
            input_list (_type_): random input weight initially passed
        """
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layers
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into output layers
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signal emberging from final output layers
        final_outputs = self.activation_function(final_inputs)

        pass