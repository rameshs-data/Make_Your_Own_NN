from neuralNetwork import neuralNetwork

def main():
    # number of input, hidden and output layers
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    # learning rate
    learning_rate = 0.3

    # create instance of neural network
    nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(nn)
    
    # query using random inputs and output results
    output = nn.query(input_list=[1.0, 0.5, -1.5])
    print(output)

if __name__ == "__main__":
    main()