from src.NNFromScratch.logging import logger

from neuralNetwork import neuralNetwork
import numpy as np
import matplotlib.pyplot

import scipy.ndimage
import scipy.misc
import glob

logger.info("Testing logger")


def main():
    # number of input, hidden and output layers
    input_nodes = 784
    hidden_nodes = 200  # <> this is just based on experimenation, this is where the learning happens
    output_nodes = 10

    # learning rate
    learning_rate = 0.1  # <> does learning rate affect training time

    # create instance of neural network
    nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # reading the sample train file & loading into list
    train_data_file = open("data/mnist_train_100.csv", "r")
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    # print the length of dataset
    # print(len(train_data_list))
    # print the first record
    # print(train_data_list[0])

    # converting and looking at one of the image
    all_values = train_data_list[0].split(sep=",")
    image_array = np.array(all_values[1:], dtype=np.uint8).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
    matplotlib.pyplot.savefig("output/samplePlot2.png")

    # * train the neural network

    epochs = 7

    for _ in range(
        epochs
    ):  # <> use epochs on training data, this is quite interesting, the model adjusts its weights in a way that improves performance across all images, not just one at a time

        for record in train_data_list:
            # split the record
            all_values = record.split(",")
            # scale and shift the input
            inputs = (np.array(all_values[1:], dtype=np.uint8) / 255.0 * 0.99) + 0.01
            # create target output (all 0.01 except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # set the corresponding label to 0.99
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)

            ## create rotated variations
            # rotated anticlockwise by x degrees
            inputs_plusx_img = scipy.ndimage.interpolation.rotate(
                inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False
            )
            nn.train(inputs_plusx_img.reshape(784), targets)
            # rotated clockwise by x degrees
            inputs_minusx_img = scipy.ndimage.interpolation.rotate(
                inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False
            )
            nn.train(inputs_minusx_img.reshape(784), targets)

    # load the test dataset into memory
    test_data_file = open("data/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # get the first test record
    all_values = test_data_list[0].split(",")
    print(all_values[0])

    # check the image
    image_array = np.array(all_values[1:], dtype="uint8").reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
    matplotlib.pyplot.savefig("output/sampleImage2.png")

    # * test the neural network

    print(
        nn.query(
            input_list=(np.array(all_values[1:], dtype="uint8") / 255.0 * 0.99 + 0.01)
        )
    )

    # score card to check network performance
    score_card = []

    for record in test_data_list:

        all_values = record.split(",")
        true_label = int(all_values[0])
        # print(true_label, "true label")

        inputs = (np.array(all_values[1:], dtype="uint8") / 255.0 * 0.99) + 0.01
        outputs = nn.query(input_list=inputs)

        predicted_label = np.argmax(outputs)
        # print(predicted_label, "network answer")

        if true_label == predicted_label:
            score_card.append(1)

        else:
            score_card.append(0)

    # print(score_card)

    score_card_array = np.array(
        score_card
    )  # <> list & array are different, that's why & how you conver to numpy array
    print("performance = ", score_card_array.sum() / score_card_array.size)


if __name__ == "__main__":
    main()
