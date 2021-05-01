import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import pickle
from network import NeuralNetwork

"""
Main Class
...

Loads necessary files,
Initializes the Network,
Runs the training loop,
Saves the model after training

"""


def convert_one_hot_to_index(one_hot_vector):
    index = 0
    for i in range(len(one_hot_vector)):
        if one_hot_vector[i] == 1:
            index = i
    return index


def convert_one_hot(word_index):
    one_hot_representation = np.zeros(250)
    one_hot_representation[word_index] = 1
    return one_hot_representation


def convert_one_hot_all_training(train_inputs, train_targets):
    # Convert train inputs into one hot representation
    converted_train_inputs = []
    converted_train_targets = []

    for i in range(len(train_inputs)):

        converted_row = []
        converted1 = convert_one_hot(train_inputs[i][0])
        converted2 = convert_one_hot(train_inputs[i][1])
        converted3 = convert_one_hot(train_inputs[i][2])
        converted_row.append(converted1)
        converted_row.append(converted2)
        converted_row.append(converted3)
        converted_train_inputs.append(converted_row)

        converted_target = convert_one_hot(train_targets[i])
        converted_train_targets.append(converted_target)

    return converted_train_inputs, converted_train_targets


def load_files():
    train_file = 'data/train_inputs.npy'
    train_file2 = 'data/train_targets.npy'
    test_file = 'data/test_inputs.npy'
    test_file2 = 'data/test_targets.npy'
    validation_file = 'data/valid_inputs.npy'
    validation_file2 = 'data/valid_targets.npy'
    vocab_file = 'data/vocab.npy'

    train_inputs = np.load(train_file)
    train_targets = np.load(train_file2)
    test_inputs = np.load(test_file)
    test_targets = np.load(test_file2)
    valid_inputs = np.load(validation_file)
    valid_targets = np.load(validation_file2)
    vocab = np.load(vocab_file)

    return train_inputs, train_targets, test_inputs, test_targets,valid_inputs, valid_targets, vocab


def train(network, converted_train_inputs, converted_train_targets):
    learning_rate = 0.001
    batch_size = 500  # 745 best (10 epoch)
    epochs = 50  # 5 best  5 epoch sonrası update olmuyor

    train_length = len(converted_train_inputs)
    total_batch_number = train_length / batch_size

    epoch_total_losses = []
    epoch_average_losses = []
    epoch_accuracies = []

    # Training loop

    for e in range(epochs):

        converted_train_inputs, converted_train_targets = shuffle(converted_train_inputs, converted_train_targets)

        if e >= 25:
            learning_rate = learning_rate / 10
        # if e >= 15:
        #    learning_rate = learning_rate/100
        batch_total_losses = []
        batch_average_losses = []
        batch_accuracies = []

        for b in range(int(total_batch_number)):
            input_batch = converted_train_inputs[b * batch_size:b * batch_size + batch_size]
            target_batch = converted_train_targets[b * batch_size:b * batch_size + batch_size]

            average_loss, total_loss, guesses, f_h_batch, s_o_batch, e_batch, o_batch, h_batch = network.forward_propagation_batch(
                batch_size, input_batch, target_batch)
            dw3, db2, dw2, db1, dw1 = network.backprop(input_batch, target_batch, f_h_batch, s_o_batch, e_batch,
                                                       o_batch, h_batch)
            network.update(dw3, db2, dw2, db1, dw1, learning_rate)

            batch_accuracy = network.calculate_training_accuracy(guesses, target_batch)

            batch_total_losses.append(total_loss)
            batch_average_losses.append(average_loss)
            batch_accuracies.append(batch_accuracy)

            if b % 10 == 0:
                print("\nAverage loss over batch:", np.round(average_loss, decimals=2))
                print("\nTotal loss over batch:", np.round(total_loss, decimals=2))
                print("\nAccuracy over batch:", np.round(batch_accuracy, decimals=2))

        batch_total_losses_avg = sum(batch_total_losses) / len(batch_total_losses)
        batch_average_losses_avg = sum(batch_average_losses) / len(batch_average_losses)
        batch_accuracies_avg = sum(batch_accuracies) / len(batch_accuracies)

        epoch_total_losses.append(batch_total_losses_avg)
        epoch_average_losses.append(batch_average_losses_avg)
        epoch_accuracies.append(batch_accuracies_avg)

    plt.plot(epoch_total_losses)
    plt.show()
    plt.plot(epoch_average_losses)
    plt.show()
    plt.plot(epoch_accuracies)
    plt.show()


def main():

    # Load Files
    train_inputs, train_targets, test_inputs, test_targets, valid_inputs, valid_targets, vocab = load_files()

    # Convert train inputs into one hot representation
    converted_train_inputs, converted_train_targets = convert_one_hot_all_training(train_inputs, train_targets)
    network = NeuralNetwork()
    train(network, converted_train_inputs,converted_train_targets)

    # Save the model as pickle
    #with open('model.pk', 'wb') as f:
    #    pickle.dump(network, f)
    #f.close()

if __name__ == '__main__':
    main()