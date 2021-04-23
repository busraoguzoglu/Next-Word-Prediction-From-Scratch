import numpy as np
from random import seed
from random import random

class NeuralNetwork:
    def __init__(self):

        # Initialize the network
        network = list()
        embedding_layer_w1 = np.random.rand(250, 16)
        network.append(embedding_layer_w1)
        hidden_layer_w2 = np.random.rand(48, 128)
        network.append(hidden_layer_w2)
        hidden_layer_b1 = np.random.rand(1, 128)
        network.append(hidden_layer_b1)
        output_layer_w3 = np.random.rand(128, 250)
        network.append(output_layer_w3)
        output_layer_b2 = np.random.rand(1, 250)
        network.append(output_layer_b2)

        self.network = network

    # Forward propagate one row:
    def forward_propagation(self, row, y):

        # row = [1x250, 1x250, 1x250] one input -> x1,x2,x3
        # y is real y for that row

        # 1. Embedding Layer
        e1 = row[0] @ self.network[0]  # OK -> 1x16
        e2 = row[1] @ self.network[0]  # OK -> 1x16
        e3 = row[2] @ self.network[0]  # OK -> 1x16

        e = np.concatenate([e1, e2, e3])
        e = np.reshape(e, (-1, 48))  # OK -> 1x48

        # 2. Hidden Layer
        h = e @ self.network[1]
        h = np.reshape(h, (-1, 128))  # OK -> 1x128
        h = h + self.network[2]  # OK -> 1x128

        # 3. Sigmoid Activation
        self.f_h = sigmoid(h)  # OK(?) -> 1x128

        # 4. Output layer
        o = self.f_h @ self.network[3]  # OK -> 1x250
        o = o + self.network[4]  # OK -> 1x250

        # 5. Softmax
        self.s_o = softmax(o)  # OK(?) -> 1x250

        # 6. Cross Entropy Loss
        loss = cross_entropy_loss(y, self.s_o)

        # Optional: Get one hot encoding of prediction
        guess = np.zeros_like(self.s_o)
        max = self.s_o[0][0]
        max_index = 0

        for i in range(250):
            if self.s_o[0][i] > max:
                max = self.s_o[0][i]
                max_index = i
        guess[0][max_index] = 1

        return loss, guess

    # Forward propagate batch:
    def forward_propagation_batch(self, batch_size, input_batch, target_batch):

        losses = []  # batch size times loss
        guesses = []

        for i in range(batch_size):
            loss, guess = self.forward_propagation(input_batch[i], target_batch[i])
            losses.append(loss)
            guesses.append(guess)

        average_loss = np.average(losses)
        total_loss = np.sum(losses)

        print("\nAverage loss over batch:", np.round(average_loss, decimals=2))
        print("\nTotal loss over batch:", np.round(total_loss, decimals=2))

        return average_loss, total_loss, guesses

    # Calculate gradients:
    def backprop(network, guesses, target_batch):

        # Return dw3, db2, dw2, db1, dw1

        so = np.array(guesses)
        so = np.squeeze(so)  # so -> nx250
        y = np.array(target_batch)  # y -> nx250

        return

def cross_entropy_loss(y, yHat):
   # ref: http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
  return -np.sum(y * np.log(yHat))

def softmax(x):
  # Numerically stable softmax based on
  # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
  b = x.max()
  y = np.exp(x - b)
  return y / y.sum()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x): # Derivative of sigmoid
  return sigmoid(x) * (1 - sigmoid(x))

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



def main():

    # Load Files
    train_inputs, train_targets, test_inputs, test_targets, valid_inputs, valid_targets, vocab = load_files()

    # Convert train inputs into one hot representation
    converted_train_inputs, converted_train_targets = convert_one_hot_all_training(train_inputs, train_targets)

    # Define the network:
    # network[0] = w1 -> (250,16)
    # network[1] = w2 -> (48,128)
    # network[2] = b1 -> (1, 128)
    # network[3] = w3 -> (128,250)
    # network[4] = b2 -> (1, 250)
    network = NeuralNetwork()

    # Example batch size = 5
    input_batch = converted_train_inputs[0:50]
    target_batch = converted_train_targets[0:50]

    average_loss, total_loss, guesses = network.forward_propagation_batch(50, input_batch, target_batch)

    network.backprop(guesses, target_batch)

if __name__ == '__main__':
    main()