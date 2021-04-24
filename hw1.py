import math

import numpy as np
from random import seed
from random import random
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

class NeuralNetwork:
    def __init__(self):

        # Initialize the network
        network = list()

        # Use Xavier Initialization
        embedding_layer_w1 = np.random.rand(250, 16) / math.sqrt(250)
        network.append(embedding_layer_w1)

        hidden_layer_w2 = np.random.rand(48, 128) / math.sqrt(48)
        network.append(hidden_layer_w2)

        hidden_layer_b1 = np.zeros(128)
        network.append(hidden_layer_b1)

        output_layer_w3 = np.random.rand(128, 250) / math.sqrt(128)
        network.append(output_layer_w3)

        output_layer_b2 = np.zeros(250)
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
        f_h = sigmoid(h)  # OK(?) -> 1x128

        # 4. Output layer
        o = f_h @ self.network[3]  # OK -> 1x250
        o = o + self.network[4]  # OK -> 1x250

        # 5. Softmax
        s_o = softmax(o)  # OK(?) -> 1x250

        # 6. Cross Entropy Loss
        loss = cross_entropy_loss(y, s_o)

        # Optional: Get one hot encoding of prediction
        guess = np.zeros_like(s_o)
        max = s_o[0][0]
        max_index = 0

        for i in range(250):
            if s_o[0][i] > max:
                max = s_o[0][i]
                max_index = i
        guess[0][max_index] = 1

        return loss, guess, f_h, s_o, e

    # Forward propagate batch:
    def forward_propagation_batch(self, batch_size, input_batch, target_batch):

        losses = []  # batch size times loss
        guesses = []
        f_h_batch = [] # hidden layer
        s_o_batch =[] # output layer
        e_batch = [] # embeddings for batch

        for i in range(batch_size):
            loss, guess, f_h, s_o, e = self.forward_propagation(input_batch[i], target_batch[i])
            losses.append(loss)
            guesses.append(guess)
            f_h_batch.append(f_h)
            s_o_batch.append(s_o)
            e_batch.append(e)

        average_loss = np.average(losses)
        total_loss = np.sum(losses)

        return average_loss, total_loss, guesses, f_h_batch, s_o_batch, e_batch

    # Calculate gradients:
    def backprop(self, input_batch, target_batch, f_h_batch, s_o_batch, e_batch):

        # Return dw3, db2, dw2, db1, dw1

        so = np.array(s_o_batch)
        so = np.squeeze(so)  # so -> nx250
        y = np.array(target_batch)  # y -> nx250
        fh = np.array(f_h_batch)
        fh = np.squeeze(fh) # fh -> nx128
        e = np.array(e_batch)
        e = np.squeeze(e) # e -> nx48

        t = (so - y)
        dw3 = np.dot(fh.T, t) # dw3 -> 128x250

        db2 = t
        db2 = db2.sum(axis=0) # db2 -> 1x250 olmalı ama n x 250, avg aldım, emin değilim.

        w3 = self.network[3]
        a = np.dot(t,w3.T) # a -> nx128
        a = dsigmoid(a)
        dw2 = np.dot(e.T,a)

        db1 = a
        db1 = db1.sum(axis=0)  # db1 -> 1x128 olmalı ama n x 128, avg aldım, emin değilim.

        w2 = self.network[1]
        w2_split = np.split(w2, 3)
        w21 = w2_split[0]
        w22 = w2_split[1]
        w23 = w2_split[2] # All (w21 w22 w23) -> 16x128

        y1 = np.dot(a, w21.T) # nx16
        y2 = np.dot(a, w22.T) # nx16
        y3 = np.dot(a, w23.T) # nx16

        batch_size = len(input_batch)
        x1 = []
        x2 = []
        x3 = []

        for i in range(batch_size):
            x1.append(input_batch[i][0])
            x2.append(input_batch[i][1])
            x3.append(input_batch[i][2])

        x1 = np.array(x1)
        x1 = np.squeeze(x1) # x1 -> nx250
        x2 = np.array(x2)
        x2 = np.squeeze(x2)  # x2 -> nx250
        x3 = np.array(x3)
        x3 = np.squeeze(x3)  # x3 -> nx250

        r1 = np.dot(x1.T, y1) # 250x16
        r2 = np.dot(x2.T, y2) # 250x16
        r3 = np.dot(x3.T, y3) # 250x16

        dw1 = r1+r2+r3 # dw1 -> 250x16

        return dw3, db2, dw2, db1, dw1

    def update(self, dw3, db2, dw2, db1, dw1, learning_rate):
        # network[0] = w1 -> (250,16)
        # network[1] = w2 -> (48,128)
        # network[2] = b1 -> (1, 128)
        # network[3] = w3 -> (128,250)
        # network[4] = b2 -> (1, 250)

        self.network[0] -= learning_rate * dw1
        self.network[1] -= learning_rate * dw2
        self.network[2] -= learning_rate * db1
        self.network[3] -= learning_rate * dw3
        self.network[4] -= learning_rate * db2

    def train(self, converted_train_inputs, converted_train_targets):

        learning_rate = 0.001
        batch_size = 25
        epochs = 20

        train_length = len(converted_train_inputs)
        total_batch_number = train_length / batch_size

        epoch_total_losses = []
        epoch_average_losses = []
        epoch_accuracies = []

        # Training loop

        for e in range(epochs):
            #if e >= 15:
            #    learning_rate = learning_rate/10
            #if e >= 15:
            #    learning_rate = learning_rate/15
            batch_total_losses = []
            batch_average_losses = []
            batch_accuracies = []

            for b in range(1200):
                input_batch = converted_train_inputs[b * batch_size:b * batch_size + batch_size]
                target_batch = converted_train_targets[b * batch_size:b * batch_size + batch_size]

                average_loss, total_loss, guesses, f_h_batch, s_o_batch, e_batch = self.forward_propagation_batch(
                    batch_size, input_batch, target_batch)
                dw3, db2, dw2, db1, dw1 = self.backprop(input_batch, target_batch, f_h_batch, s_o_batch, e_batch)
                self.update(dw3, db2, dw2, db1, dw1, learning_rate)

                batch_accuracy = self.calculate_training_accuracy(guesses, target_batch)

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

        # Save embeddings for tSNE
        self.return_and_save_embeddings()

    def calculate_training_accuracy(self, guesses, train_targets):
        batch_size = len(train_targets)
        true_count = 0

        guesses = np.array(guesses)
        guesses = np.squeeze(guesses)

        for i in range(batch_size):
            guess_index = convert_one_hot_to_index(guesses[i])
            train_index = convert_one_hot_to_index(train_targets[i])
            #print(guess_index)
            #print(train_index)
            if guess_index == train_index:
                true_count+=1

        accuracy = true_count/batch_size
        return accuracy

    def return_and_save_embeddings(self):
        np.save('embeddings.npy', self.network[0])
        return self.network[0]

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

def tsne_visualization():

    embedding = np.load('embeddings.npy')
    X = np.array(embedding[0:3])

    X = np.array([[-1.89947302, -1.71278179, -2.01633414, -1.89693931],
                  [-9.18220909, -8.27972830, -9.74712541, -9.16996093],
                  [-2.36182356, -2.12968984, -2.50712985, -2.35867313]])
    print(np.round(X, decimals=1))

    results = TSNE(n_components=2).fit_transform(X)
    #tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    #plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'])
    plt.show()

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

    network.train(converted_train_inputs,converted_train_targets)
    #tsne_visualization()

if __name__ == '__main__':
    main()