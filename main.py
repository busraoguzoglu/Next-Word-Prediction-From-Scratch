import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.utils import shuffle
import pickle
from network import NeuralNetwork


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

def tsne_visualization(model):

    #embedding = np.load('embeddings.npy')
    embedding = model.network[0]

    X = np.array(embedding)
    print(X[0:3])

    X_rounded = np.round(X, decimals=1)

    results = TSNE(n_components=2).fit_transform(X_rounded)
    tsne_results = pd.DataFrame(results, columns=['tsne1', 'tsne2'])

    words = np.load('data/vocab.npy')

    for i in range(250):
        plt.scatter(tsne_results['tsne1'][i], tsne_results['tsne2'][i], marker='x', color='red')
        plt.text(tsne_results['tsne1'][i], tsne_results['tsne2'][i], words[i], fontsize=9)

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

    # Save the model as pickle
    #with open('model.pk', 'wb') as f:
    #    pickle.dump(network, f)
    #f.close()

if __name__ == '__main__':
    main()