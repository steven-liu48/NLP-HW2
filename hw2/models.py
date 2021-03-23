"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File

Steven Liu
xl2948
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class DenseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, weight):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.

        # Attributes needed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = weight

        # Layers
        # Embedding layer, initialized with the pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(weight)
        # Hidden layers
        self.dense1 = nn.Linear(self.input_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, 4)
        # Activation function
        self.relu = nn.ReLU()
        # Output layer
        self.softmax = nn.Softmax()

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class

        # 1) Put the words through an Embedding layer
        x = self.embedding(x).float()
        # 2) Take the sum of all word embeddings in a sentence
        x = torch.sum(x, 1)
        # 3) Feed the result into 2-layer feedforward network
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, weight):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.

        # Attributes needed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = weight

        # Layers
        # Embedding layer, initialized with the pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(weight)
        # RNN Layer
        self.rnn = nn.RNN(self.input_size, self.hidden_size, 2)
        # Linear layer for classification
        self.linear = nn.Linear(hidden_size, 4)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        # Get list of true sentence lengths
        len_list = []
        for vector in x:
            len_list.append(len(torch.nonzero(vector)))
        # Put the words through the embedding layer
        x = self.embedding(x).float()
        # Pack the sequence and feed into the 2-layer RNN
        x = rnn.pack_padded_sequence(x, len_list, batch_first=True, enforce_sorted=False)
        output, h_n = self.rnn(x)
        # Get the last output state and feed into the dense layer
        return self.linear(h_n[-1])


# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    def __init__(self):
        super(ExperimentalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        raise NotImplementedError
