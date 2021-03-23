"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Main File

Steven Liu
xl2948
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    # TODO: Given a model, data, and loss function, you should do the following:
    # TODO: 1) Loop through the whole train dataset performing batch optimization with the optimizer of your choice,
    # TODO: updating the model parameters with each batch (we suggest you use torch.optim.Adam to start);
    # TODO: 2) Each time you reach the end of the train dataset (one "epoch"), calculate the loss on the whole dev set;
    # TODO and 3) stop training and return the model once the development loss stops improving (called early stopping).
    # TODO: Make sure to print the dev set loss each epoch to stdout.

    last_loss = float("inf")
    epoch = 1
    while True:
        # Perform batch optimization
        for local_X, local_y in train_generator:
            model.zero_grad()
            pred = model(local_X)
            loss = loss_fn(pred, local_y)  # Calculate loss
            loss.backward()  # Back propagation
            optimizer.step()

        # Calculate loss on the whole dev set
        dev_loss = 0
        for local_X, local_y in dev_generator:
            dev_loss += loss_fn(model(local_X), local_y).item()

        # Print dev set loss
        print("Epoch", epoch, "- loss:", dev_loss)
        epoch += 1

        # Early stopping
        if last_loss < dev_loss:
            break
        else:
            last_loss = dev_loss

    return model


# extension-grading
# Extension 2: Train function with learning rate scheduler
def train_model_with_scheduler(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    # TODO: Given a model, data, and loss function, you should do the following:
    # TODO: 1) Loop through the whole train dataset performing batch optimization with the optimizer of your choice,
    # TODO: updating the model parameters with each batch (we suggest you use torch.optim.Adam to start);
    # TODO: 2) Each time you reach the end of the train dataset (one "epoch"), calculate the loss on the whole dev set;
    # TODO and 3) stop training and return the model once the development loss stops improving (called early stopping).
    # TODO: Make sure to print the dev set loss each epoch to stdout.

    last_loss = float("inf")
    epoch = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)  # extension-grading

    while True:
        # Perform batch optimization & upload model parameters
        for local_X, local_y in train_generator:
            model.zero_grad()
            pred = model(local_X)
            loss = loss_fn(pred, local_y)
            loss.backward()
            optimizer.step()

        scheduler.step()  # extension-grading

        # Calculate loss on the whole dev set
        dev_loss = 0
        for local_X, local_y in dev_generator:
            dev_loss += loss_fn(model(local_X), local_y)

        # Print dev set loss
        print("Epoch", epoch, "- loss:", dev_loss.item())
        epoch += 1

        # Early stopping
        if last_loss < dev_loss:
            break
        else:
            last_loss = dev_loss

    return model


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss.item())
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                      BATCH_SIZE,
                                                                                                      EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    # TODO: for each of the two models, you should 1) create it,
    # TODO 2) run train_model() to train it, and
    # TODO: 3) run test_model() on the result

    if args.model == "dense":
        print("Running Dense Network...")
        # Create Dense Network
        hidden_size = 100
        dense_network_model = models.DenseNetwork(EMBEDDING_DIM, hidden_size, embeddings)
        # Train Dense Network
        optimizer = optim.Adam(dense_network_model.parameters())
        trained_dense_network = train_model(dense_network_model, loss_fn, optimizer, train_generator, dev_generator)
        # Test Dense Network
        test_model(trained_dense_network, loss_fn, test_generator)
        '''
        [Example Running Result]
        Running Dense Network...
        Epoch 1 - loss: 26.624701619148254
        Epoch 2 - loss: 26.074581384658813
        Epoch 3 - loss: 25.847457885742188
        Epoch 4 - loss: 25.71996521949768
        Epoch 5 - loss: 25.60013437271118
        Epoch 6 - loss: 25.544367909431458
        Epoch 7 - loss: 25.474766969680786
        Epoch 8 - loss: 25.463510870933533
        Epoch 9 - loss: 25.4806067943573
        Test loss: 
        25.759105682373047
        F-score: 
        0.4385805866922266
        '''

    elif args.model == "RNN":
        print("Running RNN...")
        # Create Recurrent Network
        hidden_size = 100
        RNN_model = models.RecurrentNetwork(EMBEDDING_DIM, hidden_size, embeddings)
        # Train Dense Network
        optimizer = optim.Adam(RNN_model.parameters())
        trained_RNN = train_model(RNN_model, loss_fn, optimizer, train_generator, dev_generator)
        # Test Dense Network
        test_model(trained_RNN, loss_fn, test_generator)
        '''
        [Example Running Result]
        Running RNN...
        Epoch 1 - loss: 26.202155232429504
        Epoch 2 - loss: 26.0027813911438
        Epoch 3 - loss: 25.59765136241913
        Epoch 4 - loss: 25.593361020088196
        Epoch 5 - loss: 25.344910621643066
        Epoch 6 - loss: 25.324756622314453
        Epoch 7 - loss: 25.221933603286743
        Epoch 8 - loss: 25.211925745010376
        Epoch 9 - loss: 25.232311010360718
        Test loss: 
        25.492740631103516
        F-score: 
        0.4485606102904638
        '''

    # extension-grading
    # Extension 1: Changes to the preprocessing of the data (Tweet Tokenizer in NLTK)
    # Note that this extension re-vectorizes the data with the tweet tokenizer, so before you run other models,
    # please vectorize again using the default vectorizer. The code changes are made in utils.py.
    elif args.model == "extension1":
        print("Running Extension1: Tweet Tokenizer...")
        # Re-vectorize with tweet tokenizer
        train, dev, test = utils.get_data(DATA_FN)
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data_extension(train, dev, test, BATCH_SIZE, EMBEDDING_DIM, tweet_tokenizer=True)
        # Create Dense Network
        hidden_size = 100
        dense_network_model = models.DenseNetwork(EMBEDDING_DIM, hidden_size, embeddings)
        # Train Dense Network
        optimizer = optim.Adam(dense_network_model.parameters())
        trained_dense_network = train_model(dense_network_model, loss_fn, optimizer, train_generator, dev_generator)
        # Test Dense Network
        test_model(trained_dense_network, loss_fn, test_generator)
        '''
        [Example Running Result]
        Epoch 1 - loss: 26.779205203056335
        Epoch 2 - loss: 26.27608859539032
        Epoch 3 - loss: 26.121569991111755
        Epoch 4 - loss: 25.98335337638855
        Epoch 5 - loss: 25.886319518089294
        Epoch 6 - loss: 25.828333616256714
        Epoch 7 - loss: 25.81446599960327
        Epoch 8 - loss: 25.874130845069885
        Test loss: 
        25.889612197875977
        F-score: 
        0.44295003530045585
        '''

    # extension-grading
    # Extension 2: Different strategies for optimization and training (adding a learning rate scheduler)
    # The code changes are made above (Line 92 - 140).
    elif args.model == "extension2":
        print("Running Extension2: Learning Rate Scheduler...")
        # Create Dense Network
        hidden_size = 100
        dense_network_model = models.DenseNetwork(EMBEDDING_DIM, hidden_size, embeddings)
        # Train Dense Network using train_model_with_scheduler()
        optimizer = optim.Adam(dense_network_model.parameters())
        trained_dense_network = train_model_with_scheduler(dense_network_model, loss_fn, optimizer, train_generator, dev_generator)  # extension-grading
        # Test Dense Network
        test_model(trained_dense_network, loss_fn, test_generator)
        '''
        [Example Running Result]
        Epoch 1 - loss: 26.716489791870117
        Epoch 2 - loss: 26.059986114501953
        Epoch 3 - loss: 25.806568145751953
        Epoch 4 - loss: 25.610559463500977
        Epoch 5 - loss: 25.503353118896484
        Epoch 6 - loss: 25.4748592376709
        Epoch 7 - loss: 25.43999481201172
        Epoch 8 - loss: 25.408884048461914
        Epoch 9 - loss: 25.36359214782715
        Epoch 10 - loss: 25.328798294067383
        Epoch 11 - loss: 25.32452964782715
        Epoch 12 - loss: 25.325151443481445
        Test loss: 
        25.702045440673828
        F-score: 
        0.46330625478767345
        '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
