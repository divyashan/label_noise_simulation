import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pdb
from sklearn import datasets
#import tensorflow as tf 
#from tensorflow.python import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.initializers import Zeros
#from keras.optimizers import SGD

from utils import plot_data, TestCallback

IMAGE_DIR = "./plots/"
N_EXAMPLE_OPTS = [100, 200, 1000]

# Plot styles
sns.set_style("whitegrid")
flat_ui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.hls_palette(8, l=.3, s=.8))
palette = sns.hls_palette(8, l=.3, s=.8)

random.seed(1000)
N_EPOCHS = 50

def gen_class_0():
    mean = [1, 0]
    cov = [[1, 0], [0, 10]]
    return np.random.multivariate_normal(mean, cov )

def gen_class_1():
    mean = [5, 0]
    cov = [[1, 0], [0, 1]]
    return np.random.multivariate_normal(mean, cov )


def gen_data(p, n_examples=100):
    X = np.zeros((n_examples, 2))
    Y = np.zeros((n_examples))

    for i in range(n_examples):
        rand_num = np.random.rand()
        if rand_num < p:
            X[i] = gen_class_0()
            Y[i] = 0
        else:
            X[i] = gen_class_1()
            Y[i] = 1
    return X, Y

def gen_data_dist(data_params, n_examples=1000):
    if "clustering" in data_params:
        X, Y = datasets.make_blobs(n_samples=n_examples, 
                           n_features=2,
                           centers= data_params["clustering"]["centers"])
    else:
        p = data_params["class_split"]
        dist_1 = data_params["dist_1"]
        dist_0 = data_params["dist_0"]
        X = np.zeros((n_examples, 2))
        Y = np.zeros((n_examples))

        for i in range(n_examples):
            rand_num = np.random.rand()
            if rand_num < p:
                X[i] = dist_0.sample()
                Y[i] = 0
            else:
                X[i] = dist_1.sample()
                Y[i] = 1
    return X, Y


def add_noise_to_class(delta, label, y):
# Assumption: labels are binary, 0 and 1
    y_tilde = np.copy(y)
    for i in range(len(y)):
        if y[i] == label:
            y_tilde[i] = abs(label-1) if np.random.rand() < delta else label 
    return y_tilde

def gen_corrupted_labels(delta_0, delta_1, y):
    
    # dictionaries of indices mapped to whether or not the value 
    # should be flipped
    flip_zero = {i: True if np.random.rand() < delta_0 else False for i, x in enumerate(y) if x == 0}
    flip_one = {i: True if np.random.rand() < delta_1 else False for i, x in enumerate(y) if x == 1}
    
    y_tilde = np.copy(y)
    for index in flip_zero:
        if flip_zero[index]:
            y_tilde[index] = 1
            
    for index in flip_one:
        if flip_one[index]:
            y_tilde[index] = 0
    return y_tilde

def build_model():
    model = Sequential()
    model.add(Dense(1, kernel_initializer= Zeros(), input_shape=(2,)))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def run_test(p = .5, delta_0=0, delta_1=None, n_examples=100, n_noise_runs=5):
    # p: P(Y = 0)
    # delta_0: P(Y' = 1 | Y = 0)
    # delta_1: P(Y' = 0 | Y = 1)
    # n_examples: Batch size
    # n_noise_runs: Number of noisy minimizations to simulate. Each noise model is 

    if not delta_1:
        delta_1 = delta_0

    N_TRAIN = int(.5*n_examples)
    N_VAL = int(.25*n_examples)
    N_TEST = int(.25*n_examples)

    X, y = gen_data(p, n_examples=n_examples)

    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
    X_val, y_val = X[N_TRAIN:N_TRAIN+N_VAL], y[N_TRAIN:N_TRAIN+N_VAL]
    X_test, y_test = X[N_TRAIN+N_VAL:], y[N_TRAIN+N_VAL:]

    y_train_tildes = [gen_corrupted_labels(delta_0, delta_1, y_train) for i in range(n_noise_runs)]


    true_params = []
    noisy_params = []
    true_model = build_model()
    noise_models = [build_model() for i in range(n_noise_runs)]

    for i in range(N_EPOCHS+1):
        curr_true_weights = true_model.get_weights()[0]
        curr_noisy_weights = [noise_model.get_weights()[0] for noise_model in noise_models]
        curr_noisy_weights = np.squeeze(np.array(curr_noisy_weights), 2).T

    true_params.append(curr_true_weights)
    noisy_params.append(curr_noisy_weights)


    true_model.fit(X_train, y_train, batch_size=N_TRAIN, epochs=1, 
                validation_data=(X_val, y_val), 
                verbose=0)
    for noise_model, y_train_tilde in zip(noise_models, y_train_tildes):
        noise_model.fit(X_train, y_train_tilde, batch_size=N_TRAIN, epochs=1, 
                validation_data=(X_val, y_val), 
                verbose=0)


    true_p = np.hstack(true_params).T
    noisy_p = np.array(noisy_params)
    plt.plot(true_p[:,0], true_p[:,1], marker='o')
    [plt.plot(noisy_p[:,0,i], noisy_p[:,1,i], marker='o', color=palette[1], alpha=.5) for i in range(n_noise_runs)]
    plt.xlabel("theta[0]")
    plt.ylabel("theta[1]")
    plt.title("Delta_0 = " + str(delta_0) + " // Delta_1 = " + str(delta_1) + " // P = " + str(p) + " // N = " + str(n_examples))
    plt.savefig(IMAGE_DIR + str(p) + "_" + str(delta_0) + "_" + str(delta_1) + "_" + str(n_examples) + ".png")
plt.clf()



