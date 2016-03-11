# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 02:05:24 2016

@author: tunoat
"""
from __future__ import absolute_import
from __future__ import print_function
sentences = [
 'fred loves pizza',
 'marie loves noodle',
 'marie loves fred',
 'fred loves marie',
 'fred hates noodle',
 'marie hates pizza',
 'marie loves noodle very much',
 'fred loves pizza very much',
 'marie hates pizza very much',
 'fred hates noodle very much',
 'fred always eat a pizza',
 'marie always eat a noodle',
 'pizza is expensive',
 'noodle is cheap',
 'pizza is italian food',
 'noodle is chinese food',
 'chinese eat noodle',
 'italian eat pizza',
 'pizza eat with sauce',
 'noodle eat with soup'
 ]
 
#==============================================================================
# sentences = [
# 'fred loves pizza',
# 'marie loves noodle',
# 'fred hates noodle',
# 'marie hates pizza'
# ]
# 
#==============================================================================
# 1. create dictionary
i = 0
vocab = dict()
for sentence in sentences:
    wordList = sentence.split()
    for word in wordList:
        if not word in vocab:
            vocab[word] = i
            i+= 1

# 2. create the matrix represent of word in sentence
import numpy as np
def n_gram_training_set(windowSize):
    eyeMat = np.eye(len(vocab))
    n_gram_list = []
    X = []
    Y = []
    windowSize += 1
    for sentence in sentences:
        wordList = sentence.split()
        idx = 0
        for word in wordList:
            for left in range(windowSize):
                if (idx - left != idx) and (idx - left > -1):
                    X.append(eyeMat[vocab[wordList[idx]]])
                    Y.append(vocab[wordList[idx-left]])
                    wordPair = (wordList[idx], wordList[idx-left])
                    n_gram_list.append(wordPair)
            for right in range(windowSize):
                if (idx + right != idx) and  (idx + right < len(wordList)):
                    X.append(eyeMat[vocab[wordList[idx]]])
                    Y.append(vocab[wordList[idx+right]])
                    wordPair = (wordList[idx], wordList[idx+right])
                    n_gram_list.append(wordPair)
            idx+=1
    return n_gram_list, X, Y

training_set, X , Y = n_gram_training_set(2)
X = np.array(X)
Y = np.array(Y)

num_examples = len(X) # training set size
nn_input_dim = len(vocab) # input layer dimensionality
nn_output_dim = len(vocab) # output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# helper function softmax
def softmax(input_x,model):
    #W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    W1, W2 = model['W1'],model['W2']
    # Forward propagation
    #z1 = X.dot(W1) + b1
    z1 = input_x.dot(W1)
    a1 = np.tanh(z1)
    #z2 = a1.dot(W2) + b2
    z2 = a1.dot(W2)
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs
    
# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    #W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    W1, W2 = model['W1'],model['W2']
    probs = softmax(X, model)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), Y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    return np.argmax(softmax(x,model), axis=1)
    
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    #nn_input_dim, nn_output_dim = number of words in vocab
    #nn_hdim = hidden node as a vector of each word    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    #b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    #b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
 
        # Forward propagation
        #z1 = X.dot(W1) + b1
        z1 = X.dot(W1)
        a1 = np.tanh(z1)
        #z2 = a1.dot(W2) + b2
        z2 = a1.dot(W2)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), Y] -= 1
        dW2 = (a1.T).dot(delta3)
        #db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        #db1 = np.sum(delta2, axis=0)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        #b1 += -epsilon * db1
        W2 += -epsilon * dW2
        #b2 += -epsilon * db2
         
        # Assign new parameters to the model
        #model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        model = { 'W1': W1, 'W2': W2}
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print("Loss after iteration %i: %f" %(i, calculate_loss(model)))
     
    return model

# Build a model with a 3-dimensional hidden layer
model = build_model(5, print_loss=True)
# Plot the decision boundary

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(model['W1'])
X_new = pca.transform(model['W1'])

# 11. Plotting to vector space using PCA
for word in vocab:  
    plt.scatter(X_new[vocab[word]][0],
                X_new[vocab[word]][1])
    plt.annotate(word, (X_new[vocab[word]][0],
                        X_new[vocab[word]][1]))

import operator

def cosine_similarity(positive=None, negative=None):  
    ranking = dict()
    ivector = np.zeros((1, model['W1'].shape[1]))
    wordList = []
    if not positive is None:
        for word in positive:
            ivector += model['W1'][vocab[word]]
            wordList.append(word)
    if not negative is None:
        for word in negative:
            ivector -= model['W1'][vocab[word]]
            wordList.append(word)
    for word in vocab:
        if not word in wordList:
            tvector = model['W1'][vocab[word]]    
            cosine_similarity = (np.dot(ivector,tvector.T) / 
                (np.linalg.norm(ivector)*np.linalg.norm(tvector)))
            ranking[word] = cosine_similarity
    ranking = sorted(ranking.items(), key=operator.itemgetter(1)
        , reverse=True) # sorted from value, used 0 for key
    return ranking


print(cosine_similarity(positive=['fred','noodle'],negative=['pizza']))
print(cosine_similarity(positive=['marie','pizza'],negative=['noodle']))

print(cosine_similarity(positive=['noodle','sauce'],negative=['soup']))
print(cosine_similarity(positive=['pizza','soup'],negative=['sauce']))
