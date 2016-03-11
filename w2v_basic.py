# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 02:05:24 2016

@author: tunoat
"""
from __future__ import absolute_import
from __future__ import print_function
sentences = [
'fred loves pizza',
'marie loves noodle'
]

# 1. create dictionary
i = 0
vocab = dict()
for sentence in sentences:
    wordList = sentence.split()
    for word in wordList:
        if not word in vocab:
            vocab[word] = i
            i+= 1
            
            
def doc2bow(sentences, vocab):
    sentences_bow = list()
    for sentence in sentences:
        wordList = sentence.split()
        wordIdxVec = [vocab[word] for word in wordList]
        sentences_bow.append(wordIdxVec)
    return sentences_bow


#sentences_bow = np.array(doc2bow(sentences, vocab))


# 2. create the matrix represent of word in sentence
import numpy as np
hidden = 2
V, N = len(vocab), hidden
WI = (np.random.random((V,N)) - 0.5)/N # Weight input - Vocab x Hidden dimension
WO = (np.random.random((N,V)) - 0.5)/N # Weight output - Hidden x Vocab dimension 

# 3. Generate softmax function
def softmax(wordInput,wordOutput):
    topTerm = np.exp(np.dot(WI[vocab[wordInput]], WO.T[vocab[wordOutput]]))
    bottomTerm = sum(np.exp(np.dot(WI[vocab[wordInput]], WO.T[vocab[w]])) for w in vocab)
    prob = topTerm/bottomTerm
    return prob
    
def n_gram_training_set(windowSize):
    n_gram_list = []
    windowSize += 1
    for sentence in sentences:
        wordList = sentence.split()
        idx = 0
        for word in wordList:
            for left in range(windowSize):
                if (idx - left != idx) and (idx - left > -1):
                    wordPair = (wordList[idx], wordList[idx-left])
                    n_gram_list.append(wordPair)
            for right in range(windowSize):
                if (idx + right != idx) and  (idx + right < len(wordList)):
                    wordPair = (wordList[idx], wordList[idx+right])
                    n_gram_list.append(wordPair)
            idx+=1
    return n_gram_list

training_set = n_gram_training_set(2)

# 4. Gradient function
def gradient(probability, target, word_input_vector, learning_rate = 1):
    error = target - probability
    grad = learning_rate * error * word_input_vector 
    return grad

def maximize_probability(w_input, w_target, tolerance=0.01, learning_rate = 1):
    target_prob = 1.00 - tolerance
    p = 0
    while p < target_prob:
        for word in vocab:
            p_out_in = softmax(w_input, word)    
            t = 1 if word == w_target else 0
            grad = gradient(p_out_in, t, WI[vocab[w_input]])
            # 6. perform update hidden to output layer weight
            WO.T[vocab[word]] = (WO.T[vocab[word]] + grad)
        # 7. update the input to hidden weight
        WI[vocab[w_input]] = WI[vocab[w_input]] + learning_rate * WO.sum(1)
        p = softmax(w_input, w_target)
    


# 8. recompute to see if the prob is really increase
input_word = 'fred'
target_word = 'pizza'
print(softmax(input_word, target_word))
maximize_probability(input_word, target_word, 0.03)
print(softmax(input_word, target_word))

# 9. working with context of word
target_word = ['noodle','pizza']  
context = ['marie','fred']
h = (WI[vocab['marie']] + WI[vocab['fred']]) / 2
for word in vocab:  
    p = (np.exp(np.dot(WO.T[vocab[word]], h)) / 
         sum(np.exp(np.dot(WO.T[vocab[w]], h)) 
             for w in vocab))
    if word in target_word:  
        t = 1
    else:
        t=0
    error = t - p
    WO.T[vocab[word]] = (
        WO.T[vocab[word]] + 0.1 * 
        error * h)
for word in context:  
    WI[vocab[word]] = (
        WI[vocab[word]] + (1. / len(context)) *
        0.1 * WO.sum(1))

print(softmax('fred', 'pizza'))
print(softmax('marie', 'pizza'))
print(softmax('marie', 'noodle'))
print(softmax('fred', 'noodle'))

import operator
# 10. calculate cosine similarity
#'marie loves noodle'
#'fred loves pizza',
def cosine_similarity(positive=None, negative=None):  
    ranking = dict()
    ivector = np.zeros((1,WI.shape[1]))
    wordList = []
    if not positive is None:
        for word in positive:
            ivector += WI[vocab[word]]
            wordList.append(word)
    if not negative is None:
        for word in negative:
            ivector -= WI[vocab[word]]
            wordList.append(word)
    for word in vocab:
        if not word in wordList:
            tvector = WI[vocab[word]]    
            cosine_similarity = (np.dot(ivector,tvector.T) / 
                (np.linalg.norm(ivector)*np.linalg.norm(tvector)))
            ranking[word] = cosine_similarity
    ranking = sorted(ranking.items(), key=operator.itemgetter(1)
        , reverse=True) # sorted from value, used 0 for key
    return ranking
print(cosine_similarity(positive=['fred','noodle'],negative=['pizza']))
print(cosine_similarity(positive=['marie','pizza'],negative=['noodle']))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(WI)
X_new = pca.transform(WI)

# 11. Plotting to vector space using PCA
for word in vocab:  
    plt.scatter(X_new[vocab[word]][0],X_new[vocab[word]][1])
    plt.annotate(word, (X_new[vocab[word]][0],X_new[vocab[word]][1]))