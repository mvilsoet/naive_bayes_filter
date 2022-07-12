# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
import scipy as sp
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:

    for i,x in enumerate(X): #i is index, x is an element of X (aka an email)
        if y[i] == 1: #is the email labeled as spam or ham?
            for word in x: #iterate through the words in the email
                if word not in pos_vocab.keys(): #check if word is in pos_vocab as a key
                    pos_vocab[word] = 0 #add it 
                pos_vocab[word] += 1 #increment it

        else:
            for word in x:
                if word not in neg_vocab.keys():
                    neg_vocab[word] = 0
                neg_vocab[word] += 1

    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: pairs of words 
        values: number of times the word pair appears 
    """

    ##TODO:
    pos_vocab, neg_vocab = create_word_maps_uni(X, y, max_size=None)
    # print(pos_vocab)
    for i,x in enumerate(X): #i is index, x is an element of X (aka an email)
        if y[i] == 1: #is the email labeled as spam or ham?
            for word_idx in range(len(x[:-1])): #iterate through the words in the email
                if x[word_idx] + " " + x[word_idx+1] not in pos_vocab.keys(): #check if word is in pos_vocab as a key
                    pos_vocab[x[word_idx] + " " + x[word_idx+1]] = 0 #add it 
                pos_vocab[x[word_idx] + " " + x[word_idx+1]] += 1 #increment it

        else:
            for word_idx in range(len(x[:-1])): #iterate through the words in the email
                if x[word_idx] + " " + x[word_idx+1] not in neg_vocab.keys(): #check if word is in pos_vocab as a key
                    neg_vocab[x[word_idx] + " " + x[word_idx+1]] = 0 #add it 
                neg_vocab[x[word_idx] + " " + x[word_idx+1]] += 1 #increment it


    # print(pos_vocab["subject"])
    # print(pos_vocab["you are"])

    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    neg_prior = 1-pos_prior #pos_prior -> guess likelihood(prior) of any random email being SPAM

    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels)

    ham_likelihood = {}
    spam_likelihood = {}
    
    ham_type = len(neg_vocab)
    spam_type = len(pos_vocab)
    ham_tokens = sum(neg_vocab.values())
    spam_tokens = sum(pos_vocab.values())

    OOV_ham_prob = ( laplace / (ham_tokens + laplace*(ham_type)) )
    OOV_spam_prob = ( laplace / (spam_tokens + laplace*(spam_type)) )

    for word in neg_vocab:
        ham_likelihood[word] = ( (neg_vocab[word] + laplace) / (ham_tokens + laplace*(1+ham_type)) )
    for word in pos_vocab:
        spam_likelihood[word] = ( (pos_vocab[word] + laplace) / (spam_tokens + laplace*(1+spam_type)) )

    dev_labels = []
    for dev_email in dev_set: 

        ham_likelihood_sum = 0 
        spam_likelihood_sum = 0
        for word in dev_email:
            if word in ham_likelihood:
                ham_likelihood_sum += np.log(ham_likelihood[word]) #using training set's likehood in novel set
            else:
                ham_likelihood_sum += np.log(OOV_ham_prob)
                
            if word in spam_likelihood:
                spam_likelihood_sum += np.log(spam_likelihood[word]) #using training set's likehood in novel set
            else:
                spam_likelihood_sum += np.log(OOV_spam_prob)

        #print(ham_likelihood_sum)

        fx_ham = np.log(neg_prior) + ham_likelihood_sum
        fx_spam = np.log(pos_prior) + spam_likelihood_sum

        dev_labels.append( np.argmax([fx_ham,fx_spam])) #appends 0 if fx_ham is larger to mark dev_email as ham

    return dev_labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    neg_prior = 1-pos_prior #pos_prior -> guess likelihood(prior) of any random email being SPAM

    pos_vocab, neg_vocab = create_word_maps_bi(train_set, train_labels)

    ham_likelihood = {}
    spam_likelihood = {}
    
    ham_type = len(neg_vocab)
    spam_type = len(pos_vocab)
    ham_tokens = sum(neg_vocab.values())
    spam_tokens = sum(pos_vocab.values())

    for nGram in neg_vocab:
        ham_likelihood[nGram] = ( (neg_vocab[nGram] + unigram_laplace) / (ham_tokens + unigram_laplace*(1+ham_type)) )
    for nGram in pos_vocab:
        spam_likelihood[nGram] = ( (pos_vocab[nGram] + unigram_laplace) / (spam_tokens + unigram_laplace*(1+spam_type)) )

    dev_labels = []
    for dev_email in dev_set: 

        ham_likelihood_sum_uni = 0 
        ham_likelihood_sum_bi = 0
        spam_likelihood_sum_uni = 0
        spam_likelihood_sum_bi = 0

        for word in dev_email:
            if word in ham_likelihood:
                ham_likelihood_sum_uni += np.log(ham_likelihood[word]) #using training set's likehood in novel set
            else:
                ham_likelihood_sum_uni += np.log( unigram_laplace / (ham_tokens + unigram_laplace*(ham_type)) )
                
            if word in spam_likelihood:
                spam_likelihood_sum_uni += np.log(spam_likelihood[word]) #using training set's likehood in novel set
            else:
                spam_likelihood_sum_uni += np.log( unigram_laplace / (spam_tokens + unigram_laplace*(spam_type)) )
            
        for word_idx in range(len(dev_email[:-1])): 
            bigram = dev_email[word_idx] + " " + dev_email[word_idx+1]

            if bigram in ham_likelihood:
                ham_likelihood_sum_bi += np.log(ham_likelihood[bigram])
            else:
                ham_likelihood_sum_bi += np.log( bigram_laplace / (ham_tokens + bigram_laplace*(ham_type)) )

            if bigram in spam_likelihood:
                spam_likelihood_sum_bi += np.log(spam_likelihood[bigram])
            else:
                spam_likelihood_sum_bi += np.log( bigram_laplace / (spam_tokens + bigram_laplace*(spam_type)) )

        fx_ham = (1-bigram_lambda)*( np.log(neg_prior) + ham_likelihood_sum_uni ) + (bigram_lambda)*( np.log(neg_prior) + ham_likelihood_sum_bi )
        fx_spam = (1-bigram_lambda)*( np.log(pos_prior) + spam_likelihood_sum_uni ) + (bigram_lambda)*( np.log(pos_prior) + spam_likelihood_sum_bi )

        dev_labels.append( np.argmax([fx_ham,fx_spam]) ) #appends 0 if fx_ham is larger to mark dev_email as ham

    return dev_labels