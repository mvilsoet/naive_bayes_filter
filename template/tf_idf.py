# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021
# Modified by Mohit Goyal (mohit@illinois.edu) on 01/16/2022
"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator


# def compute_tf_idf(train_set, train_labels, dev_set):
#     """
#     train_set - List of list of words corresponding with each movie review
#     example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
#     Then train_set := [['like','this','movie'], ['i','fall','asleep']]

#     train_labels - List of labels corresponding with train_set
#     example: Suppose I had two reviews, first one was positive and second one was negative.
#     Then train_labels := [1, 0]

#     dev_set - List of list of words corresponding with each review that we are testing on
#               It follows the same format as train_set

#     Return: A list containing words with the highest tf-idf value from the dev_set documents
#             Returned list should have same size as dev_set (one word from each dev_set document)
#     """



#     # TODO: Write your code here
    


#     # return list of words (should return a list, not numpy array or similar)
#     return []

def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """


    # TODO: Write your code here


    # TF
    for dev_email in dev_set:
        word_freq = Counter() #counter data structure, uses hash table similar to dict
        tf_idf = {} #just a dict

        for word in dev_email:
            word_freq.update({word: 1}) #+1 on counter

        for word, count in word_freq.items():
            # IDF
            word_count = 0
            for train_email in train_set:
                if word in train_email:
                    word_count += 1
                    
            #print("!")
            tf_idf[word] = count/len(dev_email) * np.log(len(train_set) / (1 + word_count))
        # ===tf_idf dict finished===

        max_tfidf = []
        max_val = 0
        max_word = ""
        for word, tfidf in tf_idf.items():
            if tfidf > max_val:
                max_val = tfidf
                max_word = word
                
        #print("?")
        max_tfidf.append(max_word)

    # return list of words (should return a list, not numpy array or similar)
    return max_tfidf