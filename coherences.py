# coding: utf-8
from __future__ import print_function, division
from munkres import Munkres # for Hungarian algorithm


import codecs, re
import sys, os, glob
import tqdm
import time, copy

import numpy as np
from numpy.linalg import norm
from scipy import stats 
from scipy.misc import comb
from math import floor, ceil, log
import matplotlib.pyplot as plt
from itertools import groupby



from document_helper import regex
from document_helper import read_plaintext

# ### 2-old Mimno

# ###### coh_mimno(window, num_tokens)
# 
# Считает когерентности тем в модели по Мимно
# 
# *Вход*:  
# window [int] - ширина окна, в котором ищем топ-слова  
# num_tokens [int] - число топ-слов, по которым считается когерентность
# 
# *Выход*:  
# {'means': [., ...], 'medians': [., ...]} [dict] - словарь со значениями - листами когерентностей для тем модели, ключами - индикаторами, как считались эти когерентности: как среднее или медиана по score($w_i$, $w_j$)

# In[13]:

def coh_mimno(window, num_top_tokens, model, topics, file):
    means = [0 for i in range(len(topics))] # part of result
    medians = [0 for i in range(len(topics))] # part of result
    
    topic_words = [model.score_tracker['TopTokensScore'].last_tokens[topic] for topic in topics]
    assert (num_top_tokens <= len(topic_words[0])),"Ask more top-tokens than available"
    topic_words = [topic_words[i][:num_top_tokens] for i in range(len(topics))]
    
    le = 0 # left pointer
    ri = 0 # right pointer
    
    # number of articles containing single wi
    dw = [[0] * len(topic_words[i]) for i in range(len(topics))]
    
    matrix = (
        [[[0 for i in range(0, len(topic_words[k]))] for j in range(0, len(topic_words[k]))]
           for k in range(len(topics))]
    ) # matrix[t] = d(wi, wj) - number of articles containing both wi and wj
    
    for line in file:
        doc_num, data = read_plaintext(line)
              
        if (len(data) < window):
            continue
              
        left = 0
        right = left + window
        
        dw_current = [[0] * len(topic_words[k]) for k in range(len(topics))]
        matrix_current = [[[0 for i in range(0, len(topic_words[k]))] for j in range(0, len(topic_words[k]))] for k in range(len(topics))]
        
        while (right <= len(data)):
            current_slice = data[left:right] # excluding right
            current_points = [[0] * len(topic_words[k]) for k in range(len(topics))] # counting addition to p(wi) at the current slice
            
            for k in range(len(topics)):
                for i in range(0, len(current_slice)):
                    if (current_slice[i] in topic_words[k]):
                        current_points[k][topic_words[k].index(current_slice[i])] = 1
                        dw_current[k][topic_words[k].index(current_slice[i])] = 1
                if (any(current_points[k])):
                    # FIXIT: with np may be faster
                    for i in range(0, len(current_points[k])-1):
                        if (current_points[k][i] == 0):
                            continue
                        for j in range(i+1, len(current_points[k])):
                            if (current_points[k][j] == 0):
                                continue
                            matrix_current[k][i][j] = 1
                            matrix_current[k][j][i] = 1
            left += 1
            right += 1
                    
        dw = np.add(dw, dw_current)
        matrix = np.add(matrix, matrix_current)
    
    dw = np.array(dw)
    pmi_list = [np.array([0.0] * comb(len(topic_words[i]), 2, exact=True)) for i in range(len(topics))]
    
    # counting score(wi, wj)
    for l in range(len(topics)):
        k = 0
        for i in range(0, len(topic_words[l])-1):
            if (dw[l][i] == 0):
                continue
            for j in range(i+1, len(topic_words[l])):
                #if (dw[l][j] == 0):
                #    continue
                pmi_list[l][k] = log((matrix[l][i][j] + 1)/ dw[l][i])
                k += 1
    
    means = np.array([np.mean(pmi_list[i]) for i in range(len(topics))])
    medians = np.array([np.median(pmi_list[i]) for i in range(len(topics))])
    
    return {'means': means[np.argsort(means)[:-1]],
            'medians': medians[np.argsort(medians)[:-1]]}

# ### 3-old "Cosine"

# ###### coh_cosine(num_tokens)
# 
# Считает когерентности тем в модели по косинусу угла между тематическими векторами топ-слов темы
# 
# *Вход*:  
# num_tokens [int] - число топ-слов, по которым считается когерентность
# 
# *Выход*:  
# {'means': [., ...], 'medians': [., ...]} [dict] - словарь со значениями - листами когерентностей для тем модели, ключами - индикаторами, как считались эти когерентности: как среднее или медиана по score($w_i$, $w_j$)


def angle_cosine(wi, wj):
    return np.dot(wi, wj) / norm(wi) / norm(wj)

    
def coh_cosine(num_top_tokens, model, topics, phi_val, phi_rows):  
    topic_words = [model.score_tracker['TopTokensScore'].last_tokens[topic] for topic in topics]
    assert (num_top_tokens <= len(topic_words[0])),"Ask more top-tokens than available"
    topic_words = [topic_words[i][:num_top_tokens] for i in range(len(topics))]
    
    # topic_vecs[t] = (t's top-words) x (distribution by topics)
    topic_vecs = [[phi_val[phi_rows.index(word)] for word in topic_words[l]] for l in range(len(topics))]
    
    pmi_list = [np.array([0.0] * comb(len(topic_words[l]), 2, exact=True)) for l in range(len(topics))]
        
    # counting score(wi, wj)
    for l in range(len(topics)):
        k = 0
        for i in range(0, len(topic_words[l])-1):
            for j in range(i+1, len(topic_words[l])):
                pmi_list[l][k] = angle_cosine(topic_vecs[l][i], topic_vecs[l][j])
                k += 1
    
    means = np.array([np.mean(pmi_list[i]) for i in range(len(topics))])
    medians = np.array([np.median(pmi_list[i]) for i in range(len(topics))])
    
    return {'means': means[np.argsort(means)[:-1]],
            'medians': medians[np.argsort(medians)[:-1]]}
            
            

    
    

def coh_newman(window, num_top_tokens, model, topics, file):
    means = [0 for i in range(len(topics))] # part of future result
    medians = [0 for i in range(len(topics))] # part of future result
    
    topic_words = [model.score_tracker['TopTokensScore'].last_tokens[topic] for topic in topics]
    assert (num_top_tokens <= len(topic_words[0])),"Ask more top-tokens than available"
    topic_words = [topic_words[i][:num_top_tokens] for i in range(len(topics))]
    
    le = 0 # left pointer
    ri = 0 # right pointer
    pw = [[0] * len(topic_words[i]) for i in range(len(topics))] # single probabilities
    N = 0 # total number of windows
    matrix = (
        [[[0 for i in range(0, len(topic_words[0]))]
           for j in range(0, len(topic_words[0]))] for topic in topics]
    ) # matrix[t] = (p(wi, wj))
    
    for line in file:
        doc_num, data = read_plaintext(line)
        
        if (len(data) < window):
            continue
            
        left = 0
        right = left + window
        
        while (right <= len(data)): # excluding right, so <= is OK
            # get window сonsecutive words from the file text
            current_slice = data[left:right] # excluding right
            # counting addition to p(wi) at the current slice
            current_points = [[0] * len(topic_words[i]) for i in range(len(topics))]
            
            for k in range(0, len(topics)):
                for i in range(0, len(current_slice)):
                    if (current_slice[i] in topic_words[k]):
                        current_points[k][topic_words[k].index(current_slice[i])] += 1
                        pw[k][topic_words[k].index(current_slice[i])] += 1
                if (any(current_points[k])):
                    # FIXIT: with np may be faster
                    for i in range(0, len(current_points[k])-1):
                        if (current_points[k][i] == 0):
                            continue
                        for j in range(i+1, len(current_points[k])):
                            if (current_points[k][j] == 0):
                                continue
                            matrix[k][i][j] += 1
                            matrix[k][j][i] += 1
            left += 1
            right += 1
            
        N += left + 1
    
    pw = np.array(pw)
    pmi_list = [np.array([0.0] * comb(len(topic_words[i]), 2, exact=True)) for i in range(len(topics))]
    
    # counting PMI
    for l in range(len(topics)):
        k = 0
        for i in range(0, len(topic_words[l])-1):
            if (pw[l][i] == 0):
                continue
            for j in range(i+1, len(topic_words[l])):
                if (pw[l][j] == 0):
                    continue
                under_log = (N * matrix[l][i][j] + 1) / pw[l][i] / pw[l][j]
                pmi_list[l][k] = log(under_log)
                k += 1
                
    means = np.array([np.mean(pmi_list[i]) for i in range(len(topics))])
    medians = np.array([np.median(pmi_list[i]) for i in range(len(topics))])
    
    return {'means': means[np.argsort(means)[:-1]],
            'medians': medians[np.argsort(medians)[:-1]]}


