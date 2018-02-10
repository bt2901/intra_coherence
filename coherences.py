# coding: utf-8
from __future__ import print_function, division
from munkres import Munkres # for Hungarian algorithm
from collections import defaultdict, Counter

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
from itertools import groupby, combinations

import pandas as pd

from document_helper import regex
from document_helper import read_plaintext

def write_matrix_to_file(filename, matrix):
    np.save(filename, matrix)





''' 

def calc_umass_coherence(toptokens, coocur, freq, n_titles):
    # toptokens ARE SORTED
    # calc coocurence
    # coocur, freq, n_titles = read_coocurence(toptokens)
    coherence = 0.0
    for i, toptok in enumerate(toptokens):
        for j, toptok2 in enumerate(toptokens[:i]):
            if not freq[toptok2]:
                print toptok2
                print freq[toptok2]
                
            val = float(coocur[toptok][toptok2] + 1.0/n_titles) / freq[toptok2]
            # print(val, np.log(val))
            coherence += np.log(val)
    return coherence
     
def calc_pmi_coherence(toptokens, coocur, freq, n_titles):
    # calc coocurence
    # coocur, freq, n_titles = read_coocurence(toptokens)

    pmi = 0
    for i, toptok1 in enumerate(toptokens):
        for j, toptok2 in enumerate(toptokens):
            prob_both = (coocur[toptok1][toptok2] + 1.0/n_titles) / n_titles
            prob_independent = float(freq[toptok2] * freq[toptok1]) / (n_titles * n_titles)
            pmi += np.log(prob_both / prob_independent)
    return pmi

     
def calc_avg_coherence(top_tokens_score, unified_dict, num_topics, coocur_file, n_titles):
    top_tokens_triplets = zip(top_tokens_score.topic_index, zip(top_tokens_score.token, top_tokens_score.weight))
    UMass = np.zeros(num_topics)
    PMI = np.zeros(num_topics)
    for topic_index, group in itertools.groupby(top_tokens_triplets, key=lambda (topic_index, _): topic_index):
        group_words = [token for topic_id, (token, weight) in group]
        coocur, freq = read_coocurence_cached(group_words, coocur_file)
        UMass[topic_index] = calc_umass_coherence(group_words, coocur, freq, n_titles)
        PMI[topic_index] = calc_pmi_coherence(group_words, coocur, freq, n_titles)
    
    return {"UMass": np.mean(UMass), "PMI": np.mean(PMI)}
''' 
import pickle
    
def read_coocurence_cached(topic_words, topics, window, file):
    unknown_cooc = set()
    try:
        with codecs.open("coocur.p", "r", encoding="utf8") as f_c:
            with codecs.open("freq.p", "r", encoding="utf8") as f_f:
                coocur = pickle.load( f_c )
                freq = pickle.load( f_f )
    except:
        print("Pickles not found")
        coocur = defaultdict(Counter)
        freq = Counter()
    
    for k in range(0, len(topics)):
        for t1, t2 in combinations(topic_words[k], r=2):
            if t2 not in coocur[t1]:
                unknown_cooc.add(t1)
                unknown_cooc.add(t2)
        for toptok in topic_words[k]:
            if toptok not in freq:
                unknown_cooc.add(toptok)

    unknown_tokens = list(unknown_cooc)
    if unknown_tokens:
        print(u", ".join(unknown_tokens))
        matrix, pw, N = calc_coocur_matrix_legacy([unknown_tokens], ["fake_topic"], window, file)
        for i, t1 in enumerate(unknown_tokens):
            for j, t2 in enumerate(unknown_tokens):
                coocur[t1][t2] = matrix[0][i][j]
                coocur[t2][t1] = matrix[0][j][i]
            freq[t1] = pw[0][i]
        freq["__n_windows__"] = N
        with codecs.open("coocur.p", "wb", encoding="utf8") as f:
            pickle.dump(coocur, f)
        with codecs.open("freq.p", "wb", encoding="utf8") as f:
            pickle.dump( freq, f)
        print("Pickles updated")
    else: print("Result was cached, using it")

    return coocur, freq

def generate_coocur_matrix(topic_words, topics, window, file, coocur_dict, freq_dict):
    matrix = (
        [[[0 for i in range(0, len(topic_words[0]))]
            for j in range(0, len(topic_words[0]))] 
                for topic in topics]
    ) # matrix[t] = (p(wi, wj))
    pw = [[0] * len(topic_words[i]) for i in range(len(topics))] # single probabilities

    for k in range(0, len(topics)):
        for i, w1 in enumerate(topic_words[k]):
            for j, w2 in enumerate(topic_words[k]):
                matrix[k][i][j] = coocur_dict[w1][w2]
                matrix[k][j][i] = coocur_dict[w2][w1]
            pw[k][i] = freq_dict[w1]

    return matrix, pw, freq_dict["__n_windows__"]

def calc_coocur_matrix_legacy(topic_words, topics, window, file):
    matrix = (
        [[[0 for i in range(0, len(topic_words[0]))]
            for j in range(0, len(topic_words[0]))] 
                for topic in topics]
    ) # matrix[t] = (p(wi, wj))
    pw = [[0] * len(topic_words[i]) for i in range(len(topics))] # single probabilities

    N = 0 # total number of windows
    # coocur_dfs = [pd.DataFrame(data=np.array(matrix[k]), columns=topic_words[k], index=topic_words[k]) for k, _ in enumerate(topics)] 
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
                            #print(topic_words[k][i], topic_words[k][j])
                            #print((topic_words[k][i], topic_words[k][j]) in coocur_dfs[k].columns) 
                            #print(coocur_dfs[k].columns)
                            # coocur_dfs[k].loc[topic_words[k][i], topic_words[k][j]] += 1
                            # coocur_dfs[k].loc[topic_words[k][j], topic_words[k][i]] += 1
            left += 1
            right += 1
            
        N += left + 1
    # for k, topic in enumerate(topics):
        # coocur_dfs[k].to_csv("coocur_{}.csv".format(topic), encoding="utf8")
    return matrix, pw, N


def calc_coocur_matrix(topic_words, topics, window, file):
    coocur, freq = read_coocurence_cached(topic_words, topics, window, file)
    matrix, pw, N = generate_coocur_matrix(topic_words, topics, window, file, coocur, freq)# matrix[t] = (p(wi, wj))
    return matrix, pw, N

def calc_in_same_document_matrix(topic_words, topics, window, file):
    matrix = (
        [[[0 for i in range(0, len(topic_words[k]))] 
            for j in range(0, len(topic_words[k]))]
                for k in range(len(topics))]
    ) # matrix[t] = d(wi, wj) - number of articles containing both wi and wj
    
    # number of articles containing single wi
    dw = [[0] * len(topic_words[i]) for i in range(len(topics))]

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
    

    return matrix, dw

def calc_in_same_document_matrix_new(topic_words, topics, window, file):
    pass

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
    
    
    matrix, dw = calc_in_same_document_matrix(topic_words, topics, window, file)
    write_matrix_to_file("MIMNO_matrix.csv", matrix) 
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
    matrix, pw, N = calc_coocur_matrix(topic_words, topics, window, file)
    write_matrix_to_file("NEWMAN_matrix.csv", matrix) 
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


