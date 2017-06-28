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


def coh_toplen(threshold, topics, files, files_path,
               phi_val, phi_cols, phi_rows,
               theta_val, theta_cols, theta_rows,
               general_penalty=0.005):
    
    # lists of lists of topics' lengths
    top_lens = [[] for i in range(len(topics))]
    
    known_words = phi_rows
        
    for f in files:
        if get_docnum(f) not in theta_cols:
            continue
        # positions of topic-related words in the document (and these words as well)
        # (pos_topic_words[topic_num][f][idx] = word)
        pos_topic_words = [{} for topic in topics]
        
        file = codecs.open(os.path.join(files_path, f), 'r', 'utf-8')
        data_untouched = regex.sub('', file.read()).split()
        file.close()
        
        data, doc_ptdw = calc_doc_ptdw(
            f=f, topics=topics, known_words=known_words,
            phi_val=phi_val, phi_rows=phi_rows,
            theta_val=theta_val, theta_cols=theta_cols
        )
        
        for j in range(len(data_untouched)):
            word = data_untouched[j]
            
            if (word not in known_words):
                continue
                
            p_tdw_list = doc_ptdw[data.index(word)]
            pos_topic_words[np.argmax(p_tdw_list)][j] = word
        
        pos_list = [sorted(pos_topic_words[l]) for l in range(len(topics))]
            
        for l in range(len(topics)):
            if (len(pos_list[l]) == 0):
                continue
                
            j = 0
            while (j < len(pos_list[l])):
                i = pos_list[l][j]

                idx = i # first word is also taking part in calculations
                cur_sum = threshold

                while (cur_sum >= 0 and idx < len(data_untouched)):
                    word = data_untouched[idx]
                    
                    # word is out of Phi
                    if (word not in known_words):
                        cur_sum -= general_penalty
                        idx += 1
                        continue
                        
                    p_tdw_list = doc_ptdw[data.index(word)]

                    argsort_list = np.argsort(np.array(p_tdw_list))
                    idxmax = argsort_list[-1 - bool(argsort_list[-1] == l)]
                    cur_sum += p_tdw_list[l] - p_tdw_list[idxmax]

                    # remove those topic words, which have already taken part in topic length evaluation
                    # TODO: maybe it would be better to do smth else instead of just throwing them out 
                    if (idx in pos_list[l]):
                        j = pos_list[l].index(idx)

                    idx += 1

                top_lens[l].append(idx - i)
                j += 1
    
    # if some topics didn't appear in the documents
    for i in range(len(top_lens)):
        if (len(top_lens[i]) == 0):
            top_lens[i].append(0)
    
    means = np.array([np.mean(np.array(p)) for p in top_lens])
    medians = np.array([np.median(np.array(p)) for p in top_lens])
    # trow away max value
    return {'means': means[np.argsort(means)[:-1]],
            'medians': medians[np.argsort(medians)[:-1]]}

        
def distance_L2(wi, wj):
    return norm(np.subtract(wi, wj))

            
def coh_semantic(window, topics, files, files_path,
                 phi_val, phi_cols, phi_rows,
                 theta_val, theta_cols, theta_rows):
    
    known_words = phi_rows
    
    means = [0 for i in range(len(topics))]
    N_list = [0 for i in range(len(topics))] # pairs examined
    for f in files:
        if get_docnum(f) not in theta_cols:
            continue
        # positions of topic-related words in the document
        pos_topic_words = [{} for topic in topics]
    
        data, doc_ptdw = calc_doc_ptdw(
            f=f, topics=topics, known_words=known_words,
            phi_val=phi_val, phi_rows=phi_rows,
            theta_val=theta_val, theta_cols=theta_cols
        )
        
        if (len(data) < window):
            continue
        
        for i in range(0, len(data)):
            word = data[i]
                
            p_tdw_list = doc_ptdw[i]
            pos_topic_words[np.argmax(p_tdw_list)][i] = word
        
        pos_list = [sorted(pos_topic_words[l]) for l in range(len(topics))]
    
        # counting score(wi, wj)
        for l in range(len(topics)):            
            if (len(pos_list[l]) == 0):
                continue

            for i in range(0, len(pos_list[l])-1):
                pos_i = pos_list[l][i]
                vec1 = phi_val[phi_rows.index(pos_topic_words[l][pos_i])]

                for j in range(i+1, min(i+window, len(pos_list[l]))):
                    pos_j = pos_list[l][j]
                    vec2 = phi_val[phi_rows.index(pos_topic_words[l][pos_j])]

                    means[l] += distance_L2(vec1, vec2)
                    N_list[l] += 1
    
    means = np.array(means)
    N_list = np.array(N_list)
    
    res = np.divide(-1 * means, (N_list + 0.001))
    
    # throw out max value when counting mean
    # because if there is a background theme
    # it may affect the result largely
    return {'means': res[np.argsort(res)[:-1]]}
