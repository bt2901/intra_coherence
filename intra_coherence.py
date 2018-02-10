# coding: utf-8
from __future__ import print_function, division
from munkres import Munkres # for Hungarian algorithm

import pandas as pd

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

from document_helper import calc_doc_ptdw, read_plaintext

        
def distance_L2(wi, wj):
    return norm(np.subtract(wi, wj))

class coh_toplen_calculator(object):
    def __init__(self, params, topics):
        # lists of lists of topics' lengths
        self.top_lens = [[] for i in range(len(topics))]
        self.topics = topics
        self.params = params
        
        model = {"lenghts_topic_{}".format(t): [] for t, n in enumerate(topics)}
        model["doc_len"] = []
        self.details = pd.DataFrame(model)
        self.details.index.name = "doc_num"
        
    def update(self, doc_num, data, doc_ptdw, phi_val, phi_rows):
        local_top_lens = self.measure(self.params, self.topics, doc_num, data, doc_ptdw, phi_val, phi_rows)
        #print(".", sep="", end="")
        
        #s = pd.Series({"lengths_topic_{}".format(t): local_top_lens[t] for t, n in enumerate(self.topics)})
        s = pd.Series({"lengths_topic_{}".format(t): len(local_top_lens[t]) for t, n in enumerate(self.topics)})
        s["doc_len"] = len(data)
        self.details.loc[doc_num] = s 
        
        for i, topic_name in enumerate(self.top_lens):
            self.top_lens[i] += local_top_lens[i]
            
    def output_details(self, filename):
        self.details.to_csv(filename, sep=";", encoding='utf-8')
            
    def output(self):
        # if some topics didn't appear in the documents
        for i in range(len(self.top_lens)):
            if (len(self.top_lens[i]) == 0):
                self.top_lens[i].append(0)
        
        means = np.array([np.mean(np.array(p)) for p in self.top_lens])
        medians = np.array([np.median(np.array(p)) for p in self.top_lens])
        
        #details = pd.Series(local_top_lens)
        #details.name = doc_num

        # trow away max value
        return {'means': means[np.argsort(means)[:-1]],
                'medians': medians[np.argsort(medians)[:-1]]}
    
    def measure(self, params, topics, doc_num, data, doc_ptdw, phi_val, phi_rows):
        
        threshold, general_penalty = params["threshold"], params["general_penalty"]
        # lists of lists of topics' lengths
        top_lens = [[] for i in range(len(topics))]
        
        known_words = phi_rows

        # positions of topic-related words in the document (and these words as well)
        # (pos_topic_words[topic_num][f][idx] = word)
        pos_topic_words = [{} for topic in topics]
        t0 = time.time()
        time_pos_words = 0
        
        for j, word in enumerate(data):
                            
            p_tdw_list = doc_ptdw[j]
            pos_topic_words[np.argmax(p_tdw_list)][j] = word
        
        pos_list = [sorted(pos_topic_words[l]) for l in range(len(topics))]
        time_pos_words += time.time() - t0
        #print ("time_pos_words: {} sec".format(time_pos_words))

        time_while = 0
            
        for l in range(3):
        #for l in range(len(topics)):
            if (len(pos_list[l]) == 0):
                continue
                
            j = 0
            while (j < len(pos_list[l])):
                i = pos_list[l][j]

                idx = i # first word is also taking part in calculations
                cur_sum = threshold

                t0 = time.time()
                while (cur_sum >= 0 and idx < len(data)):
                    word = data[idx]
                    
                    # word is out of Phi
                    if (word not in known_words):
                        cur_sum -= general_penalty
                        idx += 1
                        continue
                        
                    p_tdw_list = doc_ptdw[idx]

                    argsort_list = np.argsort(np.array(p_tdw_list))
                    idxmax = argsort_list[-1 - bool(argsort_list[-1] == l)]
                    cur_sum += p_tdw_list[l] - p_tdw_list[idxmax]

                    # remove those topic words, which have already taken part in topic length evaluation
                    # TODO: maybe it would be better to do smth else instead of just throwing them out 
                    if (idx in pos_list[l]):
                        j = pos_list[l].index(idx)

                    idx += 1

                time_while += time.time() - t0
                top_lens[l].append(idx - i)
                j += 1
        #print ("time_while: {} sec".format(time_while))
        return top_lens

def coh_semantic_inner(params, topics, doc_num, data, doc_ptdw,
                 phi_val, phi_rows):
    
    known_words = phi_rows
    window = params["window"]
    total_cost = np.zeros( (len(topics),) )
    n_pairs_examined = np.zeros( (len(topics),) )
    
    if (len(data) < window):
        return np.array(means), np.array(N_list)
    
    # positions of topic-related words in the document
    pos_topic_words = [{} for topic in topics]
            
    for i, word in enumerate(data):
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

                total_cost[l] += distance_L2(vec1, vec2)
                n_pairs_examined[l] += 1
    return total_cost, n_pairs_examined

def coh_semantic_inner_alt(params, topics, doc_num, data, doc_ptdw,
                 phi_val, phi_rows):
    
    known_words = phi_rows
    window = params["window"]
    total_cost = np.zeros( (len(topics),) )
    n_pairs_examined = np.ones( (len(topics),) )
    
    if (len(data) < window):
        return np.array(total_cost), np.zeros( (len(topics),) )
    
    n_pairs_examined = len(data) - window + 1
    for i in range(n_pairs_examined):
        cur_window = doc_ptdw[i:i+window, :]
        total_cost += np.var(cur_window) 
            
    return total_cost, n_pairs_examined

        
class coh_semantic_calculator(object):
    def __init__(self, params, topics):
        # lists of lists of topics' lengths
        self.means = np.zeros( (len(topics),) )
        self.N_list = np.zeros( (len(topics),) ) # pairs examined
        self.mode = "alt"

        self.topics = topics
        self.params = params
        model = {"cost_topic_{}".format(t): [] for t, n in enumerate(topics)}
        model["n_pairs"] = []
        self.details = pd.DataFrame(model)
        self.details.index.name = "doc_num"
        
    def update(self, doc_num, data, doc_ptdw, phi_val, phi_rows):
        local_means, local_N_list = self.measure(self.params, self.topics, doc_num, data, doc_ptdw, phi_val, phi_rows)
        self.means += local_means
        self.N_list += local_N_list
        
        s = pd.Series({"cost_topic_{}".format(t): local_means[t] for t, n in enumerate(self.topics)})
        s["n_pairs"] = local_N_list
        self.details.loc[doc_num] = s 
    
    def output_details(self, filename):
        self.details.to_csv(filename, sep=";", encoding='utf-8')
            
    def output(self):
        res = np.divide(-1 * self.means, (self.N_list + 0.001))
        
        # throw out max value when counting mean
        # because if there is a background theme
        # it may affect the result largely
        return {'means': res[np.argsort(res)[:-1]], 'medians': np.full(res.shape, np.nan)}
    
    def measure(self, params, topics, doc_num, data, doc_ptdw, phi_val, phi_rows):
        if self.mode == "alt":
            return coh_semantic_inner_alt(params, topics, doc_num, data, doc_ptdw, phi_val, phi_rows)
        return coh_semantic_inner(params, topics, doc_num, data, doc_ptdw, phi_val, phi_rows)


        
class coh_focon_calculator(object):
    def __init__(self, params, topics):
        # lists of lists of topics' lengths
        self.res = 0.0

        self.details = pd.DataFrame({"cost": []})
        self.details.index.name = "doc_num"

        self.topics = topics
        self.params = params
        
    def update(self, doc_num, data, doc_ptdw, phi_val, phi_rows):

        local_res = self.measure(self.params, self.topics, doc_num, data, doc_ptdw, phi_val, phi_rows)
        s = pd.Series({"cost": local_res})
        
        self.details.loc[doc_num] = s
        self.res += local_res

    def output_details(self, filename):
        self.details.to_csv(filename, sep=";", encoding='utf-8')
        
    def output(self):
        return self.res
    
    def measure(self, params, topics, doc_num, data, doc_ptdw, phi_val, phi_rows):

        threshold = params["focon_threshold"]
        res = 0.0

        known_words = phi_rows

        cur_threshold = 0
        backgrnd = -1
        '''
        for j, word in enumerate(data):
            # looking for the first appropriate word
            if (data[j] not in known_words or np.argmax(doc_ptdw[j]) == backgrnd):
                continue
            vec1 = doc_ptdw[j]
            
            if (word not in known_words):
                cur_threshold = 0
                vec1 = None
                continue
            
            if (vec1 is None and np.argmax(doc_ptdw[j]) == backgrnd):
                cur_threshold = 0
                continue
            elif (vec1 is None and np.argmax(doc_ptdw[j]) != backgrnd):
                vec1 = doc_ptdw[j]
                cur_threshold = 0
                continue
            elif (vec1 is not None and np.argmax(doc_ptdw[j]) == backgrnd):
                cur_threshold += 1
                if (cur_threshold <= threshold):
                    continue
                else:
                    cur_threshold = 0
                    vec1 = None
                    continue     
            # if everything's all right:

            vec2 = doc_ptdw[j]
            
            argsmax = np.argmax([vec1, vec2], axis=1)
            res += np.sum(abs(vec1[argsmax] - vec2[argsmax]))
            
            vec1 = vec2
        return -res
        '''
        return 0
