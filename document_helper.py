# coding: utf-8
from __future__ import print_function, division
from munkres import Munkres # for Hungarian algorithm


import codecs, re
import sys, os, glob
import tqdm
import time, copy
import csv

import numpy as np
from numpy.linalg import norm
from scipy import stats 
from scipy.misc import comb
from math import floor, ceil, log
import matplotlib.pyplot as plt
from itertools import groupby

import scipy as sp

debug = False
is_full = True

pn_folder = r''
vw_folder = 'pn_mixed_lemmatized' + ("_full" if is_full else "")

domain_folder = 'PNaukaMixedLemmatized' + ('_full' if is_full else '_short')

domain_path = os.path.join(pn_folder, domain_folder)
files_total = os.listdir(domain_path)
#files_total = sorted(files_total, key=my_sort_func)



regex = re.compile(u'[%s]' % re.escape('.')) # to use regex.sub('', s) further

def read_plaintext(line):
    modals = line.split("|")
    doc_num = int(modals[0].strip()) - 1
    data = modals[1].strip().split(" ")[1:]
    return doc_num, data
    
def read_plaintext_and_labels(line):
    modals = line.split("|")
    doc_num = int(modals[0].strip()) - 1
    data = modals[1].strip().split(" ")[1:]
    labels = [int(x) for x in modals[2].strip().split(" ")[1:]]
    return doc_num, data, labels

def read_file_data(f):
    file = codecs.open(os.path.join(pn_folder, domain_folder, f), 'r', 'utf-8')

    data = regex.sub(u'', file.read()).split()
    file.close()
    return data
def read_file_data(f):
    file = codecs.open(os.path.join(pn_folder, domain_folder, f), 'r', 'utf-8')

    data = regex.sub(u'', file.read()).split()
    file.close()
    return data

        
def ptdw_vectorized(words, phi_val, phi_rows, local_theta, phi_sort):
    #sort = np.argsort(phi_rows)
    #print(sort[:10])
    #print(phi_rows[:10])
    rank = np.searchsorted(phi_rows, words, sorter=phi_sort)

    idx_word_array = phi_sort[rank]
    phi_list = phi_val[idx_word_array, :]
    
    unnormalized_ptdw = phi_list * local_theta[np.newaxis, :]
    summed = np.sum(unnormalized_ptdw, axis=1)
    
    ptdw = unnormalized_ptdw / summed[:, np.newaxis]
    
    return ptdw
    
def read_words_from_file(f):
    with codecs.open(os.path.join(pn_folder, domain_folder, f), 'r', 'utf-8') as file:
        data = regex.sub('', file.read()).split()
        return data

    
def _calc_doc_ptdw(data, doc_num,
                  phi_val, phi_rows, phi_sort,
                  theta_val, theta_cols):

        idx_doc = theta_cols.index(doc_num)
        
        local_theta = theta_val[:, idx_doc]
        
        doc_ptdw = ptdw_vectorized(data, phi_val, phi_rows, local_theta, phi_sort)
        
        return doc_ptdw
    
# ===========


def get_orig_labels(data_filtered, data):
    '''
    get original_topic_labels all at once
    could be sped up, but it takes < 1 second for entire collection, not really significant 
    '''
    original_topic_labels = np.zeros( (len(data_filtered),), dtype=int)
    i, j = 0, 0
    current_topic = None
    while i < len(data_filtered):
        if data[j] == "topic":
            current_topic = int(data[j+1]) - 1 # will crash if not number
            j += 2
        if data_filtered[i] == data[j]:
            original_topic_labels[i] = current_topic
            i, j = i+1, j+1
        else:
            j += 1
    return original_topic_labels
    
    
def prs(l1, l2):
    return sp.stats.pearsonr(l1, l2)[0]

def spr(l1, l2):
    return sp.stats.spearmanr(l1, l2)[0]

def data_append(data, x):
    for num in x:
        data += '{0:.2f},'.format(num)
    data = data[:-1]
    data += '\n'
    return data


