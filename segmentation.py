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

#from document_helper import get_orig_labels, get_docnum, calc_doc_ptdw, read_file_data
from document_helper import calc_doc_ptdw, read_plaintext_and_labels

debug = True
debug = False


            
def calc_cost_matrix(topics, role_nums, f,
                       phi_val, phi_cols, phi_rows,
                       theta_val, theta_cols, theta_rows):
    labeling_time = 0        
    word = ''
    data = ''
    original_topic_num = 0
    
    known_words = phi_rows

    sum_p_tdw = np.zeros((len(role_nums), len(topics)))
    hits_num = np.zeros((len(role_nums), len(topics)))
    
    t_start = time.time()

    time_ptdw = 0
    time_cycle = 0
    time_magic = 0

    for i, line in enumerate(f):
        if debug:
            if i % 100:
                continue

        doc_num, data, original_topic_labels = read_plaintext_and_labels(line)

        t0 = time.time()
        doc_ptdw = calc_doc_ptdw(data, doc_num, 
            phi_val=phi_val, phi_rows=phi_rows,
            theta_val=theta_val, theta_cols=theta_cols
        )
        time_ptdw += time.time() - t0

        t0 = time.time()
        for i, original_topic_num in enumerate(original_topic_labels):
            sum_p_tdw[:, original_topic_num] += doc_ptdw[i]
        time_cycle += time.time() - t0
        
        t0 = time.time()
        argmax_indices = np.argmax(doc_ptdw, axis=1)
        np.add.at(hits_num, [argmax_indices, original_topic_labels], 1)
        time_magic += time.time() - t0
    print("time_ptdw: {} seconds, time_cycle: {} seconds, time_magic: {} seconds".format(time_ptdw, time_cycle, time_magic))
        
    return {'soft': sum_p_tdw, 'harsh': hits_num}

def calc_solution_cost(indexes, cost_matrix):
    
    res_s = 0
    for row, column in indexes['soft']:
        value = cost_matrix['soft'][row][column]
        res_s += value
    res_h = 0
    for row, column in indexes['harsh']:
        value = cost_matrix['harsh'][row][column]
        res_h += value

    return {'soft': res_s, 'harsh': res_h}

def segmentation_evaluation(topics, f,
                            phi_val, phi_cols, phi_rows,
                            theta_val, theta_cols, theta_rows,
                            indexes=None):
    t_mnkr, t_cost_matrix = 0, 0

    mnkr = Munkres()
    res = {'soft': 0, 'harsh': 0}
    res_list = []
    
    #return res, {'soft': [], 'harsh': []}
    topics_number = len(topics)
    
    t_start = time.time()
    # role playing
    top_role_play = (
        calc_cost_matrix(
            topics=topics, role_nums=range(1, len(topics)+1),
            f=f,
            phi_val=phi_val, phi_cols=phi_cols, phi_rows=phi_rows,
            theta_val=theta_val, theta_cols=theta_cols, theta_rows=theta_rows)
    )
    t_cost_matrix += (time.time() - t_start)

    indexes = {'soft': [], 'harsh': []}

    t0 = time.time()
    for s in ['soft', 'harsh']:
        matrix = top_role_play[s]
        #cost_matrix2 = mnkr.make_cost_matrix(matrix,
        #                               lambda cost: sys.maxsize - cost)
        cost_matrix = []
        for row in matrix:
            cost_row = [(sys.maxsize - col) for col in row]
            cost_matrix += [cost_row]
        indexes[s] = mnkr.compute(cost_matrix)
    t_mnkr += (time.time() - t0)
    
    # segmentation evaluation
    res = calc_solution_cost(indexes=indexes, cost_matrix=top_role_play)

    t_end = time.time()
    print("segmentation_evaluation: {} seconds".format(t_end - t_start))
    print("mnkr: {} seconds, cost_matrix: {} seconds".format(t_mnkr, t_cost_matrix))

    return res, indexes