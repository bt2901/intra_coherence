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



def my_sort_func(f):
    try:
        fnum = f.split("\\")[-1]
        res = int(fnum.split('.')[0])
    except ValueError:
        res = -1
    return res


pn_folder = r''
vw_folder = 'pn_mixed_lemmatized'

domain_folder = 'PNaukaMixedLemmatized_short'
domain_path = os.path.join(pn_folder, domain_folder)
files_total = os.listdir(domain_path)
files_total = sorted(files_total, key=my_sort_func)





def get_docnum(f):
    short_name = f.split('\\')[-1] # TODO: make more portable, this works only with Windows 
    doc_num = int(short_name.split('.')[0]) - 1
    return doc_num

regex = re.compile(u'[%s]' % re.escape('.')) # to use regex.sub('', s) further

def read_file_data(f):
    file = codecs.open(os.path.join(pn_folder, domain_folder, f), 'r', 'utf-8')

    data = regex.sub(u'', file.read()).split()
    file.close()
    return data

        
def ptdw_vectorized(words, topics, phi_val, phi_rows, local_theta):
    sort = np.argsort(phi_rows)
    rank = np.searchsorted(phi_rows, words, sorter=sort)
    idx_word_array = sort[rank]
    phi_list = phi_val[idx_word_array, :]
    
    unnormalized_ptdw = phi_list * local_theta[np.newaxis, :]
    summed = np.sum(unnormalized_ptdw, axis=1)
    
    ptdw = unnormalized_ptdw / summed[:, np.newaxis]
    
    return ptdw
    
def read_words_from_file(f):
    with codecs.open(os.path.join(pn_folder, domain_folder, f), 'r', 'utf-8') as file:
        data = regex.sub('', file.read()).split()
        return data

    
def calc_doc_ptdw(f, topics, known_words,
                  phi_val, phi_rows,
                  theta_val, theta_cols):
    with codecs.open(os.path.join(pn_folder, domain_folder, f), 'r', 'utf-8') as file:
        data = regex.sub('', file.read()).split()
        data = [w for w in data if (w != 'topic' and not w.isdigit() and w in known_words)]

        doc_num = get_docnum(f)
        idx_doc = theta_cols.index(doc_num)
        
        local_theta = theta_val[:, idx_doc]
        
        doc_ptdw = ptdw_vectorized(data, topics, phi_val, phi_rows, local_theta)
        
        return data, doc_ptdw
    
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



# TODO: far too huge function    
def data_results_save(pars_name, pars_segm, pars_coh, coh_names, file_name=None):
    '''
    if (len(pars_segm[list(pars_segm.keys())[0]]) !=
        len(pars_coh[list(pars_coh.keys())[0]]['newman']['mean-of-means'])):
        print('Different lengths of x- and y- arrays')
        return
    '''
    print("SAVING {}".format((len(pars_segm[list(pars_segm.keys())[0]]), len(pars_coh[list(pars_coh.keys())[0]]['newman']['mean-of-means']))))
    averaging_types = ['mean-of-means', 'mean-of-medians',
                       'median-of-means', 'median-of-medians']
    corrs = {prs: 'prs', spr: 'spr'}
    
    rows = []
    data = ''
    s = ''
    
    # for results.csv
    for pair in pars_coh:
        s = ''
        for k in range(len(pair)):
            s += '{0}={1}, '.format(pars_name[k], pair[k])
        s = s.strip()
        s = s[:-1] # comma
        s += '\n'
        rows += [[s]]

        for segm_type in ['soft', 'harsh']:
            rows += [[segm_type]]
            rows += [[
                '',
                'Newman', '', '', '',
                'Mimno', '', '', '',
                'Cosine', '', '', '',
                'SemantiC', '',
                'TopLen', '', '', '',
                'FoCon'
            ]]
            rows += [[
                '',
                '1', '2', '3', '4',
                '1', '2', '3', '4',
                '1', '2', '3', '4',
                '1', '3',
                '1', '2', '3', '4',
                '0'
            ]]

            for corr in corrs:
                row = [corrs[corr]]
                for coh in coh_names:
                    print("SAVING {}".format(coh))
                    if (coh == 'semantic'):
                        for av_type in ['mean-of-means', 'median-of-means']:
                            x = pars_segm[segm_type]
                            y = pars_coh[pair][coh][av_type]
                            x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                            y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                            x = x_tmp
                            y = y_tmp
                            row += ["'{0:.2f}".format(corr(x, y))]
                        continue
                    if (coh == 'focon'):
                        #continue
                        x = pars_segm[segm_type]
                        y = pars_coh[pair][coh]['res']
                        x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        x = x_tmp
                        y = y_tmp
                        row += ["'{0:.2f}".format(corr(x, y))]
                        continue

                    for av_type in averaging_types:
                        #continue
                        x = pars_segm[segm_type]
                        y = pars_coh[pair][coh][av_type]
                        x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        x = x_tmp
                        y = y_tmp
                        row += ["'{0:.2f}".format(corr(x, y))]

                rows += [row]
                
        rows += [['']]
    
    # for data.txt
    data = ''
    for pair in pars_coh:
        for k in range(len(pair)):
            data += '{0}={1}, '.format(pars_name[k], pair[k])
        data = data.strip()
        data = data[:-1] # comma
        data += '\n'

        for segm_type in ['soft', 'harsh']:
            data += segm_type + '\n'
            
            x = pars_segm[segm_type]
            x = sorted(x)
            data = data_append(data, x)

            for coh in coh_names:
                if (coh == 'semantic'):
                    data += 'SemantiC\n'
                    for av_type in ['mean-of-means', 'median-of-means']:
                        x = pars_segm[segm_type]
                        y = pars_coh[pair][coh][av_type]
                        x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        x = x_tmp
                        y = y_tmp
                        data = data_append(data, y)
                    continue
                if (coh == 'focon'):
                    data += 'FoCon\n'
                    x = pars_segm[segm_type]
                    y = pars_coh[pair][coh]['res']
                    x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                    y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                    x = x_tmp
                    y = y_tmp
                    data = data_append(data, y)
                    continue

                if (coh == 'toplen'):
                    data += 'TopLen\n'
                else:
                    data += coh[0].upper() + coh[1:] + '\n'

                for av_type in averaging_types:
                    x = pars_segm[segm_type]
                    y = pars_coh[pair][coh][av_type]
                    x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                    y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                    x = x_tmp
                    y = y_tmp
                    data = data_append(data, y)
        
        data += '\n'
    
    data = data.strip()
    data += '\n'
    
    if (file_name is None):
        file_name = ''
    else:
        file_name = '.' + file_name

    with codecs.open(os.path.join('results', 'results{0}.csv'.format(file_name)),
              'w', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile,
                               delimiter=',', quotechar='"',
                               quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerows(rows)
    
    with codecs.open(os.path.join('results', 'data{0}.txt'.format(file_name)),
              'w', encoding='utf-8') as f:
        f.write(data)
    
    data = ''
    for k in range(len(pars_name)):
        data += '{0}, '.format(pars_name[k])
    data = data.strip()
    data = data[:-1]
    data += ':\n'
    for pair in pars_coh:
        data += '('
        for k in range(len(pair)):
            data += '{0}, '.format(pair[k])
        data = data.strip()
        data = data[:-1]
        data += '), '
    data = data[:-2] # unnecessary space and comma
    data += '\n\n'
    
    data += 'segmentation quality:\n{0}\n'.format(pars_segm)
    data += '\n'
    data += 'coherences:\n{0}\n'.format(pars_coh)
    data = data.strip()
    data += '\n'
    
    with codecs.open(os.path.join('results', 'data_raw{0}.txt'.format(file_name)),
              'w', encoding='utf-8') as f:
        f.write(data)
