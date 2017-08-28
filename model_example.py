# coding: utf-8
from __future__ import print_function
import os, glob
import time

import numpy as np
import artm

from document_helper import pn_folder, vw_folder, files_total, domain_path
from measures_utils import ResultStorage, record_results    

print(artm.version())

def create_model(dictionary, num_tokens, num_document_passes):

    specific_topics = ['topic {}'.format(i) for i in range(1, 20)]
    scores = [artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
    artm.TopTokensScore(name='TopTokensScore', num_tokens=10), # web version of Palmetto works only with <= 10 tokens
    artm.SparsityPhiScore(name='SparsityPhiScore'),
    artm.SparsityThetaScore(name='SparsityThetaScore')]
                      
    model = artm.ARTM(topic_names=specific_topics, 
        regularizers=[], cache_theta=True, scores=scores,
        class_ids={'plain_text': 1.0})

    model.initialize(dictionary=dictionary)
    model.num_document_passes = num_document_passes
    
    return model

def create_model_with_background(dictionary, num_tokens, num_document_passes):

    sm_phi_tau = 0.0001 * 1e-4
    sp_phi_tau = -0.0001 * 1e-4

    decor_phi_tau = 1

    specific_topics = ['topic {}'.format(i) for i in range(1, 20)]
    topic_names = specific_topics + ["background"]
    scores = [
        artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
        artm.TopTokensScore(name='TopTokensScore', num_tokens=10, class_id='plain_text'), # web version of Palmetto works only with <= 10 tokens
        artm.SparsityPhiScore(name='SparsityPhiScore'),
        artm.SparsityThetaScore(name='SparsityThetaScore'),
        artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3)
        ]
                      
    model = artm.ARTM(topic_names=specific_topics + ["background"], 
        regularizers=[], cache_theta=True, scores=scores,
        class_ids={'plain_text': 1.0})

    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-sp_phi_tau, topic_names=specific_topics))
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPhi', tau=sm_phi_tau, topic_names=["background"]))
    # model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=decor_phi_tau))

    model.initialize(dictionary=dictionary)
    model.num_document_passes = num_document_passes
    
    return model




coh_names = ['newman', 'mimno', 
             'semantic', 'toplen', "focon"]

#coh_names = ['newman', 'mimno', 'toplen']

intra_coherence_params = {
    "window": 10, "threshold": 0.02, "focon_threshold": 5, "cosine_num_top_tokens": 10, "num_top_tokens": 10,
    "general_penalty": 0.005
}

num_passes_list = range(1, 20)
num_passes_list = range(1, 5)

num_top_tokens = 10




# Vowpal Wabbit
batch_vectorizer = None

if len(glob.glob(os.path.join(pn_folder, vw_folder, '*.batch'))) < 1:
    batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(pn_folder, vw_folder, 'vw_bimodal.txt'),
                                            data_format='vowpal_wabbit',
                                            target_folder=os.path.join(pn_folder, vw_folder)) 
else:
    batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(pn_folder, vw_folder),
                                            data_format='batches')

dictionary = artm.Dictionary()

dict_path = os.path.join(pn_folder, vw_folder, 'dict.dict')

if not os.path.isfile(dict_path):
    dictionary.gather(data_path=batch_vectorizer.data_path)
    dictionary.save(dictionary_path=dict_path)

dictionary.load(dictionary_path=dict_path)

N = 1
# model
model = create_model_with_background(dictionary=dictionary,
                     num_tokens=num_top_tokens,
                     num_document_passes=N) 

# number of cycles
num_of_restarts = 1

def print_status(t0, indent_number, what_is_happening):
    print('({0:>2d}:{1:>2d}){2} {3}'.format(
        int(time.time()-t0)//60//60,
        int(time.time()-t0)//60%60,
        indent*indent_number,
        what_is_happening)
    )

def randomize_model(restart_num, model):
    np.random.seed(restart_num * restart_num + 42)
    
    topic_model_data, phi_numpy_matrix = model.master.attach_model('pwt')
    
    random_init = np.random.random_sample(phi_numpy_matrix.shape)
    random_init /= np.sum(random_init, axis=0)
    np.copyto(phi_numpy_matrix, random_init)
    return topic_model_data, phi_numpy_matrix
    
t0 = time.time()

indent = '    '
indent_number = 0
data_storage = ResultStorage(coh_names, domain_path=domain_path)
for restart_num in range(num_of_restarts):
    topic_model_data, phi_numpy_matrix = randomize_model(restart_num, model)
    # range of models with different segmentation qualities
    num_passes_last = 0
    for num_passes_total in num_passes_list:
        print('************************')
        print_status(t0, indent_number, "teaching model at iter {}".format(num_passes_total))
        indent_number += 1
    
        model.fit_offline(batch_vectorizer=batch_vectorizer,
            num_collection_passes=num_passes_total-num_passes_last
        )
        num_passes_last = num_passes_total

        model_id = " {} ".format({"name": "PLSA", "restart_num": restart_num, "iter": num_passes_total})
        with record_results(model=model, vw_file="vw_bimodal.txt", at=model_id, save_in=data_storage) as recorder:
            for coh_name in coh_names:
                print_status(t0, indent_number, coh_name)
                recorder.evaluate(coh_name, intra_coherence_params)
                    
            indent_number -= 1
            indent_number -= 1

            print_status(t0, indent_number, "segmentation evaluation")
            recorder.evaluate_segmentation_quality()

            indent_number += 1
            
        indent_number -= 1
    
    #print(data_storage.segm_quality.items())
data_storage.data_results_save()
    