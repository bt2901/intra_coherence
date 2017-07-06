# coding: utf-8
from __future__ import print_function
import copy
import traceback

import numpy as np

from collections import defaultdict

from document_helper import files_total, data_results_save
from segmentation import segmentation_evaluation

from coherences import coh_newman, coh_mimno #, #coh_cosine
from intra_coherence import coh_semantic, coh_toplen, coh_focon
    

class ModelIdentifier(object):
    def __init__(self, id):
        self.id = id

coh_names = ['newman', 'mimno', 
             'semantic', 'toplen', 'focon']
coh_names_top_tokens = ['newman', 'mimno']
coh_funcs = [coh_newman, coh_mimno, 
             coh_semantic, coh_toplen, coh_focon]


class ResultStorage(object):
    def __init__(self, coh_names, domain_path):
        self.domain_path = domain_path 
        self.coh_names = coh_names
        self.segm_modes = ["soft", "harsh"]
        self.segm_quality = {s: dict() for s in self.segm_modes}
        self.coherences = {s: defaultdict(dict) for s in self.coh_names}

    def save_segm(self, model_id, segm_quality_tmp):
        for s in self.segm_modes:
            self.segm_quality[s][model_id] = (
                          np.mean(segm_quality_tmp[s])
            )

    def save_coh(self, model_id, coherences_tmp):
        for name in self.coh_names:
            for mode in coherences_tmp[name]:
                self.coherences[name][mode][model_id] = np.mean(coherences_tmp[name][mode])

    def data_results_save(self):
        raise NotImplementedError


functions_data = {name: {"func": func, "by_top_tokens": (name in coh_names_top_tokens)} for name, func in zip(coh_names, coh_funcs)}
class record_results(object):
    def __init__(self, model, files, at, save_in):
        self.save_in = save_in
        self.files = files
        self.at = at
        self.model = model
    def __enter__(self):
        self.phi = self.model.get_phi()
        self.theta = self.model.get_theta()
        self._coherences_tmp = self._create_coherences_carcass(self.save_in.coh_names)
        self._segm_quality_tmp = self._create_segm_quality_carcass()
        self.theta_cols=list(self.theta.columns)
        self.phi_cols=list(self.phi.columns)
        self.phi_rows=list(self.phi.index)
        self.theta_rows=list(self.theta.index)
        return self

    def __exit__(self, exc_type, exc_value, tr):
        if exc_type is not None:
            print(exc_type, exc_value, tr)
            traceback.print_tb(tr)
        self.save_in.save_segm(self.at, self._segm_quality_tmp)
        self.save_in.save_coh(self.at, self._coherences_tmp)

    def evaluate(self, coh_name, coh_params):
        #raise NotImplementedError
        coh_func = functions_data[coh_name]["func"]
        if (coh_name in coh_names_top_tokens):
            (window, num_top_tokens) = coh_params["window"], coh_params["num_top_tokens"]
            coh_list = coh_func(
                window=window, num_top_tokens=num_top_tokens,
                model=self.model, topics=self.model.topic_names,
                files=self.files, files_path=self.save_in.domain_path
            )
        else:
            coh_list = coh_func(
                coh_params, topics=self.model.topic_names,
                files=self.files, files_path=self.save_in.domain_path,
                phi_val=self.phi.values, phi_cols=self.phi_cols, phi_rows=self.phi_rows,
                theta_val=self.theta.values, theta_cols=self.theta_cols, theta_rows=self.theta_rows,
            )

        self._append_all_measures(self._coherences_tmp, coh_name, coh_list)
    def evaluate_segmentation_quality(self):
            cur_segm_eval, indexes = (
                segmentation_evaluation(
                    topics=self.model.topic_names,
                    collection=self.files, collection_path=self.save_in.domain_path,
                    files=self.files,
                    phi_val=self.phi.values, phi_cols=self.phi_cols, phi_rows=self.phi_rows,
                    theta_val=self.theta.values, theta_cols=self.theta_cols, theta_rows=self.theta_rows,
                    indexes=None
                )
            )
            self._segm_quality_tmp['soft'] = np.append(
                self._segm_quality_tmp['soft'], cur_segm_eval['soft']
            )
            self._segm_quality_tmp['harsh'] = np.append(
                self._segm_quality_tmp['harsh'], cur_segm_eval['harsh']
            )
            
    def _create_segm_quality_carcass(self):
        segm_quality_carcass = {mode: np.array([]) for mode in ["soft", "harsh"]}
        return segm_quality_carcass
    def _create_coherences_carcass(self, coh_names):
        coherences_carcass = defaultdict(dict)
        for cn in coh_names:
            for mode in ['mean-of-means', 'mean-of-medians', 'median-of-means', 'median-of-medians']:
                coherences_carcass[cn][mode] = np.array([])
        if "focon" in coh_names:
            coherences_carcass["focon"] = {'res': np.array([])}
                
        return coherences_carcass

    def _measures_append(self, arr, coh_name, where, what):
        arr[coh_name][where] = (np.append(arr[coh_name][where], what))
    
    def _append_all_measures(self, coherences_tmp, coh_name, coh_list):
        if (coh_name == 'focon'):
            self._measures_append(coherences_tmp, coh_name, 'res', coh_list)
            return
        
        self._measures_append(coherences_tmp, coh_name, 'mean-of-means', np.mean(coh_list['means']))
        self._measures_append(coherences_tmp, coh_name, 'median-of-means', np.median(coh_list['means']))
        if (coh_name == 'semantic'):
            return
        self._measures_append(coherences_tmp, coh_name, 'mean-of-medians', np.mean(coh_list['medians']))
        self._measures_append(coherences_tmp, coh_name, 'median-of-medians', np.median(coh_list['medians']))

