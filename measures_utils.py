# coding: utf-8
from __future__ import print_function
import copy
import traceback

import numpy as np
from scipy import stats 
import codecs, os, io
import csv
from collections import defaultdict

import pandas as pd

from document_helper import debug
from document_helper import calc_doc_ptdw, read_plaintext
from segmentation import segmentation_evaluation, output_detailed_cost

from coherences import coh_newman, coh_mimno #, #coh_cosine
from intra_coherence import coh_toplen_calculator, coh_focon_calculator, coh_semantic_calculator
from intra_coherence_legacy import coh_semantic, coh_toplen, coh_focon

import time

#debug = True
#debug = not True

def prs(l1, l2):
    return stats.pearsonr(l1, l2)[0]

def spr(l1, l2):
    return stats.spearmanr(l1, l2)[0]

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
        self.averaging_types = ['mean-of-means', 'mean-of-medians',
                           'median-of-means', 'median-of-medians']
        
        # TODO: model_id -> (measure_id -> val) 
        self.segm_quality = defaultdict(dict)
        # TODO: model_id -> (measure_id -> val) 
        self.coherences = defaultdict(lambda: defaultdict(dict))
        
        data_model = {"model_id": [], ("segm", "soft"): [], ("segm", "harsh"): []}
        for name in self.coh_names:
            for mode in self.averaging_types:
                data_model[(name, mode)] = []
        self.measures = pd.DataFrame(data_model)
        
    def save_segm(self, model_id, segm_quality_tmp):
        for s in self.segm_modes:
            self.segm_quality[model_id][s] = (
                          np.mean(segm_quality_tmp[s])
            )

    def save_coh(self, model_id, coherences_tmp):
        for name in self.coh_names:
            for mode in coherences_tmp[name]:
                self.coherences[model_id][name][mode] = np.mean(coherences_tmp[name][mode])
    def save2df(self, model_id):
        row = {"model_id": model_id}
        for s in self.segm_modes:
            row[("segm", s)] = self.segm_quality[model_id][s]
        for name in self.coh_names:
            for mode in self.averaging_types:
                row[(name, mode)] = self.coherences[model_id][name].get(mode, float("nan"))
        self.measures = self.measures.append(row, ignore_index=True)

    

    def data_results_save(self):
        pars_segm = self.segm_quality
        pars_coh = self.coherences
        
        if (len(pars_segm) != len(pars_coh)):
            print(pars_segm.keys())
            print(pars_coh.keys())
            raise ValueError('Different lengths of x- and y- arrays ({} and {})'.format(len(pars_segm), len(pars_coh)))

        coh_names = ['newman', 'mimno', 'semantic', 'toplen']
        corrs = {prs: 'prs', spr: 'spr'}
        
        # last_row = df.index[:-1]
        # corr_df = self.measures.corr("spearman").loc[(('segm', "soft"), ('segm', "harsh")), last_row]
        print("corred")
        self.measures.corr("spearman").to_csv("corred.csv")
        # TODO FIXME DEPRECATED
        corr_df = self.measures.corr("spearman").ix[(('segm', "soft"), ('segm', "harsh")), :-1]
        print(self.measures.head())
        self.measures.to_csv(r"C:\Development\Github\intratext_fixes\results\m.csv", sep=";", encoding='utf-8')
        print(corr_df.head())
        corr_df.to_csv(r"C:\Development\Github\intratext_fixes\results\corr.csv", sep=";", encoding='utf-8')
        
    


functions_data = {name: {"func": func, "by_top_tokens": (name in coh_names_top_tokens)} for name, func in zip(coh_names, coh_funcs)}
functions_data["focon"]["calc"] = coh_focon_calculator
functions_data["toplen"]["calc"] = coh_toplen_calculator
functions_data["semantic"]["calc"] = coh_semantic_calculator

class record_results(object):
    def __init__(self, model, vw_file, at, save_in, theta=None):
        self.save_in = save_in
        self.vw_file = vw_file
        self.at = at
        self.model = model
        if theta is not None:
            self.theta = theta
    def __enter__(self):
        self.phi = self.model.get_phi()
        if self.theta is None:
            self.theta = self.model.get_theta()
            #self.theta = self.model.get_phi(model_name="ptd")
        
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
        self.save_in.save2df(self.at)

    def evaluate(self, coh_name, coh_params):
        #raise NotImplementedError
        coh_func = functions_data[coh_name]["func"]
        coh_calculator = functions_data[coh_name].get("calc", None)

        phi_sort = np.argsort(self.phi_rows)

        with codecs.open(self.vw_file, "r", encoding="utf8") as f:
            if (coh_name in coh_names_top_tokens):
                should_skip = debug or 'TopTokensScore' not in self.model.score_tracker or len(self.model.score_tracker['TopTokensScore'].last_tokens) == 0
                if should_skip:
                    if not debug:
                        print("WARNING: top tokens is empty")
                    else:
                        print("skipped...")
                    res_shape = (len(self.model.topic_names) - 1,)
                    coh_list = {'means': np.full(res_shape, np.nan),
                            'medians': np.full(res_shape, np.nan)}
                else:
                    (window, num_top_tokens) = coh_params["window"], coh_params["num_top_tokens"]
                    coh_list = coh_func(
                        window=window, num_top_tokens=num_top_tokens,
                        model=self.model, topics=self.model.topic_names,
                        file=f
                    )
            else:
                time_ptdw = 0
                time_coh = 0
                
                m = coh_calculator(coh_params, self.model.topic_names)
                for i, line in enumerate(f):
                    if debug:
                        if i % 100:
                            continue
                    doc_num, data = read_plaintext(line)
                    
                    t0 = time.time()
                    doc_ptdw = calc_doc_ptdw(data, doc_num, 
                        phi_val=self.phi.values, phi_rows=self.phi_rows, phi_sort=phi_sort, 
                        theta_val=self.theta.values, theta_cols=self.theta_cols
                    )
                    time_ptdw += time.time() - t0

                    t0 = time.time()
                    m.update(doc_num, data, doc_ptdw, self.phi.values, self.phi_rows)
                    time_coh += time.time() - t0

                coh_list = m.output()
                filename = "details_of_{}_{}.csv".format(coh_name, self.at)
                filename = ''.join(char for char in filename 
                    if char in "_.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                m.output_details(os.path.join('results', filename))
                #print ("timings: p_tdw {} seconds, coh {} seconds".format(time_ptdw, time_coh))

        self._append_all_measures(self._coherences_tmp, coh_name, coh_list)
        
    def unit_test(self, coh_name, coh_params):
        coh_func = functions_data[coh_name]["func"]
        coh_calculator = functions_data[coh_name]["calc"]
        m = coh_calculator(coh_params, self.model.topic_names)
        phi_sort = np.argsort(self.phi_rows)
        with codecs.open(self.vw_file, "r", encoding="utf8") as f:
            for line in f:
                doc_num, data = read_plaintext(line)
                
                doc_ptdw = calc_doc_ptdw(data, doc_num, 
                    phi_val=self.phi.values, phi_rows=self.phi_rows, phi_sort=phi_sort, 
                    theta_val=self.theta.values, theta_cols=self.theta_cols
                )
                m.update(doc_num, data, doc_ptdw, self.phi.values, self.phi_rows)
            coh_list = m.output()
        with codecs.open(self.vw_file, "r", encoding="utf8") as f2:
            coh_list2 = coh_func(
                coh_params, self.model.topic_names, f2, 
                phi_val=self.phi.values, phi_cols=self.phi_cols, phi_rows=self.phi_rows,
                theta_val=self.theta.values, theta_cols=self.theta_cols, theta_rows=self.theta_rows,
            )
        are_equal = np.allclose(coh_list2['means'], coh_list['means'], equal_nan=True) and np.allclose(coh_list2['medians'], coh_list['medians'], equal_nan=True)
        if are_equal:
            print("OK")
        else:
            print("ERROR")
            print (coh_list)
            print (coh_list2)
            raise NotImplementedError
                        
    def evaluate_segmentation_quality(self):
        with codecs.open(self.vw_file, "r", encoding="utf8") as f:
            cur_segm_eval, indexes = (
                segmentation_evaluation(
                    topics=self.model.topic_names, f=f,
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
        with codecs.open(self.vw_file, "r", encoding="utf8") as f:
            filename = "details_of_segm_{}_{}".format("segm", self.at)
            filename = ''.join(char for char in filename 
                if char in "_.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            filename = (os.path.join('results', filename + "_{}.csv"))
            output_detailed_cost(
                topics=self.model.topic_names, f=f,
                phi_val=self.phi.values, phi_cols=self.phi_cols, phi_rows=self.phi_rows,
                theta_val=self.theta.values, theta_cols=self.theta_cols, theta_rows=self.theta_rows,
                indexes=indexes, filename=filename
            )
            
    def _create_segm_quality_carcass(self):
        segm_quality_carcass = {mode: np.array([]) for mode in ["soft", "harsh"]}
        return segm_quality_carcass
    def _create_coherences_carcass(self, coh_names):
        coherences_carcass = defaultdict(dict)
        for cn in coh_names:
            for mode in ['mean-of-means', 'mean-of-medians', 'median-of-means', 'median-of-medians']:
                coherences_carcass[cn][mode] = np.array([])
                
        return coherences_carcass

    def _measures_append(self, arr, coh_name, where, what):
        arr[coh_name][where] = (np.append(arr[coh_name][where], what))
    
    def _append_all_measures(self, coherences_tmp, coh_name, coh_list):
        if (coh_name == 'focon'):
            self._measures_append(coherences_tmp, coh_name, 'mean-of-means', coh_list)
            return
        
        self._measures_append(coherences_tmp, coh_name, 'mean-of-means', np.mean(coh_list['means']))
        self._measures_append(coherences_tmp, coh_name, 'median-of-means', np.median(coh_list['means']))
        if (coh_name == 'semantic'):
            return
        self._measures_append(coherences_tmp, coh_name, 'mean-of-medians', np.mean(coh_list['medians']))
        self._measures_append(coherences_tmp, coh_name, 'median-of-medians', np.median(coh_list['medians']))

