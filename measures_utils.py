# coding: utf-8
from __future__ import print_function
import copy
import traceback

import numpy as np

import codecs
from collections import defaultdict

from document_helper import files_total, data_results_save
from segmentation import segmentation_evaluation

from coherences import coh_newman, coh_mimno #, #coh_cosine
from intra_coherence import coh_semantic, coh_toplen, coh_focon
    

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
        
        # TODO: model_id -> (measure_id -> val) 
        self.segm_quality = {s: dict() for s in self.segm_modes}
        # TODO: model_id -> (measure_id -> val) 
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
        pars_segm = self.segm_quality
        pars_coh = self.coherences
        if (len(pars_segm) != len(pars_coh)):
            print(pars_segm.keys())
            print(pars_coh.keys())
            print(pars_segm["soft"].keys())
            print(pars_coh["mimno"].keys())
            raise ValueError('Different lengths of x- and y- arrays ({} and {})'.format(len(pars_segm), len(pars_coh)))
            #print('Different lengths of x- and y- arrays ({} and {})'.format(len(pars_segm), len(pars_coh)))
        #print(pars_segm)
        #print(pars_coh)

        coh_names = ['newman', 'mimno', 'cosine', 'semantic', 'toplen']
        averaging_types = ['mean-of-means', 'mean-of-medians',
                           'median-of-means', 'median-of-medians']
        corrs = {prs: 'prs', spr: 'spr'}
        
        rows = []
        data = ''
        
        for pair in pars_segm:
            #window, threshold = pair[0], pair[1]
            window, threshold = None, None
            rows += [['window={0}, threshold={1}'.format(window, threshold)]]

            for segm_type in ['soft', 'harsh']:
                rows += [[segm_type]]
                rows += [[
                    '',
                    'Newman', '', '', '',
                    'Mimno', '', '', '',
                    'Cosine', '', '', '',
                    'SemantiC', '',
                    'TopLen', '', '', ''
                ]]
                rows += [[
                    '',
                    '1', '2', '3', '4',
                    '1', '2', '3', '4',
                    '1', '2', '3', '4',
                    '1', '3',
                    '1', '2', '3', '4'
                ]]

                for corr in corrs:
                    row = [corrs[corr]]
                    for coh in coh_names:
                        if (coh == 'semantic'):
                            for av_type in ['mean-of-means', 'median-of-means']:
                                x = pars_segm[pair][segm_type]
                                y = pars_coh[pair][coh][av_type]
                                x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                                y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                                x = x_tmp
                                y = y_tmp
                                row += ["'{0:.2f}".format(corr(x, y))]
                            continue

                        for av_type in averaging_types:
                            x = pars_segm[pair][segm_type]
                            y = pars_coh[pair][coh][av_type]
                            x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                            y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                            x = x_tmp
                            y = y_tmp
                            row += ["'{0:.2f}".format(corr(x, y))]

                    rows += [row]
                    
            rows += [['']]
            
        for pair in pars_segm:
            #window, threshold = pair[0], pair[1]
            window, threshold = None, None
            data += 'window={0}, threshold={1}\n'.format(window, threshold)

            for segm_type in ['soft', 'harsh']:
                data += segm_type + '\n'
                
                x = pars_segm[pair][segm_type]
                x = sorted(x)
                data = data_append(data, x)

                for coh in coh_names:
                    if (coh == 'semantic'):
                        data += 'SemantiC\n'
                        for av_type in ['mean-of-means', 'median-of-means']:
                            x = pars_segm[pair][segm_type]
                            y = pars_coh[pair][coh][av_type]
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
                        x = pars_segm[pair][segm_type]
                        y = pars_coh[pair][coh][av_type]
                        x_tmp = np.array([u for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        y_tmp = np.array([v for (u, v) in sorted(zip(x, y), key=lambda pair: pair[0])])
                        x = x_tmp
                        y = y_tmp
                        data = data_append(data, y)
            
            data += '\n'
        
        data = data.strip()

        with open(os.path.join('results', 'results.csv'), 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile,
                                   delimiter=',', quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerows(rows)
        
        with open(os.path.join('results', 'data.txt'), 'a', encoding='utf-8') as f:
            f.write(data)
        
        data = ''
        data += 'window, threshold:\n'
        for pair in pars_segm:
            data += '({0}, {1}), '.format(pair[0], pair[1])
        data = data[:-2]
        data += '\n\n'
        
        data += 'pars_segm:\n{0}\n'.format(pars_segm)
        data += '\n\n\n'
        data += 'pars_coh:\n{0}\n'.format(pars_coh)
        data = data.strip()
        
        with open(os.path.join('results', 'data_raw.txt'), 'a', encoding='utf-8') as f:
            f.write(data)
    
        raise NotImplementedError


functions_data = {name: {"func": func, "by_top_tokens": (name in coh_names_top_tokens)} for name, func in zip(coh_names, coh_funcs)}
class record_results(object):
    def __init__(self, model, vw_file, at, save_in):
        self.save_in = save_in
        self.vw_file = vw_file
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
        with codecs.open(self.vw_file, "r", encoding="utf8") as f:
            if (coh_name in coh_names_top_tokens):
                if len(self.model.score_tracker['TopTokensScore'].last_tokens) == 0:
                    print("WARNING: top tokens is empty")
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
                coh_list = coh_func(
                    coh_params, self.model.topic_names, f, 
                    phi_val=self.phi.values, phi_cols=self.phi_cols, phi_rows=self.phi_rows,
                    theta_val=self.theta.values, theta_cols=self.theta_cols, theta_rows=self.theta_rows,
                )

        self._append_all_measures(self._coherences_tmp, coh_name, coh_list)
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

