


pn_folder = 'PNauka'
pn_folder = r'C:\sci_reborn\synth_coherence\PNauka'
vw_folder = 'pn_mixed_lemmatized_full'

domain_folder = 'PNaukaMixedLemmatized_short'
domain_path = os.path.join(pn_folder, domain_folder)
files_total = os.listdir(domain_path)
files_total = sorted(files_total, key=my_sort_func)

# ===========


    
    
    
    
    

#==============================




    

coh_names = ['newman', 'mimno', 'cosine', 'semantic', 'toplen']
coh_names_top_tokens = ['newman', 'mimno', 'cosine']
coh_funcs = [coh_newman, coh_mimno, coh_cosine, coh_semantic, coh_toplen]


# assume that items in pars_segm and pars_coh
# are in the same order
def data_results_save(pars_segm, pars_coh):
    if (len(pars_segm) != len(pars_coh)):
        print('Different lengths of x- and y- arrays')
        return

    coh_names = ['newman', 'mimno', 'cosine', 'semantic', 'toplen']
    averaging_types = ['mean-of-means', 'mean-of-medians',
                       'median-of-means', 'median-of-medians']
    corrs = {prs: 'prs', spr: 'spr'}
    
    rows = []
    data = ''
    
    for pair in pars_segm:
        window, threshold = pair[0], pair[1]
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
        window, threshold = pair[0], pair[1]
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
