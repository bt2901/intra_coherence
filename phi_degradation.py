import artm, tqdm, os
import numpy as np
import pandas as pd


from document_helper import pn_folder, vw_folder
from measures_utils import ResultStorage, record_results    

uniform = False

def damage_model(model, topics_list, alpha, seed=42):
    topic_model_data, phi_numpy_matrix = model.master.attach_model('pwt')
    np.random.seed(seed)
    if uniform:
        random_init = np.random.random_sample(phi_numpy_matrix.shape)
        random_init /= np.sum(random_init, axis=0)
    else:
        random_init = np.zeros(phi_numpy_matrix.shape)
        N = phi_numpy_matrix.shape[0]
        for i in range(20):
            random_init[:, i] = np.random.dirichlet(alpha=[0.01]*N, size=1)

    damaged = (1-alpha) * phi_numpy_matrix + alpha * random_init 
    for topic in topics_list:
        phi_numpy_matrix[:, topic] = damaged[:, topic] 
    return model, topic_model_data, phi_numpy_matrix

def refresh_model(model, batch_vectorizer):
    model.master.clear_score_array_cache()
    model.master.clear_score_cache()
    return model.transform(batch_vectorizer=batch_vectorizer)

def write_topics_coherence(model, alpha, results_df):
    for topic_id in model.topic_names:
        name = "TopTokens_for_" + topic_id
        coh = model.get_score(name).average_coherence
        #print ("{}: {}".format(topic_id, coh))
        results_df.loc[alpha, topic_id] = coh

def create_model(dictionary, num_tokens, num_document_passes):

    specific_topics = ['topic {}'.format(i) for i in range(20)]
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

def load_saved_model(batch_vectorizer):
    pp = r"C:\Development\Github\intratext_fixes\sgm200\good_default_model_20_t\result_good_default_model_20_t_phi.txt"
    phi_good_pd = pd.read_csv(pp, index_col=0, encoding="utf8")

    model = create_model(
        dictionary=dictionary,
        num_tokens=10,
        num_document_passes=1
    )
    phi_pd = model.get_phi()
    for word in phi_pd.index:
        if word in phi_good_pd.index:
            phi_pd.loc[word] = np.copy(phi_good_pd.loc[word])
        else:
            phi_pd.loc[word] = np.ones(phi_pd.shape[1]) * 1e-12
    phi_reborn_pd = phi_pd
    phi_reborn_matrix = phi_reborn_pd.as_matrix() / phi_reborn_pd.as_matrix().sum(axis=0)
    topic_model_data, phi_numpy_matrix = model.master.attach_model('pwt')
    np.copyto(phi_numpy_matrix, phi_reborn_matrix)
    print model.get_theta()
    return model, topic_model_data, phi_numpy_matrix

if __name__ == "__main__":
    pn_folder = r'C:\Development\Github\news_analysis\plaintext200'

    dm = "plain_text"
    coh_names = ['newman', 'mimno', 'semantic', 'toplen', "focon"]
    coh_names = ['toplen']
    coh_names = ['semantic']
    intra_coherence_params = {
        "window": 10, "threshold": 0.02, "focon_threshold": 5, "cosine_num_top_tokens": 10, "num_top_tokens": 10,
        "general_penalty": 0.005
    }

    dictionary = artm.Dictionary(dictionary_path=r"{}\dict.dict".format(pn_folder), name="PN-DICT-COOC")
    batch_vectorizer = artm.BatchVectorizer(data_path=pn_folder, data_format='batches')

    good_model = artm.load_artm_model(r'C:\Development\Github\news_analysis\plaintext200\good_default_model_20_t\dump')
    good_model.regularizers.data.clear()
    #good_model, _, _ = load_saved_model(batch_vectorizer)

    theta = good_model.transform(batch_vectorizer=batch_vectorizer)
    vw_file = os.path.join(r"C:\Development\Github\intratext_fixes", "sgm200", 'vw_bimodal.txt')
    for topic_id in good_model.topic_names:
        name = "TopTokens_for_" + topic_id
        s = artm.TopTokensScore(name=name, num_tokens=10, class_id=dm, dictionary=dictionary, topic_names=[topic_id])
        good_model.scores.add(s)
                    
    step = 0.05
    results = pd.DataFrame(columns=good_model.topic_names,
                           index=np.arange(0, 1.01, step))

    data_storage = ResultStorage(coh_names, domain_path=None)
    for alpha in tqdm.tqdm(np.arange(0, 1.01, step)):
        model = good_model.clone()
        model, _, _ = damage_model(model, range(20), alpha=alpha, seed=1)
        theta = refresh_model(model, batch_vectorizer)
        model_id = " alpha_{} ".format(alpha)
        with record_results(model=model, vw_file=vw_file, at=model_id, save_in=data_storage, theta=theta) as recorder:
            for coh_name in coh_names:
                print coh_name
                recorder.evaluate(coh_name, intra_coherence_params)
                    
            #recorder.evaluate_segmentation_quality()
        write_topics_coherence(model, alpha, results)
            
    data_storage.data_results_save()
    #results.to_pickle("phi_degradation_dirichlet_newman.pkl") 
    #results.to_pickle("phi_degradation_uniform_newman.pkl") 
    results.to_pickle("phi_degradation_dirichlet_newman.pkl") 
    #results_toplen.to_pickle("phi_degradation_dirichlet_toplen.pkl") 
    #print results







