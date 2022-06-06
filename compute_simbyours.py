# import ppi_prediction as ppi
# import evaluating as evlt
import matplotlib.pyplot as plt
import functools
from pygosemsim import term_set
from pygosemsim import similarity
from pygosemsim import annotation
from pygosemsim import graph
from pygosemsim import download
from pygosemsim import graph
import networkx as nx
from numpy import array, exp, mean, sqrt, dot
G = graph.from_resource("go-basic")
similarity.precalc_lower_bounds(G)

obo_file = 'pygosemsim/_resources/go-basic.obo'

fp=open(obo_file,'r')
obo_txt=fp.read()
fp.close()
#取[Term]和[Typedef]中间的字符信息
obo_txt=obo_txt[obo_txt.find("[Term]")-1:]
obo_txt=obo_txt[:obo_txt.find("[Typedef]")]
# obo_dict=parse_obo_txt(obo_txt)
id_namespace_dicts = {}
id_name_dicts = {}

for Term_txt in obo_txt.split("[Term]\n"):
    if not Term_txt.strip():
        continue
    name = ''
    ids = []
    for line in Term_txt.splitlines():
        if   line.startswith("id: "):
            ids.append(line[len("id: "):]) 
        elif line.startswith("name: "):
            name=line[len("name: ")]
        elif line.startswith("namespace: "):
             name_space=line[len("namespace: "):]
        elif line.startswith("alt_id: "):
            ids.append(line[len("alt_id: "):])
    
    for t_id in ids:
        id_namespace_dicts[t_id] = name_space
        id_name_dicts[t_id] = name
        
# get all used goid
obo_file = 'pygosemsim/_resources/go-basic.obo'

fp=open(obo_file,'r')
obo_txt=fp.read()
fp.close()
#取[Term]和[Typedef]中间的字符信息
obo_txt=obo_txt[obo_txt.find("[Term]")-1:]
obo_txt=obo_txt[:obo_txt.find("[Typedef]")]

alt_id2id = {}

for Term_txt in obo_txt.split("[Term]\n"):
    if not Term_txt.strip():
        continue
    name = ''
    ids = []
    for line in Term_txt.splitlines():
        if   line.startswith("id: "):
            ids.append(line[len("id: "):]) 
        
        elif line.startswith("alt_id: "):
            ids.append(line[len("alt_id: "):])
    if len(ids) > 1:
        for t_id in ids[1:]:
            alt_id2id[t_id] = ids[0]     

import pandas as pd
from tqdm import tqdm
from pygosemsim import term_set

def compute_bma_results(dataset_file_path, annot, method):
    compute_results = []
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    for index, row in tqdm(measure_df.iterrows()):
        p1 = row['Uniprot ID1']
        p2 = row['Uniprot ID2']
        if p1 not in annot.keys() or p2 not in annot.keys():
            compute_results.append(0)
        else:
            if name_space in ['biological_process', 'molecular_function', 'cellular_component']:
                trpv1 = [x for x in annot[p1]["annotation"].keys() if id_namespace_dicts[x] == name_space ]
                trpa1 = [x for x in annot[p2]["annotation"].keys() if id_namespace_dicts[x] == name_space ]
            else:
                trpv1 =   annot[p1]["annotation"].keys()  
                trpa1 =   annot[p2]["annotation"].keys()  
            sf = functools.partial(term_set.sim_func, G, method)
            sim_score = term_set.sim_bma(trpv1, trpa1, sf)
            if sim_score is None:
                sim_score = 0
            compute_results.append(sim_score)
    return compute_results
            
from sklearn.metrics.pairwise import cosine_similarity
 
def BMA(sent1, sent2):
    au1 = [max([cosine_similarity(s,t)[0][0] for t in sent2]) for s in sent1]
    au2 = [max([cosine_similarity(s,t)[0][0] for t in sent1]) for s in sent2]
    data = round((mean(au1)+mean(au2))/2.0,5)
    return data
 
        

    
def extrtactembedding( dataset_file_path,anno_dict, go_name_prefix, num_proj_hidden=256):
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    all_protein = list(measure_df['Uniprot ID1']) + list(measure_df['Uniprot ID2'])
    all_protein = list(set(all_protein))
    prot_emb_dict = {}
    for prot_id in  all_protein:
        embedding = []
        if prot_id in anno_dict.keys():
            all_gos = list(anno_dict[prot_id]["annotation"].keys())
            
            
            if name_space in ['biological_process', 'molecular_function', 'cellular_component']:
                all_gos = [x for x in all_gos if id_namespace_dicts[x] == name_space ]
                 

            for go in all_gos:
                go_emb = np.load(go_name_prefix+go+'.npy')
                embedding.append(go_emb )
                
            if embedding == []:
                embedding.append( np.zeros((1, num_proj_hidden)))
#                 print(prot_id,all_gos)
            prot_emb_dict[prot_id] = embedding
        else:
            embedding.append( np.zeros((1, num_proj_hidden)))
            prot_emb_dict[prot_id] = embedding
    return prot_emb_dict


import numpy as np
import joblib
import multiprocessing as mp

def bma_score(dataset_file_path,anno_dict,prot_emb_dict, num_proj_hidden=256):
     
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    bert_scores = []
    pool = mp.Pool(mp.cpu_count())
    
    bert_scores = pool.starmap(BMA, [(prot_emb_dict[row['Uniprot ID1']], prot_emb_dict[row['Uniprot ID2']]) for index, row in measure_df.iterrows()])

    pool.close()  
    return bert_scores


def extrtactgraphembedding( dataset_file_path,anno_dict, nodes_emb_dict, num_proj_hidden=256):
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    all_protein = list(measure_df['Uniprot ID1']) + list(measure_df['Uniprot ID2'])
    all_protein = list(set(all_protein))
    prot_emb_dict = {}
    for prot_id in  all_protein:
        embedding = []
        if prot_id in anno_dict.keys():
            all_gos = list(anno_dict[prot_id]["annotation"].keys())
            if name_space in ['biological_process', 'molecular_function', 'cellular_component']:
                all_gos = [x for x in all_gos if id_namespace_dicts[x] == name_space ]
              
            for go in all_gos:
                
                if go in nodes_emb_dict.keys():
                    embedding.append(nodes_emb_dict[go].reshape(1,-1) )
                elif go in alt_id2id.keys():
                    embedding.append(nodes_emb_dict[alt_id2id[go]].reshape(1,-1) )
                
              
                
                
            if embedding == []:
                embedding.append( np.zeros((1, num_proj_hidden)))
#                 print(prot_id,all_gos)
            prot_emb_dict[prot_id] = embedding
        else:
            embedding.append( np.zeros((1, num_proj_hidden)))
            prot_emb_dict[prot_id] = embedding
    return prot_emb_dict

import evaluating as evlt
import ppi_prediction as ppi

name_spaces = ['biological_process', 'molecular_function', 'cellular_component', 'all']  # bp, mf, cc, all
name_space = name_spaces[2]
num_proj_hidden = 1024
# node_emb_dict = joblib.load('kgsim-benchmark/GO/NLayerDeeperGCN_woft_tsdae_avg_all_512_16layer20000_'+name_space)

# node_emb_dict = joblib.load('kgsim-benchmark/GO/gin_woft_tsdae_avg_all_512_12layer10000_'+name_space)

node_emb_dict = joblib.load('kgsim-benchmark/GO/GRACE_4Layers_namedef'+name_space)


def compute_score4dataset(dataset, dataset_type):
    if 'HS' in dataset:
        annot = annotation.from_resource("goa_human")
        dataset_file_path = 'kgsim-benchmark/DataSets/'+dataset+'.csv'
        print('HS')
    elif 'EC' in dataset:
        annot = annotation.from_resource("ecocyc")
        dataset_file_path = 'kgsim-benchmark/DataSets/'+dataset+'.csv'
        print('EC')
    elif 'SC' in dataset:
        annot = annotation.from_resource("goa_yeast")
        dataset_file_path = 'kgsim-benchmark/DataSets/'+dataset+'.csv'
        print('SC')
    elif 'DM' in dataset:
        annot = annotation.from_resource("goa_fly")
        dataset_file_path = 'kgsim-benchmark/DataSets/'+dataset+'.csv'
        print('DM')
    
    prot_emb_dict = extrtactgraphembedding(dataset_file_path,annot, node_emb_dict,num_proj_hidden)

    bma_gcn_scores = bma_score(dataset_file_path,annot, prot_emb_dict,num_proj_hidden)
    return bma_gcn_scores

    
datasets = ['PPI_EC3',  'PPI_EC1', 'PPI_DM3', 'PPI_DM1', 'PPI_HS3', 'PPI_HS1', 'PPI_SC3', 'PPI_SC1']
dataset_type = 'PPI'
ours_similarity = {}
for dataset in datasets:
    bma_gcn_scores = compute_score4dataset(dataset, dataset_type)
    ours_similarity[dataset] = bma_gcn_scores



datasets = ['MF_EC3',  'MF_EC1', 'MF_DM3', 'MF_DM1', 'MF_HS3', 'MF_HS1', 'MF_SC3', 'MF_SC1']
dataset_type = 'MF'

for dataset in datasets:
    bma_gcn_scores = compute_score4dataset(dataset, dataset_type)
    ours_similarity[dataset] = bma_gcn_scores


# # # # # go_name_prefix =  'kgsim-benchmark/DataSets/ouBioBERT_wo_traing_allgo'

node_emb_dict = joblib.load('kgsim-benchmark/DataSets/sbert_tsdae_avg_allgo_namedef')

num_proj_hidden = 768

datasets = ['PPI_EC3',  'PPI_EC1', 'PPI_DM3', 'PPI_DM1', 'PPI_HS3', 'PPI_HS1', 'PPI_SC3', 'PPI_SC1']
dataset_type = 'PPI'
tsdae_similarity = {}
for dataset in datasets:
    bma_gcn_scores = compute_score4dataset(dataset, dataset_type)
    tsdae_similarity[dataset] = bma_gcn_scores
    
    
datasets = ['MF_EC3',  'MF_EC1', 'MF_DM3', 'MF_DM1', 'MF_HS3', 'MF_HS1', 'MF_SC3', 'MF_SC1']
dataset_type = 'MF'

for dataset in datasets:
    bma_gcn_scores = compute_score4dataset(dataset, dataset_type)
    tsdae_similarity[dataset] = bma_gcn_scores

node_emb_dict = joblib.load('kgsim-benchmark/DataSets/ouBioBERT_wo_traing_allgo_namedef')

num_proj_hidden = 768


oubiobert_similarity = {}
datasets = ['PPI_EC3',  'PPI_EC1', 'PPI_DM3', 'PPI_DM1', 'PPI_HS3', 'PPI_HS1', 'PPI_SC3', 'PPI_SC1']
dataset_type = 'PPI'

for dataset in datasets:
    bma_gcn_scores = compute_score4dataset(dataset, dataset_type)
    oubiobert_similarity[dataset] = bma_gcn_scores
    
datasets = ['MF_EC3',  'MF_EC1', 'MF_DM3', 'MF_DM1', 'MF_HS3', 'MF_HS1', 'MF_SC3', 'MF_SC1']
dataset_type = 'MF'

for dataset in datasets:
    bma_gcn_scores = compute_score4dataset(dataset, dataset_type)
    oubiobert_similarity[dataset] = bma_gcn_scores

joblib.dump(oubiobert_similarity, '/home/junjie/OntoGraphEMB/bib_exp0512/results/oubiobert-'+name_space)
joblib.dump(tsdae_similarity, '/home/junjie/OntoGraphEMB/bib_exp0512/results/tsdae-'+name_space)
joblib.dump(ours_similarity, '/home/junjie/OntoGraphEMB/bib_exp0512/results/gt2vec-'+name_space)


