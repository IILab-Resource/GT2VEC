
import networkx as nx
import torch
import obonet
from tqdm import tqdm
import pandas as pd
from optim import AdamW, PlainRAdam, RAdam




go_file = 'kgsim-benchmark/GO/go-basic.obo'
go_graph = obonet.read_obo(go_file)
id_to_name = {id_: data.get('name') for id_, data in go_graph.nodes(data=True)} ## by default obsolete already removed
id_to_def = {id_: data.get('def').split('"')[1] for id_, data in go_graph.nodes(data=True)} ## by default obsolete already removed

# convert to plain old digraph
go_graph = nx.DiGraph(go_graph)

go_branch = 'p'
namespaces = {'p': 'biological_process', 'c': 'cellular_component', 'f': 'molecular_function'}

nodes_branch = (n for n in go_graph if go_graph._node[n]['namespace'] == namespaces[go_branch])
go_graph_branch = go_graph.subgraph(nodes_branch)

# remove node attributes by serializing to adjacency matrix
edgelist = nx.to_edgelist(go_graph_branch)
go_graph = nx.DiGraph(edgelist)
go_graph = go_graph.to_undirected()
# reverse the direction of edges so that root nodes have highest degree
# go_graph = go_graph.reverse()
    
   
    # convert to pytorch-geomtric dataset
from torch_geometric.utils import from_networkx
    
data = from_networkx(go_graph)
nodes = list(go_graph.nodes())


# In[3]:


data.edge_index.size()


# In[4]:


from sentence_transformers import SentenceTransformer, SentencesDataset, losses, InputExample,models
from torch import nn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
 
import joblib


# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 

device = torch.device("cpu")

sbert  = SentenceTransformer('sbert/data/tsdae-model-avg' , device=device)
x_sent = [id_to_name[nodes[i]] + ' ' + id_to_def[nodes[i]] for i in range(len(nodes))]
nodes_len = len(nodes)
x = torch.zeros(( nodes_len, 768))

batch_s = 2048*10

for i in   range(0,nodes_len,batch_s):
    if i+1 > nodes_len:
        start = i
        end  =  nodes_len
    else:
        start = i
        end  =  (i+1) * batch_s
    x[start:end] = sbert.encode( x_sent[start:end], convert_to_tensor=True)  



import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import BootstrapContrast
from torch_geometric.nn import GCNConv,GATConv,global_add_pool,GINConv

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.PReLU(), nn.Linear(out_dim, out_dim)))

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.activation = torch.nn.PReLU()
        

    def forward(self, x, edge_index, batch):
        z = x
        
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
           
        return z

# class GConv(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, activation, num_layers):
#         super(GConv, self).__init__()
#         self.activation = activation()
#         self.layers = torch.nn.ModuleList()
#         self.layers.append(GCNConv(input_dim, hidden_dim))
#         for _ in range(num_layers - 1):
#             self.layers.append(GCNConv(hidden_dim, hidden_dim ))
#         self.norm = torch.nn.LayerNorm(hidden_dim)
        
        
#     def forward(self, x, edge_index, edge_weight=None):
#         z = x
#         for i, conv in enumerate(self.layers):
#             z = conv(z, edge_index, edge_weight)
#             z = self.activation(z)
#             z = F.dropout(z, p=0.2, training=self.training)
#         z = self.norm(z)
#         return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, x,  edge_index, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model( x,  edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()

 

from GCL.models import DualBranchContrast

# aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)  ])
# aug2 = A.Compose([A.EdgeRemoving(pe=0.1), A.FeatureMasking(pf=0.2)  ])
aug1 = A.Compose([A.EdgeRemoving(pe=0.3),  A.FeatureDropout(pf=0.2)])
aug2 = A.Compose([A.EdgeRemoving(pe=0.2),   A.FeatureDropout(pf=0.3)])

hidden_dim = 1024
gconv = GConv(input_dim=768, hidden_dim=hidden_dim,  num_layers= 2).to(device)

# gconv = GConv(input_dim=768, hidden_dim=hidden_dim,activation=torch.nn.PReLU, num_layers= 2).to(device)
encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=2048).to(device)
contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.1), mode='L2L', intraview_negs=True).to(device)
# contrast_model = DualBranchContrast(loss=L.BootstrapLatent(), mode='L2L', intraview_negs=True).to(device)

# optimizer = Adam(encoder_model.parameters(), lr=0.0005)
optimizer = AdamW(encoder_model.parameters(), lr=0.0005)


epochs = 1000
with tqdm(total=epochs, desc='(T)') as pbar:
    for epoch in range(1, epochs+1):
        loss = train(encoder_model, contrast_model, x.to(device), data.edge_index.to(device), optimizer)
        pbar.set_postfix({'loss': loss})
        pbar.update()


encoder_model.eval()
z, _, _ = encoder_model(x.to(device), data.edge_index.to(device))
z = z.cpu().detach().numpy() 

import joblib
node_emb_dict = {}
for i, node_name in  enumerate(nodes):
    node_emb_dict[node_name] = z[i]
joblib.dump( node_emb_dict,  'kgsim-benchmark/GO/GRACE_4Layers_namedef'+namespaces[go_branch])
    
    
    
    