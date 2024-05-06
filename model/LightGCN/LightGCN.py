import numpy as np
import scipy.sparse as sp
# import dgl
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, normal_


class LightGCN(nn.Module):
    def __init__(self, config, graph):
        super(LightGCN, self).__init__()
        self.config = config
        self.embedding_user = torch.nn.Embedding(self.config["n_users"], self.config["dim"])
        self.embedding_item = torch.nn.Embedding(self.config["n_items"], self.config["dim"])
        self.norm_adj = graph.to(self.config["device"])
        self.n_layers = self.config["n_layers"] 
        self.device = self.config["device"]
        normal_(self.embedding_user.weight.data, std=0.1)
        normal_(self.embedding_item.weight.data, std=0.1)

        self.f = nn.Sigmoid()
    

    def _get_rep(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        representations = torch.cat([users_emb, items_emb])
        all_layer_rep = [representations]
        # row, column = self.norm_adj.indices()
        # g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            # representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            representations = torch.sparse.mm(self.norm_adj, representations)
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        users, items = torch.split(final_rep, [self.config["n_users"], self.config["n_items"]])
        return users, items

    def forward(self, user_list, pos_items, neg_items):
        user_r, item_r = self._get_rep()
        user_emb = self.embedding_user(torch.LongTensor(user_list).to(self.config["device"]))
        posI_emb = self.embedding_item(torch.LongTensor(pos_items).to(self.config["device"]))
        negI_emb = self.embedding_item(torch.LongTensor(neg_items).to(self.config["device"]))
        reg = (user_emb.norm(dim=1).pow(2).mean() + posI_emb.norm(dim=1).pow(2).mean() + negI_emb.norm(dim=1).pow(2).mean())
        return user_r[torch.LongTensor(user_list).to(self.config["device"]), :], \
               item_r[torch.LongTensor(pos_items).to(self.config["device"]), :], \
               item_r[torch.LongTensor(neg_items).to(self.config["device"]), :], reg
    
    def predict(self, user_list):
        user_r, item_r = self._get_rep()
        user_emb = user_r[torch.LongTensor(user_list).to(self.config["device"]), :]
        item_emb = item_r
        return self.f(torch.matmul(user_emb, item_emb.t()))

    def _generate_graph(self, adj_mat):
        degree = np.array(np.sum(adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = self._get_sparse_tensor(norm_adj)
        return norm_adj
    
    def _get_sparse_tensor(self, mat):
        coo = mat.tocoo()
        indexes = np.stack([coo.row, coo.col], axis=0)
        indexes = torch.tensor(indexes, dtype=torch.int64, device=self.config["device"])
        data = torch.tensor(coo.data, dtype=torch.float32, device=self.config["device"])
        sp_tensor = torch.sparse.FloatTensor(indexes, data, torch.Size(coo.shape)).coalesce()
        return sp_tensor
