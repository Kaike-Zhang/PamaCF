import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, normal_


class MF(nn.Module):
    def __init__(self, config):
        super(MF, self).__init__()
        self.config = config
        self.user_emb = torch.nn.Embedding(self.config["n_users"], self.config["dim"])
        self.item_emb = torch.nn.Embedding(self.config["n_items"], self.config["dim"])
        normal_(self.user_emb.weight.data, std=0.1)
        normal_(self.item_emb.weight.data, std=0.1)

    def forward(self, user_list, pos_items, neg_items):
        user_emb = self.user_emb(torch.LongTensor(user_list).to(self.config["device"]))
        posI_emb = self.item_emb(torch.LongTensor(pos_items).to(self.config["device"]))
        negI_emb = self.item_emb(torch.LongTensor(neg_items).to(self.config["device"]))
        reg = (user_emb.norm(dim=1).pow(2).mean() + posI_emb.norm(dim=1).pow(2).mean() + negI_emb.norm(dim=1).pow(2).mean())

        return user_emb, posI_emb, negI_emb, reg
    
    def predict(self, user_list):
        user_emb = self.user_emb(torch.LongTensor(user_list).to(self.config["device"]))
        return torch.mm(user_emb, self.item_emb.weight.t())