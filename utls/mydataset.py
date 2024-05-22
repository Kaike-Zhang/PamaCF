import csv
import json
import os
import random
import numpy as np
import torch
import pickle
from datetime import datetime
from collections import defaultdict, Counter
from torch.utils.data import Dataset



class BasicDataset(Dataset):
    def __init__(self, path, config) -> None:
        super().__init__()
        self.path = path
        self.config = config

        self.load_data()

    def __len__(self):
        return NotImplementedError
    
    def __getitem__(self, index):
        return index

    def _load_org_data(self):
        '''
        data format:
        User ID; Item ID; Time; Score;
        '''
        with open(os.path.join(self.path, "data.txt"), 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            data = list(reader)

        while True:
            # Count interactions for both users and items
            user_interactions = Counter([row[0] for row in data])
            item_interactions = Counter([row[1] for row in data])

            # Filter out users with fewer interactions than min_interaction
            new_data = [row for row in data if user_interactions[row[0]] >= self.config["min_interaction"]]

            # Filter out items with fewer interactions than min_interaction
            new_data = [row for row in data if item_interactions[row[1]] >= self.config["min_interaction"]]

            if len(new_data) == len(data):
                break 
            else:
                data = new_data

        user_ids = {user: idx for idx, user in enumerate(set(row[0] for row in data))}
        item_ids = {item: idx for idx, item in enumerate(set(row[1] for row in data))} 

        if len(data[0]) > 2:
            data.sort(key=lambda row: datetime.strptime(row[2], '%Y%m%d%H%M'))

        user_item_dict = defaultdict(list)
        for row in data:
            user_id, item_id = user_ids[row[0]], item_ids[row[1]]
            user_item_dict[user_id].append(item_id)

        if not os.path.exists(os.path.join(self.path, "processed")): os.makedirs(f'{self.path}/processed', exist_ok=True)
        with open(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle"), 'wb') as f:
            pickle.dump((user_item_dict, len(user_ids), len(item_ids), item_ids), f)
        
        return user_item_dict, len(user_ids), len(item_ids), item_ids

    def _load_fakes(self, item_map):
        with open(os.path.join(self.path, f"{self.config['attack_type']}.json"), 'r') as f:
                fake_data = json.load(f)
        fake_user_interactions, target_items = fake_data["fake_users"], fake_data["target_items"]
        self.target_item = [item_map[item] for item in target_items]
        self.n_fake_users = 0
        for _, fake_interaction in fake_user_interactions.items():
            if len(fake_interaction) < self.config["min_interaction"]:
                continue
            self.user_interactions[self.n_users + self.n_fake_users] = [item_map[item] for item in fake_interaction]
            self.n_fake_users += 1
        self.n_users += self.n_fake_users

    def load_data(self):
        if os.path.exists(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle")):
            with open(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle"), 'rb') as f:
                self.user_interactions, self.n_users, self.n_items, item_map = pickle.load(f)
        else:
            self.user_interactions, self.n_users, self.n_items, item_map = self._load_org_data()
        
        if self.config["with_fakes"]:
            self._load_fakes(item_map)

    def get_train_batch(self, user_list):
        raise NotImplementedError

    def get_val_batch(self, user_list):
        raise NotImplementedError
    
    def get_test_batch(self, user_list, is_clean=False):
        raise NotImplementedError
    

class CFDataset(BasicDataset):
    def __init__(self, path, config) -> None:
        super().__init__(path, config)
        self.split_ratio = [0.7, 0.1, 0.2]
        self._build_set()
    
    def __len__(self):
        return self.n_train_num

    def _build_set(self):
        self.n_train_num = 0
        self.train_data = [[] for _ in range(self.n_users)]
        self.val_data = [[] for _ in range(self.n_users)]
        self.test_data = [[] for _ in range(self.n_users)]

        all_num = 0

        for user in range(self.n_users):
            n_inter_items = len(self.user_interactions[user])
            n_train_items = int(n_inter_items * self.split_ratio[0])
            n_test_items = int(n_inter_items * self.split_ratio[2])
            self.train_data[user] += self.user_interactions[user][:n_train_items]
            self.val_data[user] += self.user_interactions[user][n_train_items:-n_test_items]
            self.test_data[user] += self.user_interactions[user][-n_test_items:]
            self.n_train_num += n_train_items
            all_num += n_inter_items
        
        self.avg_inter = int(self.n_train_num / self.n_users)
        print(f"#User: {self.n_users}, #Item: {self.n_items}, #Ratings: {all_num}, AvgLen: {int(10 * (all_num / self.n_users)) / 10}, Sparsity: {100 - int(10000 * all_num / (self.n_users * self.n_items)) / 100}")

 
    def get_train_batch(self, inter_list):
        inter_list = inter_list.squeeze().tolist()
        pos_item_list = []
        neg_item_list = []
        user_list = np.random.randint(0, self.n_users, len(inter_list))
        for user in user_list:
            pos_item_list.append(np.random.choice(self.train_data[user]))
            neg_item = random.randint(0, self.n_items-1)
            while neg_item in self.train_data[user]:
                neg_item = random.randint(0, self.n_items-1)
            neg_item_list.append(neg_item)
        return user_list, np.array(pos_item_list), np.array(neg_item_list)
    
    def get_val_batch(self, user_list):
        # user_list = user_list.squeeze().tolist()
        return np.array(user_list), [self.val_data[user] for user in user_list], [self.train_data[user] for user in user_list]
    
    def get_test_batch(self, user_list, is_clean=False):
        # user_list = user_list.squeeze().tolist()
        if is_clean and self.config["with_fakes"]:
            new_user_list = []
            for user in user_list:
                if user < self.n_users - self.n_fake_users:
                    new_user_list.append(user)
            user_list = new_user_list
        return np.array(user_list), [self.test_data[user] for user in user_list], [self.train_data[user] + self.val_data[user] for user in user_list]
    

    def gcn_graph(self):
        user_list = []
        item_list = []
        for user in range(self.n_users):
            items = self.train_data[user]
            for item in items:
                user_list.append(user)
                item_list.append(item)

        user_dim = torch.LongTensor(user_list)
        item_dim = torch.LongTensor(item_list)
        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()

        Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.n_items, self.n_users+self.n_items]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)

        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.n_items, self.n_users+self.n_items]))
        Graph = Graph.coalesce()

        return Graph        
    
