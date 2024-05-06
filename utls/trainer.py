import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utls.mydataset import *
from model.MF.MF import *
from model.LightGCN.LightGCN import *
from utls.utilize import batch_split


class BasicTrainer:
    def __init__(self, trainer_config) -> None:
        self.config = trainer_config
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.max_patience = trainer_config.get('patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)

        self._create_dataset(f"data/{trainer_config['dataset']}")
        self._create_dataloader()
        self._create_model()
        self._create_opt()

    def _create_dataset(self, path):
        raise NotImplementedError
    
    def _create_dataloader(self):
        raise NotImplementedError

    def _create_model(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = glo[f'{self.config["model"]}'](self.config["model_config"])
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _create_opt(self):
        raise NotImplementedError
    
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _eval_model(self, eval_type):
        raise NotImplementedError

    def _save_model(self):
        torch.save({
            'model': self.model.state_dict(),
        }, self.model_path)
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])

            
    
    def train(self, model_path):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        self.model_path = os.path.join(self.model_path, f"{'attack_'+ str(self.config['attack_type']) if self.config['with_fakes'] else 'normal'}_{self.config['seed']}_{datetime.now().strftime('%Y%m%d%H%M')}.pth")
        best_metrics = -1.0

        for epoch in range(1, self.n_epochs+1):
            self._train_epoch(epoch)
            
            if epoch % self.config["val_interval"] == 0:
                _, ndcg_list = self._eval_model(epoch, eval_type='val')
                if self.config["with_fakes"]:
                    self.eval_attacks(epoch)

                if ndcg_list[1] > best_metrics:
                    best_metrics = ndcg_list[1]
                    self._save_model()
                    patience = self.config["patience"]
                else:
                    patience -= self.config["val_interval"]
                    if patience <= 0:
                        print('Early stopping!')
                        break
        
        if best_metrics != -1.0:
            self._load_model(self.model_path)
        rc_list, ndcg_list = self._eval_model(eval_type='test')
        if self.config["with_fakes"]:
            self.eval_attacks()

        return rc_list, ndcg_list


    def _eval_target(self, batch_idx, k, targe_item):
        raise NotImplementedError

    def eval_attacks(self, epoch=0, path=None):
        if path is not None:
            self._load_model(path)
        top_ks = self.config["attack_top_k"]
        targe_item = self.dataset.target_item
        self.model.eval()
        hr_list = []
        ndcg_list = []
        user_list = list(range(self.dataset.n_users))
        for idx, k in enumerate(top_ks):
            avg_hr = np.zeros(len(targe_item))
            avg_ndcg = np.zeros(len(targe_item))
            all_cnt = np.zeros(len(targe_item))
            
            for batch_data in batch_split(users=user_list, batch_size=self.config["test_batch_size"]):
                hr, ndcg, cnt = self._eval_target(batch_data, k, targe_item, idx)
                avg_hr += hr
                avg_ndcg += ndcg
                all_cnt += cnt
            
            hr_list.append(np.mean(avg_hr / all_cnt))
            ndcg_list.append(np.mean(avg_ndcg / all_cnt))

        if epoch != 0:
            out_text = f"Attack at Epoch {epoch} :"
        else:
            out_text = f"Final Attack:"
            
        for i, k in enumerate(top_ks):
            out_text += f"\nT-HR@{k}: {hr_list[i]:.6f}, T-NDCG@{k}: {ndcg_list[i]:.6f};"
        
        print(out_text)
    

        

class CFTrainer(BasicTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_dataset(self, path):
        self.dataset = CFDataset(path, self.config)

    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"])

    def _create_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=int(self.config["batch_size"]), shuffle=True)
        print(f"Create Dataloader with batch_size:{int(self.config['batch_size'])}")

    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
            users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
            pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
            neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            loss = self._rec_loss(pos_logits, neg_logits)
            loss = loss.mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")

    def _eval_model(self, epoch=0, eval_type='val'):
        assert eval_type in ['val', 'test']
        self.model.eval()
        top_ks = self.config["top_k"]
        recall_list = [0 for _ in self.config["top_k"]]
        ndcg_list = [0 for _ in self.config["top_k"]]
        user_list = list(range(self.dataset.n_users))
        eval_user = 0
        for batch_data in batch_split(users=user_list, batch_size=self.config["test_batch_size"]):
            if eval_type == 'val':
                user_id_list, user_inter_list, user_train_list = self.dataset.get_val_batch(batch_data)
            else:
                user_id_list, user_inter_list, user_train_list = self.dataset.get_test_batch(batch_data)

            if len(user_id_list) == 0:
                continue
            
            score_list = self.model.predict(user_id_list)

            for idx, user_train_items in enumerate(user_train_list):
                score_list[idx, user_train_items] = float('-inf')

            batch_recall, batch_ndcg = [], []
            for k in top_ks:
                recall_k, ndcg_k = 0, 0
                for user_idx, user_inter_items in enumerate(user_inter_list):
                    _, top_indices = torch.topk(score_list[user_idx], k)

                    test_matrix = np.zeros(k)
                    test_matrix[:len(user_inter_items)] = 1

                    num_hits = sum([1 for item in user_inter_items if item in top_indices])

                    # Recall@k
                    recall_k += num_hits / len(user_inter_items) if user_inter_items else 0

                    # NDCG@k
                    dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(top_indices) if item in user_inter_items])
                    idcg = np.sum(test_matrix * 1./np.log2(np.arange(2, k + 2)))
                    idcg = 1 if idcg == 0 else idcg
                    ndcg_k += dcg / idcg

                batch_recall.append(recall_k)
                batch_ndcg.append(ndcg_k)
            eval_user += len(user_inter_list)

            recall_list = [recall + brecall for recall, brecall in zip(recall_list, batch_recall)]
            ndcg_list = [ndcg + bndcg for ndcg, bndcg in zip(ndcg_list, batch_ndcg)]

        recall_list = [recall / eval_user for recall in recall_list]
        ndcg_list = [ndcg / eval_user for ndcg in ndcg_list]

        if eval_type == 'val':
            out_text = f"Recomendation Performance at Epoch {epoch} :"
        else:
            out_text = f"Final Recomendation Performance:"

        for i, k in enumerate(top_ks):
            out_text += f"\nRecall@{k}: {recall_list[i]:.4f}, NDCG@{k}: {ndcg_list[i]:.4f};"
        
        print(out_text)

        return recall_list, ndcg_list

    def _rec_loss(self, pos, neg):
        return F.softplus(neg - pos)
    
    def _eval_target(self, batch_idx, k, targe_item, k_id=0): 
        hr = np.zeros(len(targe_item))
        ndcg = np.zeros(len(targe_item))
        cnt = np.zeros(len(targe_item))

        user_id_list, user_inter_list, user_train_list = self.dataset.get_test_batch(batch_idx, is_clean=True)

        if len(user_id_list) == 0:
            return hr, ndcg, cnt

        if user_id_list.shape[0] != 0:
            score_list = self.model.predict(user_id_list)

            for idx, user_train_items in enumerate(user_train_list):
                score_list[idx, user_train_items] = float('-inf')

            _, sorted_indices = score_list.sort(dim=1, descending=True) 
            for u in range(len(user_id_list)):
                for i, item in enumerate(targe_item):
                    if item in user_inter_list[u] or item in user_train_list[u]:
                        continue
                    rank = (sorted_indices[u] == item).nonzero().item() + 1
                    cnt[i] += 1
                    if rank <= k:
                        hr[i] += 1
                        ndcg[i] += 1 / np.log2(rank + 1)
                        self.user_with_atk[u][k_id] = True
        
        return hr, ndcg, cnt



class MFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)


class LightGCNTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_model(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset.gcn_graph())
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()



# //////////////////////////////////

class APRCFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

        self.adv_reg = trainer_config['defense_config']['adv_reg']
        self.eps = trainer_config['defense_config']['eps']
        self.begin_adv = trainer_config['defense_config']['begin_adv']
    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
            users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
            pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
            neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            loss = self._rec_loss(pos_logits, neg_logits)
            if epoch >= self.begin_adv:
                delta_users_r, delta_pos_items_r, delta_neg_items_r = \
                    torch.autograd.grad(loss.mean(), (users_emb, pos_items_emb, neg_items_emb), retain_graph=True)
                delta_users_r = F.normalize(delta_users_r, p=2, dim=1) * self.eps
                delta_pos_items_r = F.normalize(delta_pos_items_r, p=2, dim=1) * self.eps
                delta_neg_items_r = F.normalize(delta_neg_items_r, p=2, dim=1) * self.eps
                pos_logits = ((users_emb + delta_users_r) * (pos_items_emb + delta_pos_items_r)).sum(-1)
                neg_logits = ((users_emb + delta_users_r) * (neg_items_emb + delta_neg_items_r)).sum(-1)
                adv_loss = self._rec_loss(pos_logits, neg_logits)
                loss = loss.mean() + self.config["weight_decay"] * l2_norm_sq + self.adv_reg * adv_loss.mean()
            else:
                loss = loss.mean() + self.config["weight_decay"] * l2_norm_sq
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")


class MFAPRTrainer(APRCFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)


class LightGCNAPRTrainer(APRCFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_model(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset.gcn_graph())
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()


# //////////////////
class PamaCFCFTrainer(CFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

        self.adv_reg = trainer_config['defense_config']['adv_reg']
        self.eps = trainer_config['defense_config']['eps']
        self.begin_adv = trainer_config['defense_config']['begin_adv']
        self.user_lmb = trainer_config['defense_config']['user_lmb']
    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            user_id_list, pos_item_list, neg_item_list = self.dataset.get_train_batch(batch_data)
            users_emb, pos_items_emb, neg_items_emb, l2_norm_sq = self.model(user_id_list, pos_item_list, neg_item_list)
            pos_logits = torch.sum(users_emb * pos_items_emb, dim=1)
            neg_logits = torch.sum(users_emb * neg_items_emb, dim=1)
            loss = self._rec_loss(pos_logits, neg_logits)

            if epoch >= self.begin_adv:
                weight = users_emb.norm(dim=1).pow(2).detach().cpu()
                user_eps = self._get_user_eps(weight, self.user_lmb)
                delta_users_r, delta_pos_items_r, delta_neg_items_r = \
                    torch.autograd.grad(loss.mean(), (users_emb, pos_items_emb, neg_items_emb), retain_graph=True)
                
                delta_users_r = F.normalize(delta_users_r, p=2, dim=1) * user_eps.view(-1, 1).to(delta_users_r.device) * self.eps
                delta_pos_items_r = F.normalize(delta_pos_items_r, p=2, dim=1) * user_eps.view(-1, 1).to(delta_users_r.device) * self.eps
                delta_neg_items_r = F.normalize(delta_neg_items_r, p=2, dim=1) * user_eps.view(-1, 1).to(delta_users_r.device) * self.eps
                pos_logits = ((users_emb + delta_users_r) * (pos_items_emb + delta_pos_items_r)).sum(-1)
                neg_logits = ((users_emb + delta_users_r) * (neg_items_emb + delta_neg_items_r)).sum(-1)
                adv_loss = self._rec_loss(pos_logits, neg_logits)
                loss = loss.mean() + self.config["weight_decay"] * l2_norm_sq + self.adv_reg * adv_loss.mean()
            else:
                loss = loss.mean() + self.config["weight_decay"] * l2_norm_sq

            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        print(f"Epoch {epoch}: Rec Loss: {epoch_loss/len(self.dataloader):.4f}, Time: {end_t-start_t:.2f}")

    def _get_user_eps(self, weight, lmabda):
        mean_w = weight.mean()
        eps = torch.zeros(weight.shape[0])

        eps = weight / (mean_w + 1e-9)

        return lmabda * torch.sigmoid(eps - 1)
    

class MFPamaCFTrainer(PamaCFCFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)


class LightGCNPamaCFTrainer(PamaCFCFTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_model(self):
        glo = globals()
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = glo[f'{self.config["model"]}'](self.config["model_config"], self.dataset.gcn_graph())
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()