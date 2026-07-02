from collections import defaultdict

import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import math
from utils.finch import FINCH
import numpy as np
# Removed augmentation import
from utils.util import *
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedHierarchy.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def agg_func(protos):
    """
    Returns the average of the weights for each (class, domain) key.
    """
    for key, proto_list in protos.items():  # key is now (class_label, domain_label)
        if len(proto_list) > 1:
            proto = torch.zeros_like(proto_list[0].data)
            for i in proto_list:
                proto += i.data
            protos[key] = proto / len(proto_list)
        else:
            protos[key] = proto_list[0]
    return protos



def agg_func_dp(protos,dps=0.0,dpp=0.0):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            proto=dp(proto,diff_privacy_scale=dps,diff_privacy_perturbation=dpp)
            protos[label] = proto / len(proto_list)
        else:
            proto = dp(proto_list[0].data, diff_privacy_scale=dps, diff_privacy_perturbation=dpp)
            protos[label] = proto

    return protos


class feddap(FederatedModel):
    NAME = 'feddap'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(feddap, self).__init__(nets_list, args, transform)
        self.global_protos = []
        self.prev_global_protos = []
        self.intra_proto=[]
        self.local_protos = {}
        self.local_protos_augment = {}
        self.infoNCET = args.infoNCET


    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)





    
    def proto_aggregation_avg(self, local_protos_list):
        agg_protos_label = dict()

        for idx in self.online_clients:
            local_protos = local_protos_list[idx]  # keys: (class_label, domain_label)
            for key in local_protos.keys():  # key = (class_label, domain_label)
                if key in agg_protos_label:
                    agg_protos_label[key].append(local_protos[key])
                else:
                    agg_protos_label[key] = [local_protos[key]]

        # Perform averaging
        for key, proto_list in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = torch.zeros_like(proto_list[0].data)
                for p in proto_list:
                    proto += p.data
                agg_protos_label[key] = proto / len(proto_list)
            else:
                agg_protos_label[key] = proto_list[0].data

        return agg_protos_label



    def proto_aggregation_attn(self, local_protos_list, temperature=0.1):
        """
        Aggregate global prototypes using cosine similarity-based self-attention
        across local prototypes of each (class, domain).
        """
        agg_protos_label = dict()

        # Step 1: Group local prototypes per (class, domain) key
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]  # keys: (class_label, domain_label)
            for key in local_protos.keys():  # key = (class_label, domain_label)
                if key in agg_protos_label:
                    agg_protos_label[key].append(local_protos[key])
                else:
                    agg_protos_label[key] = [local_protos[key]]

        # Step 2: Perform self-attention-based aggregation
        for key, proto_list in agg_protos_label.items():
            if len(proto_list) > 1:
                stacked = torch.stack([p.detach() for p in proto_list])  # shape (N, D)

                # Cosine similarity matrix (N, N)
                sim_matrix = F.cosine_similarity(stacked.unsqueeze(1), stacked.unsqueeze(0), dim=-1)

                # Attention scores: sum row-wise → softmax
                attn_weights = torch.softmax(sim_matrix.sum(dim=1) / temperature, dim=0)  # (N,)

                # Weighted sum
                proto = torch.sum(attn_weights.view(-1, 1) * stacked, dim=0)
                agg_protos_label[key] = proto
            else:
                agg_protos_label[key] = proto_list[0].data


        return agg_protos_label


 
   

    def hierarchical_info_loss_cross_domain(self, f_now, class_label, current_domain, global_protos):
        """
        InfoNCE loss with:
        - Positives: (same class, different domain)
        - Negatives: (different class, different domain)
        """
        f_pos_list = []
        f_neg_list = []

        for (cls, domain), proto in global_protos.items():
            if domain == current_domain:
                continue  # Ignore same-domain prototypes altogether

            if cls == class_label:
                f_pos_list.append(proto.detach().unsqueeze(0))  # positive: same class, different domain
                # print(f" Positive key = (class={cls}, domain={domain})")  # ✅ Log positive pair
            else:
                f_neg_list.append(proto.detach().unsqueeze(0))  # negative: different class, different domain
                # print(f" Negative key = (class={cls}, domain={domain})")  # ✅ Log negative pair
        # print(f_neg_list)
        # If no valid positive or negative, skip loss
        if len(f_pos_list) == 0 or len(f_neg_list) == 0:
            return torch.tensor(0.0).to(self.device)

        f_pos = torch.cat(f_pos_list, dim=0).to(self.device)

        f_neg = torch.cat(f_neg_list, dim=0).to(self.device)

        return self.calculate_infonce(f_now, f_pos, f_neg)

   

    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.args.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss

    def loc_update(self, priloader_list, epoch):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        print(self.online_clients)
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i],self.client_domains[i])

        # self.global_protos = self.proto_aggregation_avg(self.local_protos)
        self.global_protos = self.proto_aggregation_attn(self.local_protos,temperature=0.001) #domain specific prototype aggregation
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader,domain_label):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
       


        iterator = tqdm(range(self.local_epoch))

      
        for iter in iterator:
            agg_protos_label = {}
            augmented_agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)

                labels = labels.to(self.device)


                f = net.features(images)
                outputs = net.classifier(f)


                lossCE = criterion(outputs, labels)

                 
                #Cross-domain protototype contrastive learning (CPCL)
                if len(self.global_protos) == 0:
                    loss_InfoNCE = 0
                else:
                    loss_InfoNCE = 0.0
                    for i, label in enumerate(labels):
                        class_label = label.item()
                        f_now = f[i].unsqueeze(0)
                  
                        loss_instance = self.hierarchical_info_loss_cross_domain(
                            f_now, class_label, domain_label, self.global_protos
                        )
                        loss_InfoNCE += loss_instance
                    loss_InfoNCE = loss_InfoNCE / len(labels)
                loss_InfoNCE = loss_InfoNCE

         
                #Domain-consistent prototype alignment (DPA)
                loss_mse= nn.MSELoss()
                if len(self.global_protos) == 0:
                    loss_intra_proto = 0
                else:
                    proto_new = copy.deepcopy(f.detach())
                    for i, label in enumerate(labels):
                        y_c = label.item()
                        key = (y_c, domain_label)
                        if key in self.global_protos:
                            if not isinstance(self.global_protos[key], list):  # Check if it's a tensor
                                proto_new[i, :] = self.global_protos[key].data
                            else:
                                proto_new[i, :] = self.global_protos[key][0].data
                    loss_intra_proto = 1-F.cosine_similarity(F.normalize(proto_new,dim=1), F.normalize(f,dim=1),dim=1).mean()



                loss = lossCE  + self.args.lamda_1*loss_intra_proto + self.args.lamda_2*loss_InfoNCE
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, loss_InfoNCE)
                optimizer.step()

                if iter == self.local_epoch - 1:
                    for i in range(len(labels)):
                        key = (labels[i].item(), domain_label)
                        if key in agg_protos_label:
                            agg_protos_label[key].append(f[i, :])
                        else:
                            agg_protos_label[key] = [f[i, :]]



        agg_protos = agg_func(agg_protos_label)
        # agg_protos = agg_func_dp(agg_protos_label,dps=0.05,dpp=0.1)
        self.local_protos[index] = agg_protos



