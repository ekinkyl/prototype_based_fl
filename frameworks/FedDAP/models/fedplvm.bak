import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
from utils.finch import FINCH
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedHierarchy.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos



def agg_func_cluster(protos):
    agg_protos={}
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
            proto_list = np.array(proto_list)
            # proto_list = np.stack(proto_list)

            c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                        ensure_early_exit=False, verbose=True)

            m, n = c.shape
            class_cluster_list = []
            for index in range(m):
                class_cluster_list.append(c[index, -1])

            class_cluster_array = np.array(class_cluster_list)
            uniqure_cluster = np.unique(class_cluster_array).tolist()
            agg_selected_proto = []

            for _, cluster_index in enumerate(uniqure_cluster):
                selected_array = np.where(class_cluster_array == cluster_index)
                selected_proto_list = proto_list[selected_array]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                agg_selected_proto.append(torch.tensor(proto))
            agg_protos[label] = agg_selected_proto
        else:
            agg_protos[label] = [proto_list[0].data]

    return agg_protos


class fedplvm(FederatedModel):
    NAME = 'fedplvm'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(fedplvm, self).__init__(nets_list, args, transform)
        self.global_protos = {}
        self.global_protos_avg={}
        self.local_protos = {}
        self.infoNCET = args.infoNCET
        self.lamb=1
        self.alpha=0.75
    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def local_cluster_collect(self,local_cluster_protos):
        global_collected_protos = {}
        for [idx, cluster_protos_label] in local_cluster_protos.items():
            for [label, cluster_protos_list] in cluster_protos_label.items():
                for i in range(len(cluster_protos_list)):
                    if label in global_collected_protos.keys():
                        global_collected_protos[label].append(cluster_protos_list[i])
                    else:
                        global_collected_protos[label] = [cluster_protos_list[i]]
        return global_collected_protos

    def proto_aggregation(self, local_protos_list):
        agg_protos_label={}
        for [label, proto_list] in local_protos_list.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)
                # proto_list = np.stack(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def proto_aggregation_avg(self, local_protos_list):
        agg_protos_label = dict()
        # for idx in self.online_clients:
        #     local_protos = local_protos_list[idx]
        for label in local_protos_list.keys():
            for i in range(len(local_protos_list[label])):
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos_list[label][i])
                else:
                    agg_protos_label[label] = [local_protos_list[label][i]]
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0]
                for i in proto_list:
                    proto += i
                agg_protos_label[label] = proto / len(proto_list)
            else:
                agg_protos_label[label] = proto_list[0]

        return agg_protos_label
    # def criterion_correction_old(self, bs,target,f_now, label, all_f_avg,all_global_protos_avg_keys):
    #
    #     data = torch.zeros(f_now.shape[0]).to(self.device)
    #     # for i in range(bs):
    #     #     data[i] = torch.cosine_similarity(f_now[i], torch.tensor(all_global_protos_avg_keys[label]).to(self.device), dim=0)
    #     data = torch.cosine_similarity(f_now, torch.tensor(all_global_protos_avg_keys[label]).to(self.device),dim=0)
    #     target = torch.tensor(np.array(torch.ones(f_now.shape[0]))).to(self.device)
    #     loss_mse = nn.MSELoss().to(self.device)
    #     loss = 2 * loss_mse(data, target)
    #     return loss

    def criterion_correction(self,  f_now, label, all_f_avg, all_global_protos_avg_keys):

        # prototype = np.array(all_f_avg)[all_global_protos_avg_keys == label.item()][0].to(self.device)
        f_idx=np.where(all_global_protos_avg_keys==label.item())[0][0]
        prototype=all_f_avg[f_idx].to(self.device)
        # for i in range(bs):
        #     data[i] = torch.cosine_similarity(f_now[i], torch.tensor(all_global_protos_avg_keys[label]).to(self.device), dim=0)
        cosine_sim = torch.cosine_similarity(f_now, prototype, dim=1)
        target = torch.ones_like(cosine_sim).to(self.device)
        loss_mse = nn.MSELoss().to(self.device)
        loss =  loss_mse(cosine_sim, target)
        # print("loss correction", loss)
        return loss

    def hierarchical_info_loss_alpha(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        f_pos = np.array(all_f)[all_global_protos_keys == label.item()][0].to(self.device)
        f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(self.device)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

        mean_f_pos = np.array(mean_f)[all_global_protos_keys == label.item()][0].to(self.device)
        mean_f_pos = mean_f_pos.view(1, -1)
        # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
        # mean_f_neg = mean_f_neg.view(9, -1)

        # loss_mse = nn.MSELoss()
        # cu_info_loss = loss_mse(f_now, mean_f_pos)

        hierar_info_loss = xi_info_loss
        return hierar_info_loss

    def calculate_infonce(self, f_now, f_pos, f_neg):
        alpha=self.alpha
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1).pow(alpha)
        l = l / self.infoNCET

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

    def loc_update(self, priloader_list,epoch):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients
        print(self.online_clients)
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        global_collected_protos = self.local_cluster_collect(self.local_protos)
        self.global_protos = self.proto_aggregation(global_collected_protos)
        self.global_protos_avg = self.proto_aggregation_avg(self.global_protos)
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)



        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f]
            mean_f = [item.detach() for item in mean_f]

        if len(self.global_protos_avg) != 0:
            all_global_protos_avg_keys = np.array(list(self.global_protos_avg.keys()))
            all_f_avg = []
            mean_f_avg = []
            for protos_key in all_global_protos_avg_keys:
                temp_f = self.global_protos_avg[protos_key]
                temp_f = torch.cat([temp_f], dim=0).to(self.device)
                temp_f = temp_f.unsqueeze(0)
                all_f_avg.append(temp_f.cpu())
                mean_f_avg.append(torch.mean(temp_f, dim=0).cpu())
            all_f_avg = [item.detach() for item in all_f_avg]
            mean_f_avg = [item.detach() for item in mean_f_avg]

        iterator = tqdm(range(self.local_epoch))

        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.classifier(f)

                lossCE = criterion(outputs, labels)

                if len(self.global_protos) == 0:
                    loss2= 0*lossCE
                else:
                    i = 0
                    loss2 = None
                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            f_now = f[i].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss_alpha(f_now, label, all_f, mean_f,
                                                                        all_global_protos_keys)

                            if loss2 is None:
                                loss2 = loss_instance
                            else:
                                loss2 += loss_instance
                        i += 1
                        loss2 = loss2 / i
                loss2=loss2
                if len(self.global_protos) == 0:
                    loss3= 0*lossCE
                else:
                    i=0
                    loss3=None
                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            f_now = f[i].unsqueeze(0)
                            loss_correction=self.criterion_correction(f_now,label,all_f,all_global_protos_keys)
                            if loss3 is None:
                                loss3=loss_correction
                            else:
                                loss3+=loss_correction
                        i += 1
                        loss3 = loss3
                loss3=loss3


                loss = lossCE + self.lamb*(loss2+loss3)
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, (loss2+loss3))
                optimizer.step()

                if iter == self.local_epoch - 1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i, :])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i, :]]

        agg_protos = agg_func_cluster(agg_protos_label)
        self.local_protos[index] = agg_protos
