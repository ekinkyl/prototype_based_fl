import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
from utils.util import set_requires_grad
import torch
import numpy as np
class FedDGGA(FederatedModel):
    NAME = 'feddgga'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedDGGA, self).__init__(nets_list,args,transform)
        self.base_step_size = 0.2

        self.step_size_decay = self.base_step_size/100
        self.agg_weight= np.ones(12)/12
        self.mu = args.mu


    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def get_local_test_acc(self, net, test_loader):
        total = 0
        top1 = 0
        net.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if isinstance(images, list):
                    images, labels = images[0].to(self.device), labels.to(self.device)
                else:
                    images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                # print(batch_idx,images.size(),labels.size(), max5.size())
                top1 += (labels == max5[:, 0:1]).sum().item()

                total += labels.size(0)

        top1acc = round(100 * top1 / total, 2)
        net.train()
        return top1acc

    def update_weight_by_GA(self, accs_before_agg, accs_after_agg, online_num, epoch_index):
        accs_before_agg = np.array(accs_before_agg)
        accs_after_agg = np.array(accs_after_agg)

        accs_diff = accs_after_agg - accs_before_agg

        step_size = self.base_step_size - (epoch_index - 1) * self.step_size_decay
        step_size *= np.ones(online_num) / online_num

        norm_gap_array = accs_diff / np.max(np.abs(accs_diff))

        self.agg_weight += norm_gap_array * step_size
        self.agg_weight = np.clip(self.agg_weight, 0, 1)

        self.agg_weight = self.agg_weight / np.sum(self.agg_weight)

        return

    def loc_update_GA(self,priloader_list,test_loaders,epoch):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients
        accs_before_agg = []
        for i in online_clients:
            acc_before_agg = self.get_local_test_acc(self.nets_list[i], priloader_list[i])
            accs_before_agg.append(acc_before_agg)

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        accs_after_agg = []
        for i in online_clients:
            acc_after_agg = self.get_local_test_acc(self.global_net, priloader_list[i])
            accs_after_agg.append(acc_after_agg)
        self.update_weight_by_GA(accs_before_agg, accs_after_agg, len(online_clients), epoch)

        self.aggregate_nets(self.agg_weight)

        return None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        self.global_net = self.global_net.to(self.device)
        global_weight_collector = list(self.global_net.parameters())
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                f=net.features(images)
                outputs = net(images)
                loss = criterion(outputs, labels)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((0.01 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += self.mu * fed_prox_reg
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

