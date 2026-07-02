import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
from collections import Counter


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str,args) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    agg_protos_label = {}
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                f = net.features(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
                # for i in range(len(labels)):
                #     if labels[i].item() in agg_protos_label:
                #         agg_protos_label[labels[i].item()].append(f[i, :])
                #     else:
                #         agg_protos_label[labels[i].item()] = [f[i, :]]
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        # if name in ['fl_digits','fl_officecaltech']:
        accs.append(top1acc)
        # x = []
        # y = []
        # for label in agg_protos_label.keys():
        #     for proto in agg_protos_label[label]:
        #
        #         # Move the tensor to the CPU if it's on GPU
        #         # if self.device == 'cuda':
        #         tmp = proto.cpu().detach().numpy()
        #         # else:
        #         #     tmp = proto.detach().numpy()
        #
        #         # Convert the tensor to a NumPy array
        #         # tmp = proto.detach().numpy()
        #         x.append(tmp)
        #         y.append(label)
        # np.save('./' +args.model  +args.dataset+ "domain{}_new".format(j)+'_protos.npy', x)
        # np.save('./' + args.model + args.dataset+ "domain{}_new".format(j)+'_labels.npy', y)
        # print("Save protos and labels successfully.")
        # elif name in ['fl_office31','fl_officehome']:
        #     accs.append(top5acc)
    net.train(status)
    return accs


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    print(" Running Domain Skew Setting")
    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            elif model.args.dataset == 'fl_pacs':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True,
                                                        p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_vlcs':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            elif model.args.dataset == 'fl_domainnet':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        if model.args.dataset == 'fl_digits':
            selected_domain_dict = {'mnist': 5, 'usps': 5, 'svhn': 5, 'syn': 5}
        elif model.args.dataset == 'fl_pacs':
            selected_domain_dict = {'photo': 3, 'art_painting': 3, 'cartoon': 3, 'sketch': 3}
        elif model.args.dataset == 'fl_officecaltech':
            selected_domain_dict = {'caltech': 3, 'amazon': 3, 'webcam': 3, 'dslr': 3}
        elif model.args.dataset == 'fl_domainnet':
            selected_domain_dict = {'clipart': 3, 'infograph': 1, 'painting': 4, 'quickdraw': 6, 'real': 4, 'sketch': 2}

        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)

    print(selected_domain_list)
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    # print(pri_train_loaders)
    model.trainloaders = pri_train_loaders
    model.client_domains=selected_domain_list
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders,epoch_index)
            # epoch_loc_loss_dict = model.loc_update_GA(pri_train_loaders,test_loaders, epoch_index)
        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME,args)
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        print('The ' + str(epoch_index) + ' Communcation Accuracy:', str(mean_acc), 'Method:', model.args.model)
        print(accs)

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)