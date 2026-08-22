"""
FedProto client — local training with prototype loss.
Ported from FedProto's lib/update.py (LocalUpdate class).
"""

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """Wrapper around a Dataset to select a subset by indices."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class FedProtoClient:
    """
    FedProto local update on a single client.

    Each round:
        1. Train local model for `train_ep` epochs
        2. Loss = CrossEntropy + ld * MSE(proto, global_proto)
        3. Collect per-class prototype means
        4. Return model weights + loss + accuracy + per-class prototypes
    """

    def __init__(self, args, dataset, idxs):
        self.args = args
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)

        # Create dataloader from the client's assigned indices
        idxs_train = list(idxs)
        self.trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.args.local_bs,
            shuffle=True,
            drop_last=True
        )

    def local_train(self, args, client_idx, global_protos, model, global_round):
        """
        Perform one round of local FedProto training.

        Args:
            args: global arguments
            client_idx: client index
            global_protos: dict {label: [proto_tensor]} from server
            model: copy of the local model to train
            global_round: current global round number

        Returns:
            model_weights: state_dict after training
            epoch_loss: dict with 'total', '1' (CE), '2' (proto) losses
            accuracy: final batch accuracy
            agg_protos_label: dict {label: [proto_tensor, ...]} per-class protos
        """
        model.train()
        epoch_loss = {'total': [], '1': [], '2': []}

        # Set optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        acc_val = 0.0
        for epoch in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': []}
            agg_protos_label = {}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images = images.to(self.device)
                labels = label_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)

                # Loss 1: cross-entropy classification loss
                loss1 = self.criterion(log_probs, labels)

                # Loss 2: prototype distance to global prototypes
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0 * loss1  # no proto loss in first round
                else:
                    proto_new = copy.deepcopy(protos.data)
                    for i, label in enumerate(labels):
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                # Collect per-class prototypes for this batch
                for i in range(len(labels)):
                    lbl = label_g[i].item()
                    if lbl in agg_protos_label:
                        agg_protos_label[lbl].append(protos[i, :])
                    else:
                        agg_protos_label[lbl] = [protos[i, :]]

                # Compute accuracy
                log_probs_clipped = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs_clipped.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(f'| Round: {global_round} | Client: {client_idx} '
                          f'| Epoch: {epoch} | [{batch_idx * len(images)}/'
                          f'{len(self.trainloader.dataset)} '
                          f'({100. * batch_idx / len(self.trainloader):.0f}%)]'
                          f'\tLoss: {loss.item():.3f} | Acc: {acc_val.item():.3f}')

                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item() if isinstance(loss2, torch.Tensor) else loss2)

            epoch_loss['total'].append(
                sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(
                sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(
                sum(batch_loss['2']) / len(batch_loss['2']))

        # Average over epochs
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label


class FedProtoLocalTest:
    """Test a local model on its assigned test data."""

    def __init__(self, args, dataset, idxs):
        self.args = args
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)
        self.testloader = DataLoader(
            DatasetSplit(dataset, list(idxs)),
            batch_size=64, shuffle=False
        )

    def test(self, args, idx, classes_list, model):
        """Test local model accuracy."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            outputs = outputs[:, 0:args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total if total > 0 else 0.0
        return loss, acc
