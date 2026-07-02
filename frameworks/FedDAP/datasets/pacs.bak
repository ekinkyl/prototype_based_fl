import torchvision.transforms as transforms
from datasets.transforms.denormalization import DeNormalize
from datasets.transforms.twocroptransform import TwoCropsTransform
from datasets.utils.federated_dataset import FederatedDataset, partition_pacs_skew_loaders
from backbone.ResNet import resnet10, resnet12, resnet18, resnet34
from backbone.efficientnet import EfficientNetB0
from backbone.googlenet import GoogLeNet
from backbone.mobilnet_v2 import MobileNetV2
from utils.conf import data_path
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
split_dict = {
    'train': 'train',
    'test': 'test',
}
def low_freq_mutate_np(amp_src, amp_trg, L=0.1, ratio=1.0):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    # print (b)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = np.random.randint(1, 10) / 10

    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
    # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
    return a_src


def source_to_target_freq(src_img, amp_trg, L=0.1, ratio=1.0):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    # amp_trg = amp_trg.transpose((2, 0, 1))
    # print('##', src_img.shape)
    src_img_np = src_img  # .cpu().numpy()
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    # print('##', amp_trg.shape)
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L, ratio=1.0)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)

class MyPACS(data.Dataset):
    def __init__(self, root, train='train', transform=None,
                 target_transform=None, data_name=None, use_fft=False, prob_domain_name=[]) -> None:
        # self.not_aug_transform = utils.Compose([utils.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.prob_domain_name = prob_domain_name
        self.use_fft = use_fft

        if use_fft:
            self.all_prob_img_paths = []
            for i in range(len(self.prob_domain_name)):
                domain_name = self.prob_domain_name[i]
                split_file = os.path.join(self.root, 'label', f'{domain_name}_{split_dict[self.train]}_kfold' + '.txt')
                prob_img_paths, _ = MyPACS.read_txt(split_file, self.root + 'raw_images/')
                self.all_prob_img_paths.append(prob_img_paths)

        self.dataset = self.__build_truncated_dataset__()

    def get_amp(self):
        site_idx = np.random.choice(len(self.prob_domain_name))

        used_img_paths = self.all_prob_img_paths[site_idx]

        used_img_path = used_img_paths[np.random.randint(len(used_img_paths) // 8)]
        img = Image.open(used_img_path).convert('RGB')
        img_np = np.asarray(img, dtype=np.float32)

        img_np = img_np.transpose((2, 0, 1))
        fft = np.fft.fft2(img_np, axes=(-2, -1))
        amp_np, pha_np = np.abs(fft), np.angle(fft)

        return amp_np

    def __build_truncated_dataset__(self):
        self.split_file = os.path.join(self.root, 'label', f'{self.data_name}_{split_dict[self.train]}_kfold' + '.txt')
        self.imgs, self.labels = MyPACS.read_txt(self.split_file, self.root + 'raw_images/')


    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            labels.append(int(line_txt[1]) - 1)
        return imgs, labels

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.use_fft:
            img_np = np.asarray(img, dtype=np.float32)

            tar_freq = self.get_amp()
            image_tar_freq = source_to_target_freq(img_np, tar_freq, L=0, ratio=1.0)
            image_tar_freq = np.clip(image_tar_freq, 0, 255)

            image_tar_freq = Image.fromarray(image_tar_freq.astype(np.uint8))
            if self.transform is not None:
                img = self.transform(img)
                image_tar_freq = self.transform(image_tar_freq)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, image_tar_freq, target
        else:

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)

class FedLeaPACS(FederatedDataset):
    NAME = 'fl_pacs'
    SETTING = 'domain_skew'
    N_CLASS = 7


    DOMAINS_LIST = ['photo', 'art_painting', 'cartoon', 'sketch']
    percent_dict = {'photo': 0.3, 'art_painting':0.3,
                         'cartoon': 0.3, 'sketch': 0.3}





    def get_data_loaders(self, selected_domain_list):

        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        Nor_TRANSFORM = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))])

        test_transform_1 = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), self.get_normalization_transform()])

        test_transform = transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ToTensor(),
             self.get_normalization_transform()])
        client_domain_name_list = self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list

        '''
        Loading the default four domains datasets
        '''
        domain_training_dataset_list = []
        domain_testing_dataset_list = []
        test_dataset_dict = {}
        train_dataset_dict = {}
        # domain_train_eval_dataset_dict = {}




        use_fft = False

        for _, domain in enumerate(self.DOMAINS_LIST):

            domain_testing_dataset = MyPACS(data_path() + 'PACS/', train='test', transform=test_transform_1, data_name=domain)


            domain_testing_dataset_list.append(domain_testing_dataset)
            test_dataset_dict[domain] = domain_testing_dataset
        for _, domain in enumerate(client_domain_name_list):

            if use_fft:
                prob_domain_name = client_domain_name_list
            else:
                prob_domain_name = []

            domain_training_dataset = MyPACS(data_path() + 'PACS/', train='train', transform=Nor_TRANSFORM,
                                             data_name=domain,
                                             use_fft=use_fft, prob_domain_name=prob_domain_name)
            domain_training_dataset_list.append(domain_training_dataset)
            train_dataset_dict[domain] = domain_training_dataset
            # domain_train_eval_dataset_dict[domain] = MyPACS(data_path() + 'PACS/', train='val', transform=train_val_transform, data_name=domain)

        # self.partition_domain_loaders(client_domain_name_list, domain_training_dataset_dict, domain_testing_dataset_dict, domain_train_eval_dataset_dict)
        traindls, testdls = partition_pacs_skew_loaders(domain_training_dataset_list, domain_testing_dataset_list, self)
        return traindls, testdls


    @staticmethod
    def get_backbone(parti_num, names_list):
        nets_dict = {'resnet10': resnet10, 'resnet12': resnet12, 'resnet18':resnet18, 'efficient': EfficientNetB0, 'mobilnet': MobileNetV2}
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                nets_list.append(resnet18(FedLeaPACS.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaPACS.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform