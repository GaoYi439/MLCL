import torch
from torch.utils.data.dataset import Dataset
import numpy as np

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ComFoldData(Dataset):
    def __init__(self, data, label, com_label, train=True):
        # reading data as table form
        self.images = data
        # 对数据进行归一化处理
        # print('normliza:', np.min(self.images, 0).reshape(1, -1).shape)
        # self.images = (self.images - np.min(self.images, 0).reshape(1, -1)) / (np.max(self.images, 0).reshape(1, -1) - np.min(self.images, 0).reshape(1, -1))
        # reading labels
        self.labels = label

        self.comp_labels = com_label

        self.train = train

    def __getitem__(self, index):
        img, target, comp_label = torch.from_numpy(self.images[index]).float(), torch.from_numpy(self.labels[index]).float(), torch.from_numpy(self.comp_labels[index]).float()
        # target, comp_label = torch.from_numpy(self.labels[index]).float(), torch.from_numpy(self.comp_labels[index]).float()
        # if self.train:
        #     return img, target, comp_label
        # else:
        #     return img, target
        return img, target, comp_label

    def __len__(self):
        return len(self.labels)

class ComFoldData_onelabel(Dataset):
    def __init__(self, data, label, com_label, one_label, train=True):
        # reading data as table form
        self.images = data
        # 对数据进行归一化处理
        # print('normliza:', np.min(self.images, 0).reshape(1, -1).shape)
        # self.images = (self.images - np.min(self.images, 0).reshape(1, -1)) / (np.max(self.images, 0).reshape(1, -1) - np.min(self.images, 0).reshape(1, -1))
        # reading labels
        self.labels = label

        self.comp_labels = com_label

        self.one_labels = one_label

        self.train = train

    def __getitem__(self, index):
        img, target, comp_label, one_label = torch.from_numpy(self.images[index]).float(), torch.from_numpy(self.labels[index]).float(), torch.from_numpy(self.comp_labels[index]).float(), torch.from_numpy(self.one_labels[index]).float()
        # target, comp_label = torch.from_numpy(self.labels[index]).float(), torch.from_numpy(self.comp_labels[index]).float()
        # if self.train:
        #     return img, target, comp_label
        # else:
        #     return img, target
        return img, target, comp_label, one_label

    def __len__(self):
        return len(self.labels)