import torch
from torch.utils.data.dataset import Dataset
import numpy as np


np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Scene_data(Dataset):
    def __init__(self, data_path, label_path, train):
        self.img_path = data_path
        # reading data as table form
        self.images = np.genfromtxt(self.img_path, delimiter=',')
        self.images = (self.images - np.min(self.images, 1).reshape(-1, 1)) / (np.max(self.images, 1).reshape(-1, 1) - np.min(self.images, 1).reshape(-1, 1))
        # reading labels
        self.label_path = label_path
        self.labels = np.genfromtxt(self.label_path, delimiter=',')
        self.train = train

        self.comp_labels = self.generate_compl_labels()


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


    def generate_compl_labels(self):
        K = np.size(self.labels, 1)
        n = np.size(self.labels, 0)

        comp_Y = np.zeros([n, K])
        labels_hat = np.array(1-self.labels, dtype=bool)  # 将多标记矩阵转化为bool型，使后续选择补标记时避开真实标记

        for i in range(n):
            candidates = np.arange(K).reshape(1, K)
            mask = labels_hat[i].reshape(1, -1)
            candidates_ = candidates[mask]
            idx = np.random.randint(0, len(candidates_))  # 随机产生取补标记的index，每个样本随机选择一个补标记
            comp_Y[i][candidates_[idx]] = 1
        return comp_Y

