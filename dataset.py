import os
from torch.utils.data import DataLoader
import numpy as np

from dataprocessing.scene import ComFoldData, ComFoldData_onelabel

'''
fold: 所处折数,python从0开始计数
n_test: 每一折的个数
'''
def ComFold(batchsize, Filename, nfold, fold):
    # print('Data Preparation of Scene')
    # train_data = os.path.join(path, 'train_scene_data.csv')
    # train_label = os.path.join(path, 'train_scene_label.csv')
    data = np.genfromtxt(Filename[0], delimiter=',')
    label = np.genfromtxt(Filename[1], delimiter=',')
    com_label = np.genfromtxt(Filename[2], delimiter=',')

    n_test = len(com_label)//nfold
    print('n size:', n_test)
    y = np.arange(len(com_label))
    start = fold*n_test
    if start+n_test > len(com_label):
        test = y[start:]
    else:
        test = y[start:start+n_test]
    train = np.setdiff1d(y, test)

    # training dataset
    train_scene = ComFoldData(data[train, :], label[train, :], com_label[train, :], train=True)
    print("train data shape:", data[train, :].shape)
    train_loader = DataLoader(
        train_scene,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    # Data loader for test dataset
    size = len(test)
    test_scene = ComFoldData(data[test, :], label[test, :], com_label[test, :], train=False)
    print("test data & label shape:", data[test, :].shape, label[test, :].shape)
    test_loader = DataLoader(
        test_scene,
        batch_size=size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, test_loader


def ComFold_onelabel(batchsize, Filename, nfold, fold):
    # print('Data Preparation of Scene')
    # train_data = os.path.join(path, 'train_scene_data.csv')
    # train_label = os.path.join(path, 'train_scene_label.csv')
    data = np.genfromtxt(Filename[0], delimiter=',')
    label = np.genfromtxt(Filename[1], delimiter=',')
    com_label = np.genfromtxt(Filename[2], delimiter=',')
    one_label = np.genfromtxt(Filename[3], delimiter=',')

    n_test = len(com_label)//nfold
    print('n size:', n_test)
    y = np.arange(len(com_label))
    start = fold*n_test
    if start+n_test > len(com_label):
        test = y[start:]
    else:
        test = y[start:start+n_test]
    train = np.setdiff1d(y, test)

    # training dataset
    train_scene = ComFoldData_onelabel(data[train, :], label[train, :], com_label[train, :], one_label[train, :], train=True)
    print("train data shape:", data[train, :].shape)
    train_loader = DataLoader(
        train_scene,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    # Data loader for test dataset
    size = len(test)
    test_scene = ComFoldData_onelabel(data[test, :], label[test, :], com_label[test, :], one_label[test, :], train=False)
    print("test data & label shape:", data[test, :].shape, label[test, :].shape)
    test_loader = DataLoader(
        test_scene,
        batch_size=size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, test_loader