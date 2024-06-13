import torch.nn as nn
import torch
import numpy as np
import dataset
from metrics import HammingLoss, OneError, Coverage, RankingLoss, AveragePrecision, predict, com_accuracy
from model.models import linear
import torch.nn.functional as F
import os
import argparse

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch implementation of TPAMI paper MLCL')
parser.add_argument('--dataset', default='scene', type=str, help='dataset name (scene)')
parser.add_argument('--num-class', default=6, type=int, help='number of classes')
parser.add_argument('--input-dim', default=294, type=int, help='number of features')
parser.add_argument('--fold', default=9, type=int, help='fold-th fold of 10-cross fold')
parser.add_argument('--model', default="linear", type=str, choices=['linear'])
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--step', default="one", type=str, help='estimating transition matrix T')
parser.add_argument('--stage', default="second", type=str, help='estimating transition matrix T or learning classifier')
parser.add_argument('--T_path', default="estimation_T/scene_T_linear.txt", type=str)
parser.add_argument('--one_label', default=False)


def Train_T(the_model, model_path, train_loader, num_class):
    # Using trained complementary label model to obtain S

    the_model.load_state_dict(torch.load(model_path))

    the_model = the_model.to(device)

    T = np.zeros((num_class, num_class))
    for k in range(num_class):
        total_K = 0
        for images, labels, complementary in train_loader:
            images, labels, complementary = images.to(device), labels.to(device), complementary.to(device)
            output = the_model(images)
            sf_pre = F.softmax(output, dim=1)

            com_hot = 1-labels.cuda().cpu().data.numpy()

            sf_pre = sf_pre.cuda().cpu().data.numpy()

            com_k = np.argwhere(com_hot[:, k] == 0)
            reg_com_k = np.sum(sf_pre[com_k[:, 0]], 0)
            T[k] += reg_com_k
            total_K += len(com_k)
        if total_K != 0:
            T[k] = T[k] / total_K
    T = T.T
    for i in range(len(T)):
        T[i][i] = 0
    T_sum = np.sum(T, 0).reshape(1, -1)
    for i in range(len(T)):
        if T_sum[0, i] != 0:
            T[:, i] = T[:, i] / T_sum[0, i]
    return T


def LabelCorrelation(file_name, K):
    '''
    calculating the label correlation matrix
    '''
    com_label = np.genfromtxt(file_name[2], delimiter=',')
    S = np.zeros([K, K])
    for i in range(K):
        for j in range(K):
            if i != j:
                num_1i = com_label[:, i] == 1
                num_1j = com_label[:, j] == 1
                num_0j = com_label[:, j] == 0
                num_0i = com_label[:, i] == 0
                A = np.array(num_1i * num_1j, dtype=float).sum()
                B = np.array(num_1i * num_0j, dtype=float).sum()
                C = np.array(num_0i * num_1j, dtype=float).sum()
                D = np.array(num_0i * num_0j, dtype=float).sum()
                S[i, j] = (A * D - B * C) / ((A + B) * (C + D) * (A + C) * (B + D)) ** 0.5
    for i in range(K):
        S[i, i] = 0
    S_sum = np.sum(S, 1).reshape(-1, 1)
    for i in range(len(S)):
        if S_sum[i] != 0:
            S[i, :] = S[i, :] / S_sum[i]
    for i in range(K):
        S[i, i] = 0
    return S

def train_com_model(train_loader, model, criterion, optimizer):
    '''
    training model to predict complementary labels
    '''
    train_loss = 0
    model.train()
    for i, (images, targets, complementary) in enumerate(train_loader):
        images, targets, complementary = images.to(device), targets.to(device), complementary.to(device)
        outputs = model(images)
        _, com_hot = torch.max(complementary.data, 1)
        loss = criterion(outputs, com_hot.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
    acc = com_accuracy(train_loader, model)
    return train_loss / len(train_loader), acc

def training(train_loader, model, optimizer, epoch, criterion, args):
    train_loss = 0
    T = np.loadtxt(args.T_path, delimiter=',')
    T = torch.from_numpy(T).float()
    T = T.to(device)
    sig = nn.Sigmoid()

    model.train()
    for i, (images, _, complementary, one_label) in enumerate(train_loader):
        images, complementary, one_label = images.to(device), complementary.to(device), one_label.to(device)
        outputs = model(images)

        pre = sig(outputs)
        q = torch.mm(pre, T.transpose(1, 0))

        if args.one_label:
            loss = criterion(q, complementary) + torch.norm(one_label - pre, 2)
        else:
            consist = torch.norm(complementary - q, 2)
            loss = criterion(q, complementary) + consist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

    return train_loss / len(train_loader)


# test the results
def validate(test_loader, model):
    with torch.no_grad():
        model.eval()
        sig = nn.Sigmoid()
        for data, targets, _, _ in test_loader:
            images, targets = data.to(device), targets.to(device)
            output = model(images)
            pre_output = sig(output)
            pre_label = predict(output)

    t_one_error = OneError(pre_output, targets)
    t_converage = Coverage(pre_output, targets)
    t_hamm = HammingLoss(pre_label, targets)
    t_rank = RankingLoss(pre_output, targets)
    t_av_pre = AveragePrecision(pre_output, targets)

    return t_hamm, t_one_error, t_converage, t_rank, t_av_pre

def TModel(args):
    '''
    estimating transition matrix T
    '''
    if args.dataset == 'scene':
        print('Data Preparation of scene')
        file_name = ["data/scene_data.csv", "data/scene_label.csv", "data/scene_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        args.num_class = 6
        args.input_dim = 294
    elif args.dataset == "yeast":
        print('Data Preparation of yeast')
        file_name = ["data/yeast_data.csv", "data/yeast_label.csv", "data/yeast_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        args.num_class = 14
        args.input_dim = 103

    # training a model that can predict complementary labels
    if args.step == "one":
        if args.model == 'linear':
            model = linear(input_dim=args.input_dim, output_dim=args.num_class)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        high_acc = 0

        for epoch in range(args.epochs):
            # with complementary labels, the trained model can predict complementary labels
            train_loss, com_acc = train_com_model(train_loader, model, criterion, optimizer)
            print('Epoch:{}. Tr loss:{}. com acc:{}.'.format(epoch + 1, train_loss, com_acc))
            if com_acc > high_acc:
                high_acc = com_acc
                torch.save(model.state_dict(), "trained_model/com_{ds}_{me}.ckpt".format(ds=args.dataset, me=args.model))

        print('high accuracy:', high_acc)
    else:  # estimating transition matrix T
        if args.model == 'linear':
            model = linear(input_dim=args.input_dim, output_dim=args.num_class)

            # these model is trained in the step one
            if args.dataset == 'scene':
                model_path = "trained_model/com_scene_linear.ckpt"
            elif args.dataset == "yeast":
                model_path = "trained_model/com_yeast_linear.ckpt"

        # according to the saved complementary labeled model, we calculate the initial transition matrix S (Eq. (6))
        S = Train_T(model, model_path, train_loader, test_loader, args.num_class)
        np.savetxt("estimation_T/{}_S.txt".format(args.dataset), S, delimiter=',', fmt='%1.5f')

        # to obtain label correlations
        C = LabelCorrelation(file_name, args.num_class)
        np.savetxt("estimation_T/{}_C.txt".format(args.dataset), C, delimiter=',', fmt='%1.5f')

        # to estimating T
        T = C@S
        for i in range(args.num_class):
            T[i, i] = 0
        T_sum = np.sum(T, 0).reshape(1, -1)
        for i in range(len(T)):
            if T_sum[0, i] != 0:
                T[:, i] = T[:, i] / T_sum[0, i]
        np.savetxt("estimation_T/{}_T.txt".format(args.dataset), T, delimiter=',', fmt='%1.5f')

def main(args):
    print(args)

    if args.dataset == 'scene':
        print('Data Preparation of scene')
        file_name = ["data/scene_data.csv", "data/scene_label.csv", "data/scene_com_label.csv", "data/one_label/scene_one.csv"]
        train_loader, test_loader = dataset.ComFold_onelabel(args.batch_size, file_name, 10, args.fold)
        args.num_class = 6
        args.input_dim = 294
    elif args.dataset == "yeast":
        print('Data Preparation of yeast')
        file_name = ["data/yeast_data.csv", "data/yeast_label.csv", "data/yeast_com_label.csv", "data/one_label/yeast_one.csv"]
        train_loader, test_loader = dataset.ComFold_onelabel(args.batch_size, file_name, 10, args.fold)
        args.num_class = 14
        args.input_dim = 103

    if args.model == 'linear':
        model = linear(input_dim=args.input_dim, output_dim=args.num_class)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    save_table = np.zeros(shape=(args.epochs, 7))
    for epoch in range(args.epochs):
        train_loss = training(train_loader, model, optimizer, epoch, criterion, args)
        t_hamm, t_one_error, t_converage, t_rank, t_av_pre = validate(test_loader, model)
        print("Epoch:{ep}, Tr_loss:{tr}, T_hamm:{T_hamm}, T_rank:{T_rank}, T_one_error:{T_one_error}, T_con:{T_con}, "
              "T_av:{T_av}".format(ep=epoch, tr=train_loss, T_hamm=t_hamm, T_rank=t_rank, T_one_error=t_one_error,
                                                    T_con=t_converage, T_av=t_av_pre))
        save_table[epoch, :] = epoch + 1, train_loss, t_hamm, t_rank, t_one_error, t_converage, t_av_pre
        if not os.path.exists('result/'):
            os.makedirs('result/')
        np.savetxt("result/{ds}_{M}_lr{lr}_wd{wd}_fold{fd}.csv".format(ds=args.dataset, M=args.model,
                                                                            lr=args.lr, wd=args.wd, fd=args.fold),
                   save_table, delimiter=',', fmt='%1.4f')

if __name__ == '__main__':

    lr_3 = ["bookmark15", "delicious15", "eurlex_dc15", "eurlex_sm15"]
    lr_2 = ["Corel16k15", "Corel5k15", "scene"]
    lr_1 = ["yeast"]

    args = parser.parse_args()

    # choosing learning rate for datasets
    if args.dataset in lr_3:
        args.lr = 0.001
    elif args.dataset in lr_2:
        args.lr = 0.01
    else:
        args.lr = 0.1


    if args.stage == "first":  # to estimate transition matrix T
        TModel(args)
    else:  # training multi-labeled classifier
        if args.dataset == "yeast":
            args.T_path = "estimation_T/yeast_T_linear.txt"
        elif args.dataset == "scene":
            args.T_path = "estimation_T/scene_T_linear.txt"

        print("Data:{ds}, Model:{me}, lr:{lr}, fold:{fd}".format(ds=args.dataset, me=args.model, lr=args.lr,
                                                                 fd=args.fold))
        main(args)









