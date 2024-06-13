import numpy as np
import torch

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

def generate_one_labels(targets):
    '''
    To select a relevant label from the set of relevant labels for an instance, which is used to the experiment on Section 5.2
    :param targets: relevant labels of instances
    :return: a matrix that each instance only with a relevant label
    '''
    K = np.size(targets, 1)
    n = np.size(targets, 0)

    truth_one = np.zeros([n, K])
    labels_hat = np.array(targets, dtype=bool)

    for i in range(n):
        candidates = np.arange(K).reshape(1, K)
        mask = labels_hat[i].reshape(1, -1)
        candidates_ = candidates[mask]
        idx = np.random.randint(0, len(candidates_))
        truth_one[i][candidates_[idx]] = 1
    return truth_one

def generate_compl_labels(train_labels, num_com):
    '''
    Generating complementary labels
    :param train_labels: relevant labels
    :param num_com: the number of complementary labels for each instance
    :return: complementary labels of all instances
    '''
    k = np.size(train_labels, 1)
    n = np.size(train_labels, 0)
    comp_Y = np.zeros([n, k])
    labels_hat = np.array(1 - train_labels, dtype=bool)  # ensure the truth labels can't be selected
    for idx in range(n):
        candidates = np.arange(k).reshape(1, k)
        for i in range(num_com):
            mask = labels_hat[idx].reshape(1, -1)
            candidates_ = candidates[mask]
            index = np.random.randint(0, len(candidates_))
            comp_Y[idx][candidates_[index]] = 1
            labels_hat[idx, candidates_[index]] = False  # expecting the selected complementary label
    return comp_Y

files = ['data/yeast_label.csv', "data/scene_label.csv"]
com_name = ['data/yeast_com_label.csv', "data/scene_com1_label.csv"]
one_name = ['data/one_label/yeast_one.csv', "data/one_label/scene_one.csv"]

for i in range(2):
    label = np.genfromtxt(files[i], delimiter=',')
    print(label.shape)
    com_label = generate_compl_labels(label, 1)
    np.savetxt(com_name[i], com_label, delimiter=',')

    one_label = generate_one_labels(label)
    np.savetxt(one_name[i], one_label, delimiter=',')