import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim

def prepare_trte_data(dataset, view_list, split, device):
    data_path = "../results/data/{}/{}".format(dataset, split)
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_path, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_path, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_path, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_path, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        data_tensor_list[i] = data_tensor_list[i].to(device)
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels


def prepare_trval_data(dataset, view_list, split, device):
    data_path = "../results/data/{}/{}".format(dataset, split)
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_path, "labels_tr.csv"), delimiter=',')
    labels_val = np.loadtxt(os.path.join(data_path, "labels_val.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_val = labels_val.astype(int)
    data_tr_list = []
    data_val_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_path, str(i)+"_tr.csv"), delimiter=','))
        data_val_list.append(np.loadtxt(os.path.join(data_path, str(i)+"_val.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_val = data_val_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_val_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        data_tensor_list[i] = data_tensor_list[i].to(device)
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["val"] = list(range(num_tr, (num_tr+num_val)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["val"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_val))

    return data_train_list, data_all_list, idx_dict, labels


def train_epoch(data_list, label, model_dict, optim_dict):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()

    optim_dict["C"].zero_grad()

    atten_data_list = model_dict["MOAM"](data_list)
    new_data = torch.cat([atten_data_list[0], atten_data_list[1], atten_data_list[2]], dim=1)
    c = model_dict["OIRL"](new_data)
    c_loss = torch.mean(criterion(c, label))
    c_loss.backward()
    optim_dict["C"].step()
    loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict


def test_epoch(data_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    atten_data_list = model_dict["MOAM"](data_list)
    new_data = torch.cat([atten_data_list[0], atten_data_list[1], atten_data_list[2]], dim=1)
    c = model_dict["OIRL"](new_data)
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob


def train_test(dataset, view_list, num_epoch, config, i, j, device, mode):
    test_inverval = 5
    save_path = "../results/model/{}/".format(dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(dataset, view_list, config['split'][i], device)
    data_tr_list, data_trval_list, trval_idx, labels_trval = prepare_trval_data(dataset, view_list, config['split'][i], device)

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]]).to(device)

    input_data_dim = [200, 200, 200]
    model_dict = init_model_dict(input_data_dim, config, device)
    for m in model_dict:
        model_dict[m].to(device)

    if mode:
        accuracy_score1 = 0
        optim_dict = init_optim(model_dict, config['lr'])
        for epoch in range(num_epoch):
            train_epoch(data_tr_list, labels_tr_tensor, model_dict, optim_dict)
            if epoch % test_inverval == 0:
                val_prob = test_epoch(data_trval_list,  trval_idx["val"], model_dict)
                if(accuracy_score(labels_trval[trval_idx["val"]], val_prob.argmax(1)) > accuracy_score1):
                    torch.save(model_dict, '../results/model/{}/{}_{}_best_model.pkl'.format(dataset, i, j))
                    accuracy_score1 = accuracy_score(labels_trval[trval_idx["val"]], val_prob.argmax(1))

    with torch.no_grad():
        # model_dict_new = torch.load('../results/model/{}/{}_{}_best_model.pkl'.format(dataset, i, j))
        model_dict_new = torch.load('../results/model/{}/{}_{}_best_model.pkl'.format(dataset, i, j), map_location='cuda:0')
        te_prob_new = test_epoch(data_trte_list, trte_idx["te"], model_dict_new)

        te_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob_new.argmax(1))
        te_F1 = f1_score(labels_trte[trte_idx["te"]], te_prob_new.argmax(1), average='macro')
        te_pre = precision_score(labels_trte[trte_idx["te"]], te_prob_new.argmax(1), average='macro', zero_division=0)
        te_rec = recall_score(labels_trte[trte_idx["te"]], te_prob_new.argmax(1), average='macro')
        print("Accuracy:{}, F1_macro:{}, precision:{}, recall:{}".format(te_acc, te_F1, te_pre, te_rec))

        return te_acc, te_F1, te_pre, te_rec

