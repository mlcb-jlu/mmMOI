import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from data_loader import load_data
from models import Autoencoder, Classifier, ML_GNN
from utils import normalize_weight, eva, fuse_weights_softmax

mseLoss = nn.MSELoss()
from construct_graph import get_graph


def train(dataset, view_list, epochs, i, config, device):
    data_train, data_val, data_test, label_train, label_val, label_test = load_data(dataset, config['split'][i])
    label_val_test = torch.cat((label_val, label_test))

    for view in view_list:
        trial = 0
        while (True):
            model_save_path = "../results/pre_model/{}".format(dataset)
            data_save_path = "../results/data/{}/{}".format(dataset, config['split'][i])
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            if not os.path.exists(data_save_path):
                os.makedirs(data_save_path)

            np.savetxt(data_save_path + "/labels_tr.csv", label_train.cpu().numpy(), delimiter=',')
            np.savetxt(data_save_path + "/labels_val.csv", label_val.numpy(), delimiter=',')
            np.savetxt(data_save_path + "/labels_te.csv", label_test.numpy(), delimiter=',')

            data = torch.cat((data_train[view], data_val[view], data_test[view])).to(device)

            view_train = data_train[view].to(device)
            view_val = data_val[view].to(device)
            view_test = data_test[view].to(device)

            label_train = label_train.to(device)
            label_val_test = label_val_test.to(device)

            feat_dim = data.shape[1]
            if view == 'exp':
                dims = [feat_dim, 4000, 2000, 200]
            elif view == 'methy':
                dims = [feat_dim, 4000, 2000, 200]
            elif view == 'mirna':
                dims = [feat_dim, 300, 300, 200]

            # =============================================== pretrain endecoder ============================
            print("{} {} 预训练...".format(dataset, view))
            autoencoder = Autoencoder(dims).to(device)
            classifier = Classifier(dims[-1], config['num_class']).to(device)

            optimizer_autoencoder = optim.Adam([{'params': autoencoder.parameters(), 'lr': 0.005},
                                                {'params': classifier.parameters(), 'lr': 0.005}])

            best_acc = 1e-12
            bad_count = 0

            for epoch in range(epochs):
                autoencoder.train()
                classifier.train()

                x_pred, z_norm = autoencoder(view_train)
                cl_pred = classifier(z_norm)
                loss_re_x = F.mse_loss(x_pred, view_train)

                loss_cl = F.cross_entropy(cl_pred, label_train)

                x_pred, z_norm = autoencoder(view_val)
                loss_re_x += F.mse_loss(x_pred, view_val)

                x_pred, z_norm = autoencoder(view_test)
                loss_re_x += F.mse_loss(x_pred, view_test)

                loss = loss_re_x + loss_cl

                optimizer_autoencoder.zero_grad()
                loss.backward()
                optimizer_autoencoder.step()

                if epoch % 1 == 0:
                    autoencoder.eval()
                    classifier.eval()
                    with torch.no_grad():
                        x_pred, z_norm = autoencoder(view_val)
                        pred = torch.argmax(classifier(z_norm), dim=1)
                        acc = metrics.accuracy_score(label_val, pred.cpu().numpy())

                        if acc > best_acc:
                            best_acc = acc
                            torch.save(autoencoder.state_dict(), model_save_path + "/{}_AE.pkl".format(config['split'][i]))
                            torch.save(classifier.state_dict(), model_save_path + "/{}_classifier_pre.pkl".format(config['split'][i]))
                        else:
                            bad_count += 1
                    if bad_count >= 200:
                        break

            autoencoder.load_state_dict(torch.load(model_save_path + "/{}_AE.pkl".format(config['split'][i])))
            classifier.load_state_dict(torch.load(model_save_path + "/{}_classifier_pre.pkl".format(config['split'][i])))

            with torch.no_grad():
                autoencoder.eval()
                classifier.eval()
                x_pred, z_norm = autoencoder(view_test)
                pred = torch.argmax(classifier(z_norm), dim=1).cpu().numpy()
                pretrain_acc, pretrain_f1, pretrain_pre, pretrain_rec = eva(label_test, pred)
                print('test_acc:{:.4f}, test_f1:{:.4f}, test_pre:{:.4f}, test_rec:{:.4f}\n'.format(pretrain_acc, pretrain_f1,
                                                                                                   pretrain_pre, pretrain_rec))

            autoencoder.eval()
            # AutoEncoder 中间嵌入划分训练集、验证集、测试集
            with torch.no_grad():
                _, embeddings_train = autoencoder(view_train)
                _, embeddings_val = autoencoder(view_val)
                _, embeddings_test = autoencoder(view_test)
                embeddings = torch.cat((embeddings_train, embeddings_val, embeddings_test))

            adjs_train = [get_graph(embeddings_train.cpu(), config['num_edge'], m).to(device) for m in ['pearson', 'euclidean']]
            adjs_val = [get_graph(embeddings_val.cpu(), config['num_edge'], m).to(device) for m in ['pearson', 'euclidean']]
            adjs_test = [get_graph(embeddings_test.cpu(), config['num_edge'], m).to(device) for m in ['pearson', 'euclidean']]
            adjs = [get_graph(embeddings.cpu(), config['num_edge'], m).to(device) for m in ['pearson', 'euclidean']]

            # =========================================Train=============================================================
            print('{} {} 开始训练...'.format(dataset, view))
            model = ML_GNN(dims[-1], config['num_edge']).to(device)
            classifier_tr = Classifier(dims[-1], config['num_class']).to(device)
            # classifier_tr.load_state_dict(parameters[1])
            classifier_tr.load_state_dict(torch.load(model_save_path + "/{}_classifier_pre.pkl".format(config['split'][i])))

            param_all = [{'params': model.parameters()}, {'params': classifier_tr.parameters()}]
            optimizer_model = optim.Adam(param_all, lr=0.005)  # 0.005

            best_a = [1, 1]
            weights = best_a
            score = [0, 0]
            weights_train = best_a
            weights_val_test = best_a

            with torch.no_grad():
                model.eval()
                z_all, q_all = model(embeddings, adjs, weights, label=False)
                kmeans = KMeans(n_clusters=config['num_class'], n_init=5)
                for v in range(3):
                    y_pred = kmeans.fit_predict(z_all[v].data.cpu().numpy())
                    model.cluster_layer[v].data = torch.tensor(kmeans.cluster_centers_).to(device)
                pseudo_label = y_pred

            bad_count = 0
            best_acc = 1e-12

            for epoch in range(epochs):
                model.train()
                classifier_tr.train()

                z_all = model(embeddings_train, adjs_train, weights, label=True)
                accs = []
                for v in range(2):
                    y_pred = torch.argmax(classifier_tr(z_all[v]), dim=1)
                    acc = metrics.accuracy_score(label_train.cpu(), y_pred.cpu().numpy())
                    accs.append(acc)
                weights_1 = normalize_weight(accs, p=2)

                cl_pred = classifier_tr(z_all[-1])
                loss_cl = F.cross_entropy(cl_pred, label_train)

                # 聚类损失反向传播
                kmeans = KMeans(n_clusters=config['num_class'], n_init=5)
                z_all, q_all = model(embeddings, adjs, weights, label=False)
                nmis = []
                for v in range(2):
                    y_pred = kmeans.fit_predict(z_all[v].detach().cpu().numpy())
                    nmi = nmi_score(pseudo_label, y_pred)
                    nmis.append(nmi)
                weights_2 = normalize_weight(nmis, p=2)

                weights = fuse_weights_softmax(weights_1, weights_2)

                kmeans = KMeans(n_clusters=config['num_class'], n_init=5)
                y_pred = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
                pseudo_label = y_pred

                p = model.target_distribution(q_all[-1])
                loss_kl = sum(F.kl_div(q.log(), p, reduction='batchmean') for q in q_all)

                loss = loss_cl + loss_kl

                optimizer_model.zero_grad()
                loss.backward()
                optimizer_model.step()
                # =========================================evaluation============================================================

                if epoch % 1 == 0:
                    model.eval()
                    classifier_tr.eval()
                    with torch.no_grad():
                        z, _ = model(embeddings_val, adjs_val, weights, label=False)
                        pred = torch.argmax(classifier_tr(z[-1]), dim=1)
                        acc = metrics.accuracy_score(label_val, pred.cpu().numpy())
                        if acc > best_acc:
                            best_acc = acc
                            model.weights = weights
                            torch.save(model.state_dict(), model_save_path + "/{}_model.pkl".format(config['split'][i]))
                            torch.save(classifier_tr.state_dict(),
                                       model_save_path + "/{}_classifier_tr.pkl".format(config['split'][i]))
                        else:
                            bad_count += 1
                    if bad_count >= 50:
                        break

            model.load_state_dict(torch.load(model_save_path + "/{}_model.pkl".format(config['split'][i])))
            classifier_tr.load_state_dict(torch.load(model_save_path + "/{}_classifier_tr.pkl".format(config['split'][i])))

            with torch.no_grad():
                model.eval()
                classifier_tr.eval()
                z, _ = model(embeddings_test, adjs_test, model.weights, label=False)  # 保存 test 表示
                pred = torch.argmax(classifier_tr(z[-1]), dim=1).cpu().numpy()
                train_acc, train_f1, train_pre, train_rec = eva(label_test, pred)
                print('test_acc:{:.4f}, test_f1:{:.4f}, test_pre:{:.4f}, test_rec:{:.4f}\n'.format(train_acc, train_f1,
                                                                                                   train_pre, train_rec))
                if train_acc >= pretrain_acc or trial == 5:
                    print('----------------{} {}第{}组训练结束----------------\n'.format(dataset, view, i + 1))
                    # 保存 test 表示
                    np.savetxt(data_save_path + "/{}_te.csv".format(view), z[-1].cpu().numpy(), delimiter=',')

                    # 保存 train 表示
                    z = model(embeddings_train, adjs_train, model.weights, label=True)
                    np.savetxt(data_save_path + "/{}_tr.csv".format(view), z[-1].cpu().numpy(), delimiter=',')

                    # 保存 val 表示
                    z, _ = model(embeddings_val, adjs_val, model.weights, label=False)
                    np.savetxt(data_save_path + "/{}_val.csv".format(view), z[-1].cpu().numpy(), delimiter=',')
                    break
                else:
                    print(f"Trial {trial} discarded. Restarting from pretraining.")
                    trial += 1


