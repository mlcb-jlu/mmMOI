import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_sub_omics(omics, samples):
    return {key: value.loc[samples.index, :] for key, value in omics.items()}


def load_data(dataset, split):
    path = "../dataset/{}/".format(dataset)
    labels = pd.read_csv(os.path.join(path, "label.csv"), sep=',', index_col=0, header=0)
    omics = {'exp': pd.read_csv(os.path.join(path, "exp.csv"), sep=',', index_col=0, header=0).T,
             'methy': pd.read_csv(os.path.join(path, "methy.csv"), sep=',', index_col=0, header=0).T,
             'mirna': pd.read_csv(os.path.join(path, "mirna.csv"), sep=',', index_col=0, header=0).T
            }

    samples_train, samples_test = train_test_split(labels, test_size=0.3, stratify=labels['cluster.id'], random_state=int(split[0]))
    samples_test, samples_val = train_test_split(samples_test, test_size=0.5, stratify=samples_test['cluster.id'],
                                                 random_state=int(split[2]))

    omics_train = get_sub_omics(omics, samples_train)
    omics_val = get_sub_omics(omics, samples_val)
    omics_test = get_sub_omics(omics, samples_test)
    label_train = torch.tensor(np.array(labels.loc[samples_train.index])).reshape(-1)
    label_val = torch.tensor(np.array(labels.loc[samples_val.index])).reshape(-1)
    label_test = torch.tensor(np.array(labels.loc[samples_test.index])).reshape(-1)

    exp_train = torch.from_numpy(omics_train['exp'].values).float()
    exp_val = torch.from_numpy(omics_val['exp'].values).float()
    exp_test = torch.from_numpy(omics_test['exp'].values).float()

    methy_train = torch.from_numpy(omics_train['methy'].values).float()
    methy_val = torch.from_numpy(omics_val['methy'].values).float()
    mehty_test = torch.from_numpy(omics_test['methy'].values).float()

    mirna_train = torch.from_numpy(omics_train['mirna'].values).float()
    mirna_val = torch.from_numpy(omics_val['mirna'].values).float()
    mirna_test = torch.from_numpy(omics_test['mirna'].values).float()

    data_train = {'exp': exp_train, 'methy': methy_train, 'mirna': mirna_train}
    data_val = {'exp': exp_val, 'methy': methy_val, 'mirna': mirna_val}
    data_test = {'exp': exp_test, 'methy': mehty_test, 'mirna': mirna_test}

    return data_train, data_val, data_test, label_train, label_val, label_test
