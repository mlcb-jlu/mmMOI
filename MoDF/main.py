import numpy as np
from train import train_test
import random
import torch
import argparse
import os
from configure import config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='OV', help='datasets: BRCA, GBM, KIPAN, OV')
parser.add_argument('--epochs', type=int, default=300, help='training epochs')
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--mode', type=bool, default=False, help='Traning mode')
parser.add_argument('--seed', type=int, default=42, help='')
args = parser.parse_args()

if __name__ == "__main__": 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    dataset = args.dataset

    view_list = ['exp', 'methy', 'mirna']

    accuracy, F1_score, precision, recall = [], [], [], []
    for i in range(5):
        accs, f1s, pres, recs = [], [], [], []
        for j in range(5):
            print("第" + str(i + 1) + "组第" + str(j + 1) + "次")

            acc, f1, pre, rec = train_test(dataset, view_list, args.epochs, config[dataset], i, j, device, args.mode)
            accs.append(acc)
            f1s.append(f1)
            pres.append(pre)
            recs.append(rec)

        accuracy.append(np.mean(accs))
        F1_score.append(np.mean(f1s))
        precision.append(np.mean(pres))
        recall.append(np.mean(recs))

    print("======== {} Final Results ========".format(dataset))
    print(f"Accuracy:  {np.mean(accuracy):.4f} ± {np.std(accuracy):.3f}")
    print(f"F1_macro:  {np.mean(F1_score):.4f} ± {np.std(F1_score):.3f}")
    print(f"Precision: {np.mean(precision):.4f} ± {np.std(precision):.3f}")
    print(f"Recall:    {np.mean(recall):.4f} ± {np.std(recall):.3f}")