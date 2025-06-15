import os
from train import train
import torch
import argparse
from configure import config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BRCA', help='datasets: BRCA, GBM, KIPAN, OV')
parser.add_argument('--epochs', type=int, default=500, help='training epochs')
parser.add_argument('--cuda_device', type=int, default=0, help='')
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    dataset = args.dataset

    view_list = ['exp', 'methy', 'mirna']

    for i in range(5):
        train(dataset, view_list, args.epochs, i, config[dataset], device)

