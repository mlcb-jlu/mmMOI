import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from construct_graph import normalize_adj

class Autoencoder(nn.Module):
    def __init__(self, dims, dropout=0.2):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(dims[1], dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(dims[2], dims[3]),
            nn.BatchNorm1d(dims[3]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dims[3], dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(dims[2], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(dims[1], dims[0]),
            nn.Sigmoid()  # 仅当输入数据是归一化后的
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class ML_GNN(nn.Module):
    def __init__(self, latent_dim, num_class):
        super().__init__()
        self.dim_list = [200, 200]
        self.gnn = GCN(latent_dim, self.dim_list, dropout=0.2)

        # 聚类中心,每个视图+总聚类中心
        self.cluster_layer = [Parameter(torch.Tensor(num_class, latent_dim)) for _ in range(2)]
        self.cluster_layer.append(Parameter(torch.Tensor(num_class, latent_dim)))
        for v in range(3):
            self.register_parameter('centroid_{}'.format(v), self.cluster_layer[v])

    def forward(self, x, adjs, weights, label, threshold=0.8):
        x = F.normalize(x, p=2, dim=1)
        omega = torch.mm(x, x.T)
        omega[omega > threshold] = 1
        omega[omega <= threshold] = 0
        adjs_refine = [omega + adjs[i] for i in range(2)]
        S = sum(weights[v] * adjs_refine[v] for v in range(2))
        S = normalize_adj(S)

        z_all = []
        q_all = []

        if label:
            for v in range(2):
                z_norm = self.gnn(x, adjs[v])
                z_all.append(z_norm)
            z_all.append(z_norm)
            return z_all
        else:
            for v in range(2):
                z_norm = self.gnn(x, adjs[v])
                z_all.append(z_norm)
                q = self.predict_distribution(z_norm, v)
                q_all.append(q)
            z_norm = self.gnn(x, S)
            z_all.append(z_norm)
            q = self.predict_distribution(z_norm, -1)
            q_all.append(q)
            return z_all, q_all

    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def target_distribution(self, q):
        weight = q ** 2/ q.sum(0)
        return (weight.t() / weight.sum(1)).T


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gc2(x, adj)
        x = F.normalize(x, p=2, dim=1)

        return x
