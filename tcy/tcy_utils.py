import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def load_graph(dataset, k, node_num):
    if k:
        # path = '../tcy_data/{}{}_graph.csv'.format(dataset, k)
        path = '../time_data/' + dataset + '/DTWD_{}.csv'.format(k)
    else:
        # path = '../tcy_data/{}_graph.csv'.format(dataset)
        # path = '../time_data/' + dataset + '/kValue/DTWD_3k.csv'
        path = '../time_data/' + dataset + '/DTWD.csv'


    # data = np.loadtxt('../tcy_data/{}_all.txt'.format(dataset))
    # n, _ = data.shape

    # path = '../time_data/' + dataset + '/' + dataset + '_DTWD.csv'

    n = node_num
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)#Scipy Sparse 矩阵转换成 torch sparse 矩阵

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def divide_dataset(x_train, y_train, percentage):
    nb_classes = len(np.unique(y_train))
    x_train_percentage = []
    y_train_percentage = []
    for i in range(nb_classes):
        # 处理每一个种类
        temp = []
        for j in range(len(x_train)):
            if y_train[j] == i:
                temp.append(x_train[j])
        num = max(int(len(temp)*percentage), 1)  # 最少每一个类别选一个
        for k in range(num):
            x_train_percentage.append(temp[k])
            y_train_percentage.append(i)
    x_train_percentage = np.array(x_train_percentage).astype(float)
    y_train_percentage = np.array(y_train_percentage).astype(int)
    return x_train_percentage, y_train_percentage

class load_data():
    def __init__(self, dataset):
        # file_name = 'E:/data/new_dataset/' + dataset + '/2.same_length/'
        file_name = '../time_data/' + dataset + '/'
        x_train = np.load(file_name + 'train_x.npy')

        y_train = np.load(file_name + 'train_y.npy')
        x_test = np.load(file_name + 'test_x.npy')
        y_test = np.load(file_name + 'test_y.npy')

        # 归一化
        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_test_mean = x_test.mean()
        x_test_std = x_test.std()

        x_train = (x_train - x_train_mean) / (x_train_std)
        x_test = (x_test - x_test_mean) / (x_test_std)

        # x_train, y_train = divide_dataset(x_train, y_train, 0.2)

        self.x = np.concatenate((x_train, x_test), 0).transpose((0, 2, 1))  # np.loadtxt('../tcy_data/'+dataset_name+'_all.txt', dtype=float)
        self.y = np.concatenate((y_train, y_test), 0)  # np.loadtxt('../tcy_data/'+dataset_name+'_label_all.txt', dtype=int)
        #  self.x = np.loadtxt('../tcy_data/{}_all.txt'.format(dataset), dtype=float)
        # # self.x = np.load(path+'_x.npy', allow_pickle=True).astype(float)
        #  self.y = np.loadtxt('../tcy_data/{}_label_all.txt'.format(dataset), dtype=int)
        # # self.y = np.load(path+'_y.npy', allow_pickle=True).astype(int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


