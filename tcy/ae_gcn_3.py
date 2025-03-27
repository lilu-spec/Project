from __future__ import print_function, division
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.nn.parameter import Parameter
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from tcy.tcy_utils import load_data, load_graph
from tcy.tcy_GNN import  GNNLayer
from sklearn.metrics import accuracy_score
from torch.nn import Linear, Conv1d
import matplotlib.pyplot as plt
import numpy as np



# torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, variable_num):
        super(AE, self).__init__()
        self.conv0 = Conv1d(in_channels=variable_num, out_channels=1, kernel_size=3, padding=1)
        self.conv1 = Conv1d(in_channels=1, out_channels=variable_num, kernel_size=3, padding=1)

        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

        self.do = nn.Dropout(0.5)

    def forward(self, x):
        pro_x = self.conv0(x).squeeze(dim=1)

        enc_h1 = F.relu(self.enc_1(pro_x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        enc_h3 = self.do(enc_h3)

        z = self.z_layer(enc_h3)
        # z = self.do(z)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        # dec_h3 = self.do(dec_h3)

        x_bar = F.relu(self.x_bar_layer(dec_h3))
        x_bar = self.do(x_bar)

        x_bar = self.conv1(x_bar.unsqueeze(dim=1))

        # enc_h1 = F.relu(self.enc_1(x))
        # enc_h1 =self.do(enc_h1)
        # enc_h2 = F.relu(self.enc_2(enc_h1))
        # enc_h2 = self.do(enc_h2)
        # enc_h3 = F.relu(self.enc_3(enc_h2))
        # enc_h3 = self.do(enc_h3)
        # z = self.z_layer(enc_h3)
        #
        # dec_h1 = F.relu(self.dec_1(z))
        # dec_h1 =self.do(dec_h1)
        # dec_h2 = F.relu(self.dec_2(dec_h1))
        # dec_h2 = self.do(dec_h2)
        # dec_h3 = F.relu(self.dec_3(dec_h2))
        # dec_h3 = self.do(dec_h3)
        # x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, pro_x


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, variable_num=None):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z,
            variable_num=variable_num
            )
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        self.fc = Linear(n_z, n_clusters)
        self.do = nn.Dropout(0.5)
        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, pro_x = self.ae(x)

        sigma = 0.3


        # h = self.gnn_1(x, adj)
        # h = self.gnn_2((1-sigma)*h, adj)
        # h = self.gnn_3((1-sigma)*h, adj)
        #
        # h = self.gnn_4((1-sigma)*h, adj,active=False)
        # h = self.do(h)
        # h =self.fc((1 - sigma) * h) #self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        # GCN Module
        h = self.gnn_1(pro_x, adj)

        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)

        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)

        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj, active=False)
        h = self.do(h)
        h = self.fc((1 - sigma) * h + sigma * z) #self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        predict = F.softmax(h, dim=1)

        # # Dual Self-supervised Module
        # q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q = q.pow((self.v + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, predict, z, pro_x

avg_acc = 0.0
index = 0
def train_sdcn(dataset):
    # file_output = open('../result/' + args.name + '/' + args.name + '_3.txt', 'a+')
    file_output = open('../result/' + args.name + '/' + args.name + '_pretrain_3.txt', 'a+')

    global_feature = np.zeros((test_num, 10))
    y_pred = np.zeros(test_num)
    y_true = np.zeros(test_num)

    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0,
                 variable_num=dimension
                 ).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
    #                                            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
    #                                            min_lr=0, eps=1e-08)

    # KNN Graph
    adj = load_graph(args.name, args.k, train_num + test_num)
    # adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = torch.LongTensor(dataset.y)

    #求均值和方差
    # mean = dataset.x[:train_num].mean()
    # std = dataset.x[:train_num].std()
    #
    # data = (data - mean)/std


    max_acc = 0
    for epoch in range(2000):

        # if(epoch>800):
        #      optimizer = Adam(model.parameters(), lr=0.0001)
        # if(epoch>200):
        #     optimizer = Adam(model.parameters(), lr=0.00005)

        x_bar, pred, z, pro_x = model(data, adj)

        # kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')

        pred_train = pred[:train_num]
        y_train = y[:train_num]

        ae_loss = F.mse_loss(x_bar[:train_num], data[:train_num])
        gcn_loss = criterion(pred_train, y_train)
        loss = gcn_loss + ae_loss

        Train_res = pred_train.data.cpu().numpy().argmax(1)
        print(epoch, ':Train_loss {:.4f},Train_acc {:.4f}'.format(loss, accuracy_score(Train_res, y_train)))
        print('     ae_loss: {:.4f}, gcn_loss: {:.4f}'.format(ae_loss, gcn_loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.step(loss)

        if epoch % 1 == 0:
            # update_interval
            _, pred, _, _ = model(data, adj)
            # tmp_q = tmp_q.data
            # p = target_distribution(tmp_q)

            # res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            print(epoch, ':acc {:.4f}'.format(accuracy_score(res2[train_num:], y[train_num:])))
            # res3 = p.data.cpu().numpy().argmax(1)  # P

            if accuracy_score(res2[train_num:], y[train_num:]) > max_acc:
                max_acc = accuracy_score(res2[train_num:], y[train_num:])

                global_feature = pred.data.cpu().numpy()[train_num:]
                y_pred = res2[train_num:]
                y_true = np.array(y[train_num:])

    print('max_accuracy: {:.4f}'.format(max_acc))
    file_output.write(str(max_acc) + os.linesep)

    global avg_acc
    avg_acc += max_acc

    # global index
    # if index == 0:
    #     print("--------------------", global_feature.shape, y_true.shape, y_pred.shape)
    #     ts = TSNE(n_components=2, init='pca', random_state=0)
    #     result = ts.fit_transform(global_feature)
    #     x_min, x_max = np.min(result, 0), np.max(result, 0)
    #     data = (result - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    #     fig = plt.figure(figsize=(16, 8))  # 创建图形实例
    #     ax = plt.subplot(121)  # 创建子图
    #
    #     flag = np.zeros(10, dtype=np.int32)
    #     for i in range(10):
    #         flag[i] = -1
    #
    #     cnt = 0
    #     for i in range(data.shape[0]):
    #         if cnt == int(y_pred[i] % 10):
    #             plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(y_pred[i] / 10), label=int(y_pred[i] % 10))
    #             cnt += 1
    #         else:
    #             plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(y_pred[i] / 10))
    #
    #     plt.legend()
    #     plt.xlim((0, 1.2))  # 指定坐标的刻度
    #     plt.ylim((0, 1.2))
    #     plt.title('t-SNE Embedding of y_pred', fontsize=20)
    #
    #     for i in range(10):
    #         flag[i] = -1
    #     ax = plt.subplot(122)
    #     cnt = 0
    #     for i in range(data.shape[0]):
    #         if cnt == int(y_true[i] % 10):
    #             plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(y_true[i] / 10), label=int(y_true[i] % 10))
    #             cnt += 1
    #         else:
    #             plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(y_true[i] / 10))
    #     plt.legend()
    #     plt.xlim((0, 1.2))  # 指定坐标的刻度
    #     plt.ylim((0, 1.2))
    #     plt.title('t-SNE Embedding of y_true', fontsize=20)
    #     plt.savefig("pendigit.pdf", bbox_inches="tight")
    #     plt.show()
    #     index = index + 1


if __name__ == "__main__":

    start = time.clock()
    for x in range(10):
        total = 0
        train_num = 0
        test_num = 0
        dimension = 4
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--name', type=str, default='StandWalkJump')
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=100, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))

        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = '../result/' + args.name + '/{}.pkl'.format(args.name)
        dataset = load_data(args.name)

        if args.name == 'ArticularyWordRecognition':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 25
            args.n_input = 144
            train_num = 275
            test_num = 300
        if args.name == 'AtrialFibrillation':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 3
            args.n_input = 640
            train_num = 15
            test_num = 15
        if args.name == 'BasicMotions':  # 最好加上预训练
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 4
            args.n_input = 100
            train_num = 40
            test_num = 40
        if args.name == 'HandMovementDirection':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 4
            args.n_input = 400
            train_num = 160
            test_num = 74
        if args.name == 'Heartbeat':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 2
            args.n_input = 405
            train_num = 204
            test_num = 205
        if args.name == 'MotorImagery':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 2
            args.n_input = 3000
            train_num = 278
            test_num = 100
        if args.name == 'NATOPS':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 6
            args.n_input = 51
            train_num = 180
            test_num = 180
        if args.name == 'PEMS-SF':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 7
            args.n_input = 144
            train_num = 267
            test_num = 173
        if args.name == 'PenDigits':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 10
            args.n_input = 8
            train_num = 7494
            test_num = 3498
        if args.name == 'SelfRegulationSCP2':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 2
            args.n_input = 1152
            train_num = 200
            test_num = 180
        if args.name == 'StandWalkJump':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 3
            args.n_input = 2500
            train_num = 12
            test_num = 15
        if args.name == 'JapaneseVowels':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 9
            args.n_input = 29
            train_num = 270
            test_num = 370
        if args.name == 'Epilepsy':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            train_num = 137
            test_num = 138
            args.n_input = 206
            args.n_clusters = 4
        if args.name == 'Libras':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            train_num = 180
            test_num = 180
            args.n_input = 45
            args.n_clusters = 15
        if args.name == 'RacketSports':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            train_num = 151
            test_num = 152
            args.n_input = 30
            args.n_clusters = 4
        if args.name == 'ERing':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            train_num = 30
            test_num = 270
            args.n_input = 65
            args.n_clusters = 6
        print(args)

        train_num = (len(dataset.x) - test_num)
        print("--------", train_num)

        train_sdcn(dataset)

    end = time.clock()
    print("The function run time is : %.03f seconds" % ((end - start) / 10.0))
    print("20% 训练集的平均最大精度：", avg_acc / 10.0)


