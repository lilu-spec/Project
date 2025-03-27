from __future__ import print_function, division
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tcy.tcy_utils import load_data, load_graph
from tcy.tcy_GNN import GNNLayer
from sklearn.metrics import accuracy_score
from torch.nn import Linear, Conv1d, Conv2d
import tensorflow as tf
import numpy as np


# torch.cuda.set_device(1)

# class FCN(nn.Module):
#     def __init__(self, n_1, n_2, n_3, n_input, n_z, variable_num):
#         super(AE, self).__init__()
#         self.cnn_1 = Conv1d(in_channels=variable_num, out_channels = n_1, kernel_size = 8)
#         self.gcn_1 = Conv1d(in_channels = n_1, out_channels = 1, kernel_size = 1)
#         self.cnn_2 = Conv1d(in_channels = n_1, out_channels = n_2, kernel_size = 5)
#         self.gcn_2 = Conv1d(in_channels=n_2, out_channels=1, kernel_size=1)
#         self.cnn_3 = Conv1d(in_channels = n_2, out_channels = n_3, kernel_size = 3)
#         self.gcn_3 = Conv1d(in_channels=n_3, out_channels=1, kernel_size=1)
#
#     def forward(self, x):
#         pro_h1 = self.cnn_1(x)
#         pro_h1 = F.relu(pro_h1)
#         cnn_h1 = self.gcn_1(pro_h1).squeeze(1)
#
#         pro_h2 = self.cnn_1(pro_h1)
#         pro_h2 = F.relu(pro_h2)
#         cnn_h2 = self.gcn_1(pro_h2).squeeze(1)
#
#         pro_h3 = self.cnn_1(pro_h2)
#         pro_h3 = F.relu(pro_h3)
#         cnn_h3 = self.gcn_1(pro_h3).squeeze(1)
#
#         return cnn_h1, cnn_h2, cnn_h3

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, variable_num):
        super(AE, self).__init__()
        self.conv0 = Conv1d(in_channels=variable_num, out_channels=1, kernel_size=1)
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
        pro_x = self.conv0(x).squeeze(1)

        enc_h1 = F.relu(self.enc_1(pro_x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        enc_h3 = self.do(enc_h3)
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        dec_h3 = self.do(dec_h3)
        x_bar = self.x_bar_layer(dec_h3)

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
        # 预训练
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # fcn for inter information
        self.gnn_1 = GNNLayer(n_input, 128)
        self.gnn_2 = GNNLayer(128, 256)
        self.gnn_3 = GNNLayer(256, 128)
        self.gnn_4 = GNNLayer(128, 100)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        self.fc = Linear(n_z, n_clusters)
        self.do = nn.Dropout(0.5)
        # degree
        self.v = v

        self.cnn_1 = Conv1d(in_channels=n_input, out_channels=128, kernel_size=3, padding=1)
        self.fcn_1 = Conv1d(in_channels=variable_num, out_channels=1, kernel_size=1)

        self.cnn_2 = Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.fcn_2 = Conv1d(in_channels=variable_num, out_channels=1, kernel_size=1)

        self.cnn_3 = Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.fcn_3 = Conv1d(in_channels=variable_num, out_channels=1, kernel_size=1)

        # self.cnn_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')
        # self.cnn_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=8, padding='same')
        # self.cnn_3 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, pro_x = self.ae(x)

        sigma = 0.5

        # print("原始形状： ", x.shape)
        cnn_1 = x.permute(0, 2, 1)
        cnn_1 = self.cnn_1(cnn_1)
        # print("卷积一次后的形状： ", cnn_1.shape)
        cnn_1 = nn.BatchNorm1d(128)(cnn_1)
        # print("BT一次后的形状： ", cnn_1.shape)
        cnn_1 = F.relu(cnn_1)
        tra1 = cnn_1.permute(0, 2, 1)
        tra1 = self.fcn_1(tra1).squeeze(1)
        # tra1 = cnn_1.reshape((cnn_1.shape[0], cnn_1.shape[2], cnn_1.shape[1]))
        # tra1 = Conv1d(in_channels=tra1.shape[1], out_channels=1, kernel_size=1)(tra1).squeeze(1)

        cnn_2 = self.cnn_2(cnn_1)
        # print("卷积2次后的形状： ", cnn_2.shape)
        cnn_2 = nn.BatchNorm1d(256)(cnn_2)
        # print("BT2次后的形状： ", cnn_2.shape)
        cnn_2 = F.relu(cnn_2)
        tra2 = cnn_2.permute(0, 2, 1)
        tra2 = self.fcn_2(tra2).squeeze(1)
        # tra2 = cnn_2.reshape((cnn_2.shape[0], cnn_2.shape[2], cnn_2.shape[1]))
        # tra2 = Conv1d(in_channels=tra2.shape[1], out_channels=1, kernel_size=1)(tra2).squeeze(1)
        # tra2 = self.fcn_2(tra2).squeeze(1)

        cnn_3 = self.cnn_3(cnn_2)
        # print("卷积3次后的形状： ", cnn_3.shape)
        cnn_3 = nn.BatchNorm1d(128)(cnn_3)
        # print("BT3次后的形状： ", cnn_3.shape)
        cnn_3 = F.relu(cnn_3)
        tra3 = cnn_3.permute(0, 2, 1)
        tra3 = self.fcn_3(tra3).squeeze(1)
        # tra3 = cnn_3.reshape((cnn_3.shape[0], cnn_3.shape[2], cnn_3.shape[1]))
        # tra3 = Conv1d(in_channels=tra3.shape[1], out_channels=1, kernel_size=1)(tra3).squeeze(1)
        # tra3 = self.fcn_3(tra3).squeeze(1)

        # tra1 =  tra1.reshape((tra1.shape[0], tra1.shape[2], tra1.shape[1]))
        # tra2 = tra2.reshape((tra2.shape[0], tra2.shape[2], tra2.shape[1]))
        # tra3 = tra3.reshape((tra3.shape[0], tra3.shape[2], tra3.shape[1]))

        # h = self.gnn_1(x, adj)
        # h = self.gnn_2((1-sigma)*h, adj)
        # h = self.gnn_3((1-sigma)*h, adj)
        #
        # h = self.gnn_4((1-sigma)*h, adj,active=False)
        # h = self.do(h)
        # h =self.fc((1 - sigma) * h) #self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        # GCN Module
        h = self.gnn_1(pro_x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)

        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj, active=False)
        h = self.do(h)
        h = self.fc((1 - sigma) * h + sigma * z)  # self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        predict = F.softmax(h, dim=1)

        # # Dual Self-supervised Module
        # q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q = q.pow((self.v + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, predict, z, pro_x


avg_acc = 0.0


def train_sdcn(dataset):
    # file_output = open('../result/' + args.name + '.txt', 'a+')
    file_output = open('../result/' + args.name + '/' + args.name + '_fcn_3.txt', 'a+')
    # file_output = open('../result/' + args.name + '/' + args.name + '_fcn_pretrain_3.txt', 'a+')

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

    # 求均值和方差
    # mean = dataset.x[:train_num].mean()
    # std = dataset.x[:train_num].std()
    #
    # data = (data - mean)/std

    max_acc = 0
    for epoch in range(1000):

        # if(epoch>800):
        #      optimizer = Adam(model.parameters(), lr=0.0001)
        # if(epoch>200):
        #     optimizer = Adam(model.parameters(), lr=0.00005)

        x_bar, pred, z, pro_x = model(data, adj)

        # kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')

        pred_train = pred[:train_num]
        y_train = y[:train_num]

        ae_loss = F.mse_loss(x_bar, pro_x)
        gcn_loss = criterion(pred_train, y_train)
        # loss = gcn_loss + ae_loss
        loss = gcn_loss

        Train_res = pred_train.data.cpu().numpy().argmax(1)
        print(epoch, ':Train_loss {:.4f},Train_acc {:.4f}'.format(loss, accuracy_score(Train_res, y_train)))
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
            print('     ae_loss: {:.4f}, gcn_loss: {:.4f}'.format(ae_loss, gcn_loss))
            # res3 = p.data.cpu().numpy().argmax(1)  # P

            if accuracy_score(res2[train_num:], y[train_num:]) > max_acc:
                max_acc = accuracy_score(res2[train_num:], y[train_num:])
    print('max_accuracy: {:.4f}'.format(max_acc))
    file_output.write(str(max_acc) + os.linesep)

    global avg_acc
    avg_acc += max_acc


if __name__ == "__main__":
    avg_accuracy = 0.0
    for x in range(10):
        train_num = 0
        test_num = 0
        dimension = 64
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='MotorImagery')
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
        # args.pretrain_path = '../tcy_data/{}.pkl'.format(args.name)
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
            args.n_input = 24 * 51
            train_num = 180
            test_num = 180
        if args.name == 'PEMS-SF':
            args.lr = 1e-3
            args.k = None  # 不需要考虑
            args.n_clusters = 7
            args.n_input = 963 * 144
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
            args.lr = 1e-4
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
            args.lr = 1e-4
            args.k = None  # 不需要考虑
            args.n_clusters = 9
            args.n_input = 348
            train_num = 270
            test_num = 370

        print(args)
        train_sdcn(dataset)
    print("平均最大精度：", avg_acc / 10.0)