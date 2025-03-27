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
import numpy as np
import tensorflow as tf

# torch.cuda.set_device(1)

def getData(x, n_input):
    x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
    x = Conv1d(in_channels=n_input, out_channels=128, kernel_size=3,padding=1)(x)
    x = nn.BatchNorm1d(128)(x)
    x = F.relu(x)

    x = Conv1d(in_channels=128, out_channels=256, kernel_size=1)(x)
    x = nn.BatchNorm1d(256)(x)
    x = F.relu(x)

    x = Conv1d(in_channels=256, out_channels=n_input, kernel_size=3,padding=1)(x)
    x = nn.BatchNorm1d(n_input)(x)
    x = F.relu(x)

    x = x.reshape(x.shape[0], x.shape[2], x.shape[1])

    # x = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation(activation='relu')(x)
    #
    # x = tf.keras.layers.Conv1D(filters=256, kernel_size=8, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation(activation='relu')(x)
    #
    # x = tf.keras.layers.Conv1D(filters=n_input, kernel_size=8, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation(activation='relu')(x)

    # x = tf.keras.layers.GlobalAveragePooling1D()(x)

    return x

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, variable_num):
        super(AE, self).__init__()
        self.conv0 = Conv1d(in_channels=variable_num, out_channels=1, kernel_size=1)
        # self.conv1 = Conv1d(in_channels=n_input, out_channels=128, kernel_size=1)
        # self.conv2 = Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        # self.conv3 = Conv1d(in_channels=256, out_channels=n_input, kernel_size=1)
        # self.batN = nn.BatchNorm1d(n_input)
        # self.conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')
        # self.conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5,  padding='same')
        # self.conv3 = tf.keras.layers.Conv1D(filters=1287, kernel_size=3, padding='same')

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

        # print("原始形状： ", x.shape)
        # pro_x = x.permute(0, 2, 1)
        # pro_x = self.conv1(pro_x)
        # pro_x = nn.BatchNorm1d(128)(pro_x)
        # # pro_x = tf.keras.layers.BatchNormalization()(pro_x)
        # # pro_x = tf.keras.layers.Activation(activation='relu')(pro_x)
        # pro_x = F.relu(pro_x)
        # # print("卷积一次后的形状： ", pro_x.shape)
        #
        # pro_x = self.conv2(pro_x)
        # pro_x = nn.BatchNorm1d(256)(pro_x)
        # pro_x = F.relu(pro_x)
        # # print("卷积2次后的形状： ", pro_x.shape)
        #
        # pro_x = self.conv3(pro_x)
        # pro_x = self.batN(pro_x)
        # pro_x = F.relu(pro_x)
        # # print("卷积3次后的形状： ", pro_x.shape)

        pro_x = x
        pro_x = self.conv0(pro_x).squeeze(1)

        # pro_x = nn.AdaptiveAvgPool2d((1, x.shape[2]))(pro_x)
        # # print("池化后的形状： ", pro_x.shape)
        # pro_x = pro_x.squeeze(1)
        # print("压缩后的形状： ", pro_x.shape)



        # print("卷积后的形状： ",self.conv0(x).shape)
        # pro_x = self.conv0(x).squeeze(1)
        # print("压缩后的形状： ", pro_x.shape)

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

        sigma = 0.5

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
    # file_output = open('../result/' + args.name + '/' + args.name + '_cnn_3.txt', 'a+')
    file_output = open('../result/' + args.name + '/' + args.name + '_cnn_pretrain_3.txt', 'a+')

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
    # x = getData(dataset.x, args.n_input)
    data = torch.Tensor(dataset.x).to(device)
    data = getData(data, args.n_input)
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

        ae_loss = F.mse_loss(x_bar[:train_num], pro_x[:train_num])
        gcn_loss = criterion(pred_train, y_train)
        loss = gcn_loss + ae_loss

        Train_res = pred_train.data.cpu().numpy().argmax(1)
        print(epoch, ':Train_loss {:.4f},Train_acc {:.4f}'.format(loss, accuracy_score(Train_res, y_train)))
        print('     ae_loss: {:.4f}, gcn_loss: {:.4f}'.format(ae_loss, gcn_loss))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
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
    print('max_accuracy: {:.4f}'.format(max_acc))
    file_output.write(str(max_acc) + os.linesep)

    global avg_acc
    avg_acc += max_acc


if __name__ == "__main__":
    avg_accuracy = 0.0
    for x in range(10):
        train_num = 0
        test_num = 0
        dimension = 2
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='PenDigits')
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
            args.n_input = 29
            train_num = 270
            test_num = 370

        print(args)
        train_sdcn(dataset)
    print("平均最大精度：", avg_acc / 10.0)