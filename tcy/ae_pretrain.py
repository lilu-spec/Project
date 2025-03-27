import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear, Conv1d
from torch.utils.data import Dataset


# torch.cuda.set_device(3)


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
        self.do=nn.Dropout(0.5)

    def forward(self, x):
        pro_x = self.conv0(x).squeeze(dim=1)

        enc_h1 = F.relu(self.enc_1(pro_x))

        enc_h2 = F.relu(self.enc_2(enc_h1))

        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))

        dec_h2 = F.relu(self.dec_2(dec_h1))

        dec_h3 = F.relu(self.dec_3(dec_h2))

        x_bar = F.relu(self.x_bar_layer(dec_h3))

        x_bar = self.conv1(x_bar.unsqueeze(dim=1))

        return x_bar, z, pro_x


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset):
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            # x = x.cuda()

            x_bar, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)#.cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # x = torch.Tensor(dataset.x)
            x = torch.Tensor(dataset.x)#.cuda().float()
            x_bar, z, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))

        torch.save(model.state_dict(), '../result/' + dataset_name + '/' + dataset_name+'.pkl')


#
# dataset_name = "ArticularyWordRecognition"
# train_num = 275
# test_num = 300
# mts_dimension = 9
# mts_length = 144
# class_num = 25

# dataset_name = "AtrialFibrillation"
# train_num = 15
# test_num = 15
# mts_dimension = 2
# mts_length = 640
# class_num = 3
#
# dataset_name = "BasicMotions"
# train_num = 40
# test_num = 40
# mts_dimension = 6
# mts_length = 100
# class_num = 4

# dataset_name = "HandMovementDirection"
# train_num = 160
# test_num = 74
# mts_dimension = 10
# mts_length = 400
# class_num = 4

dataset_name = "Heartbeat"
train_num = 204
test_num = 205
class_num = 2
mts_dimension = 61
mts_length = 405

# dataset_name = "MotorImagery"
# train_num = 278
# test_num = 100
# mts_dimension = 64
# mts_length = 3000
# class_num = 2

# dataset_name = "NATOPS"
# train_num = 180
# test_num = 180
# class_num = 6
# mts_dimension =24
# mts_length = 51

# dataset_name = "PEMS-SF"
# train_num = 267
# test_num = 173
# class_num = 7
# mts_dimension = 963
# mts_length = 144

# dataset_name = "PenDigits"
# train_num = 7494
# test_num = 3498
# mts_dimension = 2
# mts_length = 8
# class_num = 10

# dataset_name = "SelfRegulationSCP2"
# train_num = 200
# test_num = 180
# class_num = 2
# mts_dimension = 7
# mts_length = 1152

# dataset_name = "StandWalkJump"
# train_num = 12
# test_num = 15
# mts_dimension = 4
# mts_length = 2500
# class_num = 3

# dataset_name = "JapaneseVowels"
# train_num = 270
# test_num = 370
# mts_dimension = 12
# mts_length = 29
# class_num = 9

# dataset_name = "Epilepsy"
# train_num = 137
# test_num = 138
# mts_dimension = 3
# mts_length = 206
# class_num = 4

# dataset_name = "Libras"
# train_num = 180
# test_num = 180
# mts_dimension = 2
# mts_length = 45
# class_num = 15

# dataset_name = "RacketSports"
# train_num = 151
# test_num = 152
# mts_dimension = 6
# mts_length = 30
# class_num = 4

# dataset_name = "ERing"
# train_num = 30
# test_num = 270
# mts_dimension = 4
# mts_length = 65
# class_num = 6

model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=mts_length,
         n_z=100,
        variable_num=mts_dimension)
file_name = '../time_data/' + dataset_name + '/'
x_train = np.load(file_name + 'train_x.npy')
# y_train = np.load(file_name + 'train_y.npy')
# x_test = np.load(file_name + 'test_x.npy')
# y_test = np.load(file_name + 'test_y.npy')
x = x_train.transpose((0, 2, 1))
# x = np.concatenate((x_train, x_test), 0).transpose((0, 2, 1))#np.loadtxt('../tcy_data/'+dataset_name+'_all.txt', dtype=float)
# y = np.concatenate((y_train, y_test), 0)#np.loadtxt('../tcy_data/'+dataset_name+'_label_all.txt', dtype=int)

# 求均值和方差
mean = x.mean()
std = x.std()
x = (x - mean) / std

dataset = LoadDataset(x)
pretrain_ae(model, dataset)
