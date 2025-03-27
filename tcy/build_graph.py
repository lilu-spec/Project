import numpy as np
import csv
import networkx as nx
from scipy.sparse import csr_matrix
import tensorflow as tf
import sys
import math
# from tf_geometric.utils.graph_utils import convert_edge_to_directed, remove_self_loop_edge

class SAX_trans:

    def __init__(self, ts, w, alpha):
        self.ts = ts
        self.w = w
        self.alpha = alpha
        self.aOffset = ord('a')  # 字符的起始位置，从a开始
        self.breakpoints = {'3': [-0.43, 0.43],
                            '4': [-0.67, 0, 0.67],
                            '5': [-0.84, -0.25, 0.25, 0.84],
                            '6': [-0.97, -0.43, 0, 0.43, 0.97],
                            '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],

                            }
        self.beta = self.breakpoints[str(self.alpha)]

    def normalize(self):  # 正则化
        X = np.asanyarray(self.ts)
        return (X - np.nanmean(X)) / np.nanstd(X)

    def paa_trans(self):  # 转换成paa
        tsn = self.normalize()  # 类内函数调用：法1：加self：self.normalize()   法2：加类名：SAX_trans.normalize(self)
        paa_ts = []
        n = len(tsn)
        xk = math.ceil(n / self.w)  # math.ceil()上取整，int()下取整
        for i in range(0, n, xk):
            temp_ts = tsn[i:i + xk]
            paa_ts.append(np.mean(temp_ts))
            i = i + xk
        return paa_ts

    def to_sax(self):  # 转换成sax的字符串表示
        tsn = self.paa_trans()
        len_tsn = len(tsn)
        len_beta = len(self.beta)
        strx = ''
        for i in range(len_tsn):
            letter_found = False
            for j in range(len_beta):
                if np.isnan(tsn[i]):
                    strx += '-'
                    letter_found = True
                    break
                if tsn[i] < self.beta[j]:
                    strx += chr(self.aOffset + j)
                    letter_found = True
                    break
            if not letter_found:
                strx += chr(self.aOffset + len_beta)
        return strx

    def compare_Dict(self):  # 生成距离表
        num_rep = range(self.alpha)  # 存放下标
        letters = [chr(x + self.aOffset) for x in num_rep]  # 根据alpha，确定字母的范围
        compareDict = {}
        len_letters = len(letters)
        for i in range(len_letters):
            for j in range(len_letters):
                if np.abs(num_rep[i] - num_rep[j]) <= 1:
                    compareDict[letters[i] + letters[j]] = 0
                else:
                    high_num = np.max([num_rep[i], num_rep[j]]) - 1
                    low_num = np.min([num_rep[i], num_rep[j]])
                    compareDict[letters[i] + letters[j]] = self.beta[high_num] - self.beta[low_num]
        return compareDict

    def dist(self, strx1, strx2):  # 求出两个字符串之间的mindist()距离值
        len_strx1 = len(strx1)
        len_strx2 = len(strx2)
        com_dict = self.compare_Dict()

        if len_strx1 != len_strx2:
            print("The length of the two strings does not match")
        else:
            list_letter_strx1 = [x for x in strx1]
            list_letter_strx2 = [x for x in strx2]
            mindist = 0.0
            for i in range(len_strx1):
                if list_letter_strx1[i] is not '-' and list_letter_strx2[i] is not '-':
                    mindist += (com_dict[list_letter_strx1[i] + list_letter_strx2[i]]) ** 2
            mindist = np.sqrt((len(self.ts) * 1.0) / (self.w * 1.0)) * np.sqrt(mindist)
            return mindist


def SAX(signal_1, signal_2):
    # 这里的距离度量标准如下：（应该是1NN-ED）
    # 转化
    signal_1 = np.array(signal_1).astype(float)
    signal_2 = np.array(signal_2).astype(float)
    distance = 0

    for k in range(0, signal_1.shape[1]):
        x1 = SAX_trans(ts=signal_1[:, k], w=6, alpha=3)
        x2 = SAX_trans(ts=signal_2[:, k], w=6, alpha=3)
        st1 = x1.to_sax()
        st2 = x2.to_sax()
        dist = x1.dist(st1, st2)
        distance += dist
    return distance


def DTWD(signal_1, signal_2):
    globalDTW = np.zeros((len(signal_1) + 1, len(signal_2) + 1))
    globalDTW[1:, 0] = math.inf
    globalDTW[0, 1:] = math.inf

    globalDTW[0, 0] = 0
    for i in range(1, len(signal_1) + 1):
        for j in range(1, len(signal_2) + 1):
            # 这里的距离度量标准如下：（应该是1NN-DTW-D）
            # np.sum(np.square(np.array(signal_1[i - 1]).astype(float) - np.array(signal_2[j - 1]).astype(float))) 两个一维数组的差的平方之和
            # 后期如果遇到相关论文的明确标准再进行修改
            globalDTW[i, j] = np.sum(np.square(np.array(signal_1[i - 1]).astype(float) - np.array(signal_2[j - 1]).astype(float))) + min(globalDTW[i - 1, j],globalDTW[i, j - 1],globalDTW[i - 1, j - 1])
    return np.sqrt(globalDTW[len(signal_1), len(signal_2)])

def DTWI(signal_1, signal_2):
    # 这里的距离度量标准如下：（应该是1NN-DTW-i）
    # 转化
    signal_1=np.array(signal_1).astype(float)
    signal_2=np.array(signal_2).astype(float)
    distance=0
    # 维度遍历
    for k in range(0,signal_1.shape[1]):
        # 求在每个相同的维度下，两条“单变量时间序列”之间的DTW距离
        globalDTW = np.zeros((signal_1.shape[0] + 1, signal_2.shape[0] + 1))
        globalDTW[1:, 0] = math.inf
        globalDTW[0, 1:] = math.inf
        globalDTW[0, 0] = 0
        for i in range(1, signal_1.shape[0] + 1):
            for j in range(1, signal_2.shape[0] + 1):
                # 差的平方和
                globalDTW[i, j] = np.square(signal_1[i-1,k]-signal_2[j-1,k]) + min(globalDTW[i - 1, j], globalDTW[i, j - 1], globalDTW[i - 1, j - 1])
        distance += np.sqrt(globalDTW[signal_1.shape[0], signal_2.shape[0]])
    return distance


# 欧式距离建边
def educlidean_edge(signal_1, signal_2):
    signal_1 = np.array(signal_1).astype(float)
    signal_2 = np.array(signal_2).astype(float)
    distance = 0
    min_len = min(signal_1.shape[0], signal_2.shape[0])
    max_len = max(signal_1.shape[0], signal_2.shape[0])
    for i in range(0, min_len):
        # 差的平方和
        distance += np.sqrt(np.sum(np.square(signal_1[i] - signal_2[i])))
    if signal_1.shape[0] > signal_2.shape[0]:
        for j in range(min_len, max_len):
            distance += np.sqrt(np.sum(np.square(signal_1[j])))
    elif signal_1.shape[0] < signal_2.shape[0]:
        for j in range(min_len, max_len):
            distance += np.sqrt(np.sum(np.square(signal_2[j])))
    return distance

def matrix_matrix_cosine(arr, brr):
    # return arr.dot(brr.T).diagonal() / ((np.sqrt(np.sum(arr * arr, axis=1))) * np.sqrt(np.sum(brr * brr, axis=1)))
    return np.sum(arr*brr, axis=1) / (np.sqrt(np.sum(arr**2, axis=1)) * np.sqrt(np.sum(brr**2, axis=1)))

# 利用余弦相似度建边,值越大则相似度越高
def cosine_edge(x, y):
    X = matrix_matrix_cosine(x, y)
    sum = 0
    for i in range(len(X)):
        sum += X[i]
    return sum

def read_csv(filepath,sample_num):
    res = []
    for i in range(sample_num):
        with open(filepath + str(i + 1) + '.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            sample = []
            for row in reader:
                sample.append(row)
        res.append(sample)
    return res

dataset = 'HandMovementDirection'
root ='D:/001--TS-DataSet/' + dataset + '/'
trainset_num= 160
testset_num= 74
train_x=read_csv(root + '1.original/train/train', trainset_num)
train_x = np.array(train_x)
print(train_x.shape)
train_y=np.loadtxt(root + '1.original/train/train_label.csv',delimiter=',')
train_y = np.array(train_y)
print(train_y.shape)
test_x = read_csv(root + '1.original/test/test', testset_num)
test_x = np.array(test_x)
print(test_x.shape)
test_y = np.loadtxt(root + '1.original/test/test_label.csv',delimiter=',')
test_y = np.array(test_y)
print(test_y.shape)


file_name = '../time_data/' + dataset + '/'
x_train = np.load(file_name + 'train_x.npy')

y_train = np.load(file_name + 'train_y.npy')
x_test = np.load(file_name + 'test_x.npy')
y_test = np.load(file_name + 'test_y.npy')
print('%%%%%%% ', x_train.shape)
print('%%%%%%% ', x_test.shape)

dataDTW = np.concatenate((train_x, test_x),axis=0)
label = np.concatenate((train_y,test_y),axis=0)
print(dataDTW.shape)

edge = np.zeros((2,2340))
k = 0
for i in range(dataDTW.shape[0]):
    dist = np.zeros(dataDTW.shape[0])
    b = [i for _ in range(10)]
    for j in range(dataDTW.shape[0]):
        if j == i:
            dist[j] = math.inf
        else:
            # dist[j] = educlidean_edge(dataDTW[i], dataDTW[j])
            # dist[j] = DTWD(dataDTW[i], dataDTW[j])
            dist[j] = DTWI(dataDTW[i], dataDTW[j])
            # dist[j] = SAX(dataDTW[i], dataDTW[j])
    # 找出前10个距离最短的节点
    dist = np.argsort(dist)
    q = 0
    for t in range(10):
        if i < 160 and dist[t] < 160:
            if label[i] == label[dist[t]]:
                edge[0][k] = i
                edge[1][k] = dist[t]
                k += 1
                q +=1
        else:
            edge[0][k] = i
            edge[1][k] = dist[t]
            k += 1
            q +=1
    print(i, '\t', edge[1][k-q:k])

edge1 = edge[:][0:k]
np.savetxt("../time_data/" + dataset + '/'  + 'DTWI.csv', edge1.T, delimiter='\t')
