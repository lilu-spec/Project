import math
import time
import numpy as np
def distanceED(signal_1, signal_2):
    # 这里的距离度量标准如下：（应该是1NN-ED）
    # 转化
    signal_1 = np.array(signal_1).astype(float)
    signal_2 = np.array(signal_2).astype(float)
    distance = 0
    min_len = min(signal_1.shape[0], signal_2.shape[0])
    max_len = max(signal_1.shape[0], signal_2.shape[0])
    for i in range(0, min_len):
        # 差的平方和
        distance += np.sqrt(np.sum(np.square(signal_1[i] - signal_2[i])))
    if signal_1.shape[0] > signal_2.shape[0]:
        for j in range(min_len,max_len):
            distance += np.sqrt(np.sum(np.square(signal_1[j])))
    elif signal_1.shape[0]<signal_2.shape[0]:
        for j in range(min_len,max_len):
            distance += np.sqrt(np.sum(np.square(signal_2[j])))
    return distance
def distanceDTW(signal_1, signal_2):
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


#  BasicMotionsHandMovementDirectionJapaneseVowels  StandWalkJump SelfRegulationSCP2
#  StandWalkJumpAtrialFibrillation  PhonemeSpectraAtrialFibrillationSpokenArabicDigits
#  #ArticularyWordRecognition CharacterTrajectoriesNATOPS StandWalkJump

dataset_name = 'StandWalkJump'

start = time.time()
file_name = '../time_data/' + dataset_name + '/'
x_train = np.load(file_name+'train_x.npy')
y_train = np.load(file_name + 'train_y.npy')
x_test = np.load(file_name+'test_x.npy')
y_test = np.load(file_name + 'test_y.npy')

G = []
for i in range(x_train.shape[0]):  # 遍历每一个训练样本
    for j in range(i,x_train.shape[0]):
        if y_train[j] == y_train[i]:
            temp = np.zeros([2, 2])
            temp[0][0] = j
            temp[0][1] = i
            temp[1][0] = i
            temp[1][1] = j
            # 无向边
            if j == i:
                G.append(temp[0])
            else:
                G.append(temp[0])
                G.append(temp[1])

for a in range(x_test.shape[0]):
    dist = np.zeros(x_train.shape[0])
    for b in range(x_train.shape[0]):
        dist[b] = distanceDTW(x_test[a], x_train[b])
    dist = np.argsort(dist)
    for near in range(5):
        temp = np.zeros([2, 2])
        temp[0][0] = a + x_train.shape[0]
        temp[0][1] = dist[near]
        temp[1][0] = dist[near]
        temp[1][1] = a + x_train.shape[0]
        G.append(temp[0])
        G.append(temp[1])
    cur = np.zeros([2])  # 作用是+自环
    cur[0] = a + x_train.shape[0]
    cur[1] = a + x_train.shape[0]
    G.append(cur)
    print('The '+str(a+1)+'-th test sample!')

G = np.array(G).astype(int)
np.savetxt('../graph/'+dataset_name+'_graph.csv', G, fmt='%i', delimiter='\t')

