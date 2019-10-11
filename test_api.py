# coding:utf8

import numpy as np
import codecs
import torch
import torch.utils.data as D
from torch.autograd import Variable
from collections import deque
import pandas as pd

word2id = {}    # word2id的字典
id2relation = {}   # id转关系的字典

max_len = 50    # 字向量的长度

word2id_file = './data/people-relation/word2id.txt'
id2relation_file = './data/people-relation/id2relation.txt'
model_path = "./model/model_epoch99.pkl"
data_path = "./data/people-relation/test.txt"


def init_dic(word2id_file, id2relation_file):
    """ 初始化两个映射字典 """
    print("--- 初始 word2id, relation2id 映射字典 ---")
    # 读取word2id.txt
    with codecs.open(word2id_file, 'r', 'utf-8') as input_data:
        data_lines = input_data.readlines()
        for line in data_lines:
            word2id[line.split()[0]] = int(line.split()[1])

    # 读取id2relation.txt
    with codecs.open(id2relation_file, 'r', 'utf-8') as input_data:
        data_lines = input_data.readlines()
        for line in data_lines:
            id2relation[int(line.split()[0])] = line.split()[1]


def X_padding(words):
    """
        把句子（words）转为 id 形式，不足 max_len 自动补全长度，超出的截断。
        :param words 句子
        :return max_len长度的字向量
    """
    ids = []
    for i in words:
        if i in word2id:
            ids.append(word2id[i])
        else:
            ids.append(word2id["unknown"])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([word2id["blank"]] * (max_len - len(ids)))  # ids和长度为 max_len-len(idx) 的 BLACK的id的数组合并，补全

    return ids


def pos(position):
    """
    将字到实体的距离归到[0, 80]的范围内，距离小于-40的归为0，-40~40之间的 加40，大于40的归为0。
    :param position: 字到实体的距离
    :return: 返回归到[0, 80]距离
    """
    if position < -40:
        return 0
    if (position >= -40) and (position <= 40):
        return position + 40
    if position > 40:
        return 80


def position_padding(words):
    """
    将位置向量归到[0, 81]，长度归到max_len，长度超出截断，不足的结尾以81补全。
    :param words: 位置向量
    :return: 归到 [0, 81] 的位置向量
    """
    words = [pos(w) for w in words]
    if len(words) >= max_len:
        return words[:max_len]
    words.extend([81] * (max_len - len(words)))
    return words


def init_data(data_file, batch_size):
    """
        将待识别关系的文本和实体转成dataloader，每条数据的格式 实体1 实体2 二者的共同语料
        :param 存放数据的文件（包含路径）
        :return 返回数据集data和有效数据长度（末尾不够一个batch_size的数据会补齐）
        """
    data = []
    entity = []    # 存放两个实体
    with codecs.open(data_file, 'r', 'utf-8') as tfc:
        for lines in tfc:
            line = lines.split()
            data.append(line)
            entity.append(line[0] + "  " + line[1])

    data_length = len(data)
    remainder = data_length % batch_size
    t = ["-", "-", "-00000000000000000000000000000000"]     # 不够128个数据时，补充数据以保证输入的正确
    for i in range(batch_size-remainder+1):
        data.append(t)

    sen_data = deque()  # 存放句子的二维数组，其中每个句子是一个数组。
    positionE1 = deque()  # 存放每个句子中，每个字到实体1的距离向量。
    positionE2 = deque()  # 存放每个句子中，每个字到实体2的距离向量。

    for line in data:
        sentence = []
        index1 = line[2].index(line[0])
        position1 = []
        index2 = line[2].index(line[1])
        position2 = []

        for i, word in enumerate(line[2]):
            sentence.append(word)  # 句子
            position1.append(i - index1)  # 字向量，句子中每个字到第一个实体的距离
            position2.append(i - index2)  # 字向量，句子中每一个字到第二个实体的距离

        sen_data.append(sentence)
        positionE1.append(position1)
        positionE2.append(position2)

    df_data = pd.DataFrame({'words': sen_data, 'positionE1': positionE1, 'positionE2': positionE2},
                           index=range(len(sen_data)))
    df_data['words'] = df_data['words'].apply(X_padding)
    df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
    df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

    sen_data = np.asarray(list(df_data['words'].values))
    positionE1 = np.asarray(list(df_data['positionE1'].values))
    positionE2 = np.asarray(list(df_data['positionE2'].values))

    test = torch.LongTensor(sen_data[:len(sen_data) - len(sen_data) % batch_size])
    position1 = torch.LongTensor(positionE1[:len(test) - len(test) % batch_size])
    position2 = torch.LongTensor(positionE2[:len(test) - len(test) % batch_size])
    test_datasets = D.TensorDataset(test, position1, position2)

    test_dataloader = D.DataLoader(test_datasets, batch_size)

    return test_dataloader, entity


if __name__ == "__main__":
    init_dic(word2id_file, id2relation_file)

    # GPU是否可用
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location="cpu")
        model.use_gpu = False

    test_dataloader, entity = init_data(data_path, batch_size=model.batch_size)
    all_relation = []
    for sentence, pos1, pos2 in test_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence, pos1, pos2)
        if use_gpu:
            y = np.argmax(y.data.cpu().numpy(), axis=1)
        else:
            y = np.argmax(y.data.numpy(), axis=1)

        for i, re in enumerate(y):
            all_relation.append(id2relation[re])

    all_relation = all_relation[:len(entity)]
    for i, en in enumerate(entity):
        print(en, "-->", all_relation[i])
