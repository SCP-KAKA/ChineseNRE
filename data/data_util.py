# coding:utf8
import codecs
import pandas as pd
import numpy as np
from collections import deque  
import collections
import pickle

word2id = {"unknown": 0, "blank": 1}    # word2id的字典
relation2id = {}    # 从文件中读取关系和对应的id，转成关系向量
max_len = 50    # 字向量的长度


def init_dic(word2id_file, relation2id_file):
    """ 初始化两个映射字典 """
    print("--- 初始 word2id, relation2id 映射字典 --")
    # 读取word2id.txt
    with codecs.open(word2id_file, 'r', 'utf-8') as input_data:
        data_lines = input_data.readlines()
        for line in data_lines:
            word2id[line.split()[0]] = int(line.split()[1])

    # 读取relation2id.txt
    with codecs.open(relation2id_file, 'r', 'utf-8') as input_data:
        data_lines = input_data.readlines()
        for line in data_lines:
            relation2id[line.split()[0]] = int(line.split()[1])


def flatten(x):
    """
    将嵌套列表（多维数组）拉平
    :param x:
    :return:
    """
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def X_padding(words):
    """
        把句子（words）转为 id 形式，不足 max_len 自动补全长度，超出的截断。
        :param 句子
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


def update_word2id(path, train_data, test_data):
    """
    从所有数据中，补充字典中没有的字符，添加其id
    更新word2id文件，每次生成数据的时候都要先更新一遍
    """

    # 获取所有的语料（句子）
    sentence_data = []
    for item in train_data:
        sentence_data.append([w for w in item[3]])
    for item in test_data:
        sentence_data.append([w for w in item[3]])

    all_words = flatten(sentence_data)  # 将数组拉平，存放datas中所有句子中的字符
    sr_allwords = pd.Series(all_words)  # 给每个字符加上索引
    sr_allwords = sr_allwords.value_counts()  # 统计每个字符的数量
    set_words = sr_allwords.index  # 所有的字符
    # 向word2id中补充没有的字符，并按顺序添加id
    next_index = len(word2id)
    for word in set_words:
        if (word not in word2id) and (word != ""):
            word2id[word] = next_index
            next_index += 1

    with open(file=path, mode="w", encoding="utf-8") as f:
        for key in word2id:
            f.write(key + " " + str(word2id[key]) + "\n")

    print("word2id字典更新完成, 数量-->", len(word2id))


def data2pkl(file_name, data, is_train=False):
    """ 将train_data, test_data转成pkl """
    datas = deque()  # 存放句子的二维数组，其中每个句子是一个数组。
    labels = deque()  # 存放每个句子中两个实体的关系标签对应的id
    positionE1 = deque()  # 存放每个句子中，每个字到实体1的距离向量。
    positionE2 = deque()  # 存放每个句子中，每个字到实体2的距离向量。

    for line in data:
        sentence = []
        index1 = line[3].index(line[0])
        position1 = []
        index2 = line[3].index(line[1])
        position2 = []

        for i, word in enumerate(line[3]):
            sentence.append(word)  # 句子
            position1.append(i - index1)  # 字向量，句子中每个字到第一个实体的距离
            position2.append(i - index2)  # 字向量，句子中每一个字到第二个实体的距离

        datas.append(sentence)
        labels.append(relation2id[line[2]])
        positionE1.append(position1)
        positionE2.append(position2)

    df_data = pd.DataFrame({'words': datas, 'tags': labels, 'positionE1': positionE1, 'positionE2': positionE2},
                           index=range(len(datas)))
    df_data['words'] = df_data['words'].apply(X_padding)
    df_data['tags'] = df_data['tags']
    df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
    df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

    datas = np.asarray(list(df_data['words'].values))
    labels = np.asarray(list(df_data['tags'].values))
    positionE1 = np.asarray(list(df_data['positionE1'].values))
    positionE2 = np.asarray(list(df_data['positionE2'].values))

    # print(datas.shape)
    # print(labels.shape)
    # print(positionE1.shape)
    # print(positionE2.shape)

    with open(file_name, 'wb') as outp:
        if is_train:
            pickle.dump(word2id, outp)
            pickle.dump(relation2id, outp)
        pickle.dump(datas, outp)
        pickle.dump(labels, outp)
        pickle.dump(positionE1, outp)
        pickle.dump(positionE2, outp)
    print('--- 数据集' + file_name + "保存完成")


if __name__ == "__main__":
    # 加载字典
    word2id_file = './people-relation/word2id.txt'
    relation2id_file = './people-relation/relation2id.txt'
    init_dic(word2id_file, relation2id_file)

    relation_data_count = []  # 统计每种关系下的数据量
    for re in relation2id:
        relation_data_count.append(0)

    print("数据加载中...")

    train_data = []    # 训练数据
    test_data = []     # 测试数据

    # 将所有数据每种关系数据按9:1进行训练、测试集划分
    with codecs.open('./people-relation/all_data.txt', 'r', 'utf-8') as tfc:
        for lines in tfc:
            line = lines.split()
            relation_id = relation2id[line[2]]
            relation_data_count[relation_id] += 1
            if relation_data_count[relation_id] % 10 == 0:
                test_data.append(line)
            else:
                train_data.append(line)

    print("--- 训练测试集划分完成 ---")
    print("数据总量-->", relation_data_count)
    print("训练集-->", len(train_data), "测试集-->", len(test_data))

    update_word2id("./people-relation/word2id.txt", train_data, test_data)

    data2pkl("./train.pkl", train_data, is_train=True)
    data2pkl("./test.pkl", test_data)
