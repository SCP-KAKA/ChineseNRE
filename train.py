# coding:utf8
import numpy as np
import pickle
import os
import codecs
import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT

# 检测显卡是否可用
use_gpu = torch.cuda.is_available()
# use_gpu = False   # 手动禁用GPU

EPOCHS = 100    # 训练的epoch数

config = {}
config['EMBEDDING_SIZE'] = 0  # 初始化为0，在main中修改
config['EMBEDDING_DIM'] = 100
config['POS_SIZE'] = 82   # 不同数据集这里可能会报错。
config['POS_DIM'] = 25
config['HIDDEN_DIM'] = 200
config['TAG_SIZE'] = 0  # 初始化w为0，在main中修改
config['BATCH_SIZE'] = 128      # 一个batch的大小，模型训练好以后续训和使用都不能修改
config["pretrained"] = False    # 是否加载预训练模型
config['USE_GPU'] = use_gpu     # 模型是否需要在GPU上运行

train_data_file = "./data/train.pkl"    # 训练数据文件
test_data_file = "./data/test.pkl"      # 每个epoch的测试数据文件
pretrained_file = "./vec.txt"    # 预训练模型数据
continuation_model = './model/model_epoch99.pkl'    # 原有的已经训练过的模型（区别于预训练模型）
log_path = "./model/log.txt"    # 保存训练过程中模型信息日志


def init_data(train_data_path, test_data_path, batch_size):
    """
    加载训练过程中的训练、测试数据
    :param train_data_path 预处理好的训练数据，.pkl文件
    :param test_data_path  预处理好的测试数据，.pkl文件
    :param batch_size  训练过程中一个batch的大小
    :return 返回word2id（字转向量）字典，relation2id（关系转id）字典，
            以及训练、测试数据的迭代器
    """
    with open(train_data_path, 'rb') as inp:
        word2id = pickle.load(inp)
        relation2id = pickle.load(inp)
        train = pickle.load(inp)
        labels = pickle.load(inp)
        position1 = pickle.load(inp)
        position2 = pickle.load(inp)

    with open(test_data_path, 'rb') as inp:
        test = pickle.load(inp)
        labels_t = pickle.load(inp)
        position1_t = pickle.load(inp)
        position2_t = pickle.load(inp)

    print("train len", len(train))
    print("test len", len(test))
    print("word2id len", len(word2id))

    train = torch.LongTensor(train[:len(train) - len(train) % batch_size])
    position1 = torch.LongTensor(position1[:len(train) - len(train) % batch_size])
    position2 = torch.LongTensor(position2[:len(train) - len(train) % batch_size])
    labels = torch.LongTensor(labels[:len(train) - len(train) % batch_size])
    train_datasets = D.TensorDataset(train, position1, position2, labels)
    train_dataloader = D.DataLoader(train_datasets, batch_size, True, num_workers=2)

    test = torch.LongTensor(test[:len(test) - len(test) % batch_size])
    position1_t = torch.LongTensor(position1_t[:len(test) - len(test) % batch_size])
    position2_t = torch.LongTensor(position2_t[:len(test) - len(test) % batch_size])
    labels_t = torch.LongTensor(labels_t[:len(test) - len(test) % batch_size])
    test_datasets = D.TensorDataset(test, position1_t, position2_t, labels_t)
    test_dataloader = D.DataLoader(test_datasets, batch_size, True, num_workers=2)

    return word2id, relation2id, train_dataloader, test_dataloader


def init_pretrained(pre_file, word2id):
    """
    加载预训练词向量
    :param pre_file: 预训练词向量文件
    :param word2id: word2id字典
    :return: 返回与word2id等长的数组词向量数组
    """
    embedding_pre = []
    print("use pretrained embedding")
    word2vec = {}
    with codecs.open(pre_file, 'r', 'utf-8') as input_data:
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval, line.split()[1:])

    unknow_pre = []
    unknow_pre.extend([1] * 100)
    embedding_pre.append(unknow_pre)  # wordvec id 0
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)

    # embedding_pre = np.asarray(embedding_pre)
    # print(embedding_pre.shape)
    return embedding_pre


if __name__ == "__main__":
    word2id, relation2id, train_dataloader, test_dataloader = init_data(train_data_file,
                                                                        test_data_file,
                                                                        config["BATCH_SIZE"])
    # 完善配置项
    config['EMBEDDING_SIZE'] = len(word2id) + 1
    config['TAG_SIZE'] = len(relation2id)

    embedding_pre = []
    if config['pretrained']:    # 加载预训练模型数据
        embedding_pre = init_pretrained(pretrained_file, word2id)

    # 加载BiLSTM_ATT模型
    if continuation_model == "":
        model = BiLSTM_ATT(config, embedding_pre)
        if use_gpu:
            model = model.cuda()
    else:
        if use_gpu:
            model = torch.load(continuation_model)
        else:
            model = torch.load(continuation_model, map_location="cpu")
            model.use_gpu = False

    learning_rate = 0.0005
    criterion = nn.CrossEntropyLoss(size_average=True)
    if use_gpu:
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(EPOCHS):
        print("epoch:", epoch)
        acc = 0
        total = 0

        batch = 0
        for sentence, pos1, pos2, tag in train_dataloader:
            sentence = Variable(sentence)
            pos1 = Variable(pos1)
            pos2 = Variable(pos2)
            tags = Variable(tag)

            if use_gpu:
                tags = tags.cuda()
            y = model(sentence, pos1, pos2)
            loss = criterion(y, tags)
            if use_gpu:
                loss = loss.cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_gpu:
                y = np.argmax(y.data.cpu().numpy(), axis=1)
            else:
                y = np.argmax(y.data.numpy(), axis=1)
            acc_tmp = 0
            total_tmp = 0
            for y1, y2 in zip(y, tag):
                if y1 == y2:
                    acc += 1
                    acc_tmp += 1
                total += 1
                total_tmp += 1
            print("epoch:", epoch, "  batch:", batch, "  loss", loss.detach().numpy(), "  acc:", float(acc) / total)
            batch += 1

        print("train:", 100 * float(acc) / total, "%")

        """ 测试 """
        count_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # 结果中每个关系类型的数量
        count_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # 标签中每个关系类型的数量
        count_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # 预测正确的结果每个关系类型下的数量
        for sentence, pos1, pos2, tag in test_dataloader:
            sentence = Variable(sentence)
            pos1 = Variable(pos1)
            pos2 = Variable(pos2)
            y = model(sentence, pos1, pos2)
            if use_gpu:
                y = np.argmax(y.data.cpu().numpy(), axis=1)
            else:
                y = np.argmax(y.data.numpy(), axis=1)
            for y1, y2 in zip(y, tag):
                count_predict[y1] += 1
                count_total[y2] += 1
                if y1 == y2:
                    count_right[y1] += 1

        precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(count_predict)):
            if count_predict[i] != 0:
                precision[i] = float(count_right[i]) / count_predict[i]

            if count_total[i] != 0:
                recall[i] = float(count_right[i]) / count_total[i]

        precision = sum(precision) / len(relation2id)
        recall = sum(recall) / len(relation2id)
        print("准确率：", precision)
        print("召回率：", recall)
        f = (2 * precision * recall) / (precision + recall)
        print("f：", f)

        # 删除原有log文件
        if epoch == 0 and os.path.isfile(log_path):
            os.remove(log_path)

        with open(log_path, mode="a", encoding="utf-8") as log:
            log.write("epoch:" + str(epoch) + "   准确率:" + str(precision)
                      + "   召回率:" + str(recall) + "   f值：" + str(f) + "\n")

        if (epoch+1) % 5 == 0:
            model_name = "./model/model_epoch" + str(epoch) + ".pkl"    # 模型的保存路径
            torch.save(model, model_name)
            print(model_name, "has been saved")

