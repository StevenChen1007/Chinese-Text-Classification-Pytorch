# coding: UTF-8
"""
===========================================
utils.py - 数据处理工具
===========================================

主要功能：
    1. build_vocab: 构建词表（字/词 -> 索引）
    2. build_dataset: 加载并预处理数据集
    3. DatasetIterater: 数据迭代器（按batch返回数据）
    4. build_iterator: 构建迭代器的便捷函数

数据格式：
    训练数据格式：每行 "文本内容\t标签"
    例如："体育新闻标题\t7"
"""

import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


# ==================== 全局常量 ====================
MAX_VOCAB_SIZE = 10000  # 词表最大长度，超出的低频词会被丢弃
UNK, PAD = '<UNK>', '<PAD>'  # 特殊token：未知词、填充符


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    构建词表（从训练集中统计词频）
    
    Args:
        file_path: 训练集路径
        tokenizer: 分词函数（字级别或词级别）
        max_size: 词表最大大小
        min_freq: 最小词频阈值（低于此频率的词会被过滤）
    
    Returns:
        vocab_dic: 词表字典 {word: index}
    """
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            # 数据格式："文本内容\t标签"，只取文本部分
            content = lin.split('\t')[0]
            # 统计词频
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        
        # 按词频降序排序，取前max_size个，且词频>=min_freq
        vocab_list = sorted(
            [_ for _ in vocab_dic.items() if _[1] >= min_freq], 
            key=lambda x: x[1], 
            reverse=True
        )[:max_size]
        
        # 转换为 {word: index} 格式
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        
        # 添加特殊token：UNK（未知词）和 PAD（填充）
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    
    return vocab_dic


def build_dataset(config, ues_word):
    """
    构建数据集
    
    Args:
        config: 配置对象（包含路径、pad_size等参数）
        ues_word: 是否使用词级别（False则使用字级别）
    
    Returns:
        vocab: 词表字典
        train: 训练集 [(token_ids, label, seq_len), ...]
        dev: 验证集
        test: 测试集
    """
    # ==================== 1. 选择分词方式 ====================
    if ues_word:
        # 词级别：按空格分割（需要预先分好词）
        tokenizer = lambda x: x.split(' ')
    else:
        # 字级别：每个字符作为一个token（中文常用）
        tokenizer = lambda x: [y for y in x]
    
    # ==================== 2. 加载或构建词表 ====================
    if os.path.exists(config.vocab_path):
        # 词表已存在，直接加载
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        # 从训练集构建词表并保存
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, 
                           max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    
    print(f"Vocab size: {len(vocab)}")

    # ==================== 3. 加载数据集 ====================
    def load_dataset(path, pad_size=32):
        """
        加载单个数据集文件
        
        处理流程：
            1. 读取每行文本和标签
            2. 分词（字/词级别）
            3. 填充或截断到固定长度(pad_size)
            4. 将token转换为索引
        
        Args:
            path: 数据文件路径
            pad_size: 填充/截断的目标长度
        
        Returns:
            contents: [(token_ids, label, seq_len), ...]
        """
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                
                # 解析：文本内容 和 标签
                content, label = lin.split('\t')
                words_line = []
                
                # 分词
                token = tokenizer(content)
                seq_len = len(token)  # 原始序列长度
                
                # 填充或截断到pad_size
                if pad_size:
                    if len(token) < pad_size:
                        # 短文本：用PAD填充到pad_size
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        # 长文本：截断到pad_size
                        token = token[:pad_size]
                        seq_len = pad_size
                
                # 将token转换为索引（词表中查找，找不到则用UNK）
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                
                # 保存：(token索引列表, 标签, 原始长度)
                contents.append((words_line, int(label), seq_len))
        
        return contents
    
    # 加载训练集、验证集、测试集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    
    return vocab, train, dev, test


class DatasetIterater(object):
    """
    数据集迭代器
    
    功能：将数据按batch_size分批，每次迭代返回一个batch的数据
    
    使用方式：
        for (x, seq_len), y in data_iter:
            # x: token索引 [batch_size, seq_len]
            # seq_len: 原始长度 [batch_size]
            # y: 标签 [batch_size]
    """
    
    def __init__(self, batches, batch_size, device):
        """
        Args:
            batches: 数据列表 [(token_ids, label, seq_len), ...]
            batch_size: 每批数据大小（可调参数！影响训练速度和GPU内存）
            device: 运行设备（cuda/cpu）
        """
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size  # 完整batch数量
        self.residue = False  # 是否有不完整的最后一个batch
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 当前batch索引
        self.device = device

    def _to_tensor(self, datas):
        """
        将一个batch的数据转换为PyTorch张量
        
        Returns:
            (x, seq_len): 输入数据
                - x: token索引 [batch_size, seq_len]
                - seq_len: 原始序列长度 [batch_size]
            y: 标签 [batch_size]
        """
        # 提取token索引
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # 提取标签
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # 提取原始序列长度（用于变长RNN，但当前模型未使用）
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        
        return (x, seq_len), y

    def __next__(self):
        """迭代器：返回下一个batch"""
        if self.residue and self.index == self.n_batches:
            # 处理最后一个不完整的batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            # 所有batch已遍历完，重置并停止
            self.index = 0
            raise StopIteration
        else:
            # 返回一个完整的batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        """使对象可迭代"""
        return self

    def __len__(self):
        """返回总batch数量"""
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    构建数据迭代器的便捷函数
    
    Args:
        dataset: 数据集（来自build_dataset）
        config: 配置对象（需要batch_size和device）
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """
    计算已用时间
    
    Args:
        start_time: 开始时间（time.time()的返回值）
    
    Returns:
        格式化的时间差字符串，如 "0:05:23"
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# ==================== 预训练词向量提取工具 ====================
# 直接运行 python utils.py 可以提取预训练词向量
# 注意：通常不需要运行，数据集已包含提取好的词向量

if __name__ == "__main__":
    """
    提取预训练词向量
    
    功能：从搜狗新闻预训练词向量文件中，提取本数据集词表对应的词向量
    
    流程：
        1. 加载或构建词表
        2. 初始化词向量矩阵（随机）
        3. 从预训练文件中查找词表中的词，替换对应的词向量
        4. 保存为.npz文件
    
    使用方法：
        1. 下载搜狗新闻词向量: sgns.sogou.char
        2. 放到 THUCNews/data/ 目录下
        3. 运行: python utils.py
    """
    # 路径配置（按需修改）
    train_dir = "./THUCNews/data/train.txt"           # 训练集路径
    vocab_dir = "./THUCNews/data/vocab.pkl"           # 词表保存路径
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"  # 预训练词向量文件
    emb_dim = 300                                      # 词向量维度
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"  # 输出文件名
    
    # 1. 加载或构建词表
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # 字级别分词
        tokenizer = lambda x: [y for y in x]
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    # 2. 初始化词向量矩阵（随机值）
    # 词表中找不到预训练向量的词会保留随机值
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    
    # 3. 从预训练文件中提取词向量
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # 每行格式："词 dim1 dim2 dim3 ... dim300"
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            # 如果词在词表中，提取其向量
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]  # 取300维
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    
    # 4. 保存为压缩的.npz文件
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    print(f"词向量保存至: {filename_trimmed_dir}.npz")
