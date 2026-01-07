# coding: UTF-8
"""
===========================================
run.py - 训练主入口文件
===========================================

整体训练流程：
    1. 解析命令行参数（选择模型、词向量）
    2. 加载模型配置（从 models/TextRNN.py 等）
    3. 构建数据集和迭代器
    4. 初始化模型
    5. 开始训练

使用方法：
    python run.py --model TextRNN
    python run.py --model TextRNN --embedding random  # 使用随机词向量
"""

import time
import torch
import numpy as np
from train_eval import train, init_network  # 导入训练函数和权重初始化函数
from importlib import import_module          # 动态导入模块
import argparse

# ==================== 命令行参数解析 ====================
parser = argparse.ArgumentParser(description='Chinese Text Classification')

# --model: 选择模型（必填）
parser.add_argument('--model', type=str, required=True, 
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')

# --embedding: 词向量类型，默认使用预训练
parser.add_argument('--embedding', default='pre_trained', type=str, 
                    help='random or pre_trained')

# --word: 是否使用词级别，默认使用字级别（中文常用字级别）
parser.add_argument('--word', default=False, type=bool, 
                    help='True for word, False for char')

args = parser.parse_args()


if __name__ == '__main__':
    # ==================== 1. 基础配置 ====================
    dataset = 'THUCNews'  # 数据集目录名

    # 词向量选择：
    # - embedding_SougouNews.npz: 搜狗新闻预训练词向量（推荐）
    # - embedding_Tencent.npz: 腾讯预训练词向量
    # - random: 随机初始化
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    
    model_name = args.model  # 获取模型名称
    
    # FastText模型需要特殊的数据处理（包含n-gram特征）
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'  # FastText不使用预训练词向量
    else:
        # 其他模型使用通用的数据处理函数
        from utils import build_dataset, build_iterator, get_time_dif

    # ==================== 2. 动态加载模型配置 ====================
    # import_module('models.TextRNN') 等价于 import models.TextRNN
    # 这样可以根据命令行参数动态选择模型
    x = import_module('models.' + model_name)
    
    # 创建配置对象，包含所有超参数
    # 调参时主要修改这里面的参数！
    config = x.Config(dataset, embedding)
    
    # ==================== 3. 设置随机种子（保证可复现）====================
    SEED = 1  # 最终测试使用seed=1
    np.random.seed(SEED)                          # numpy随机种子
    torch.manual_seed(SEED)                       # CPU随机种子
    torch.cuda.manual_seed_all(SEED)              # 所有GPU随机种子
    torch.backends.cudnn.deterministic = True  # cuDNN使用确定性算法
    # 注意：设置deterministic=True会略微降低训练速度

    # ==================== 4. 加载数据集 ====================
    start_time = time.time()
    print("Loading data...")
    
    # build_dataset 返回：
    # - vocab: 词表字典 {word: index}
    # - train_data: 训练集 [(token_ids, label, seq_len), ...]
    # - dev_data: 验证集
    # - test_data: 测试集
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    
    # 构建数据迭代器（按batch_size分批）
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # ==================== 5. 初始化模型并训练 ====================
    # 设置词表大小（在build_dataset之后才知道）
    config.n_vocab = len(vocab)
    
    # 创建模型并移动到GPU/CPU
    model = x.Model(config).to(config.device)
    
    # 权重初始化（Transformer有自己的初始化方式）
    # 使用Xavier初始化，有助于训练稳定
    if model_name != 'Transformer':
        init_network(model)
    
    # 打印模型参数信息
    print(model.parameters)
    
    # 开始训练！
    # train函数会：
    # 1. 训练模型
    # 2. 在验证集上评估
    # 3. 保存最佳模型
    # 4. （可选）最后在测试集上评估
    # 
    # 调参阶段：do_test=False，避免使用测试集做决策
    # 最终确认：do_test=True，在测试集上评估一次
    DO_TEST = False  # 调参时设为False，最终确认时改为True
    train(config, model, train_iter, dev_iter, test_iter, do_test=DO_TEST)
