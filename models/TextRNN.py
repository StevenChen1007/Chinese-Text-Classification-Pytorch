# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):
    """
    配置参数类
    包含模型训练所需的所有超参数配置
    """
    def __init__(self, dataset, embedding):
        # ==================== 路径配置（固定参数）====================
        self.model_name = 'TextRNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集路径
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集路径
        self.test_path = dataset + '/data/test.txt'                                  # 测试集路径
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单：10类
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型保存路径
        self.log_path = dataset + '/log/' + self.model_name                          # TensorBoard日志路径
        
        # ==================== 词向量配置 ====================
        # 加载预训练词向量（搜狗新闻或腾讯词向量），或使用随机初始化
        # [可调参数] embedding选择会影响模型效果，预训练通常更好
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None
        
        # 设备配置：自动检测GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ==================== 🎯 关键超参数（需要调优）====================
        
        # [重要-正则化] dropout比例：防止过拟合
        # 建议搜索范围：[0.1, 0.3, 0.5, 0.7]
        # 原理：随机丢弃神经元，增强泛化能力
        self.dropout = 0.7  # 固定dropout*
        
        # [重要-优化器] 学习率：控制参数更新步长
        # 建议搜索范围：[1e-4, 5e-4, 1e-3, 2e-3, 5e-3]（对数尺度）
        # 原理：太大会震荡/发散，太小收敛慢
        self.learning_rate = 2e-3  # 最终配置：lr_final
        
        # [重要-模型容量] LSTM隐藏层维度
        # 建议搜索范围：[64, 128, 256, 512]
        # 原理：控制模型表达能力，太大容易过拟合
        self.hidden_size = 128  # 固定hidden*
        
        # [重要-模型深度] LSTM层数
        # 建议搜索范围：[1, 2, 3]
        # 原理：更深可以学习更复杂的特征，但也更难训练
        self.num_layers = 2  # 固定layers*
        
        # ==================== 次要超参数 ====================
        
        # [次要] 早停patience：验证集性能多少batch没提升就停止
        # 建议范围：[500, 1000, 2000]
        self.require_improvement = 1000
        
        # [次要] 训练轮数：通常配合早停使用
        # 建议范围：[5, 10, 20]
        self.num_epochs = 10
        
        # [次要] batch大小：主要影响训练速度和GPU内存
        # 注意：改变batch_size需要相应调整learning_rate
        # 建议范围：[32, 64, 128, 256]
        # 调参阶段固定为256（已测试最大可用）
        self.batch_size = 256
        
        # [次要] 序列填充长度：根据数据分布设置
        # 数据集文本长度20-30，32是合理的
        self.pad_size = 32
        
        # ==================== 自动计算的参数（不需调整）====================
        self.num_classes = len(self.class_list)                         # 类别数：10
        self.n_vocab = 0                                                # 词表大小，运行时自动赋值
        
        # 词向量维度：使用预训练则跟随预训练维度，否则默认300
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300


'''
Recurrent Neural Network for Text Classification with Multi-Task Learning
论文：https://arxiv.org/abs/1605.05101

模型结构：
    Input -> Embedding -> BiLSTM -> FC -> Output
    
BiLSTM的优势：
    - 双向处理，同时捕获前向和后向上下文信息
    - 适合处理序列数据，能学习长距离依赖
'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        # 词嵌入层：将词索引转换为稠密向量
        # freeze=False 表示训练时会微调词向量
        if config.embedding_pretrained is not None:
            # 使用预训练词向量初始化
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            # 随机初始化词向量
            # padding_idx: 指定填充token的索引，其向量固定为0
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        
        # BiLSTM层：双向长短期记忆网络
        # - input_size: 输入维度（词向量维度）
        # - hidden_size: 隐藏层维度
        # - num_layers: LSTM堆叠层数
        # - bidirectional=True: 双向LSTM，输出维度为 hidden_size * 2
        # - batch_first=True: 输入格式为 [batch, seq_len, feature]
        # - dropout: 层间dropout（仅在num_layers>1时生效）
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        # 全连接分类层
        # 输入维度：hidden_size * 2（双向拼接）
        # 输出维度：类别数
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        # x是一个tuple: (token_ids, seq_len)
        x, _ = x  # 取出token_ids，忽略seq_len
        
        # 词嵌入：[batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        # 例如：[128, 32] -> [128, 32, 300]
        out = self.embedding(x)
        
        # BiLSTM前向传播
        # 输入：[batch_size, seq_len, embed_dim]
        # 输出：[batch_size, seq_len, hidden_size * 2]
        # 第二个返回值是(h_n, c_n)，这里忽略
        out, _ = self.lstm(out)
        
        # 取最后一个时间步的隐藏状态作为句子表示
        # out[:, -1, :] -> [batch_size, hidden_size * 2]
        # 然后通过全连接层分类 -> [batch_size, num_classes]
        out = self.fc(out[:, -1, :])
        
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
