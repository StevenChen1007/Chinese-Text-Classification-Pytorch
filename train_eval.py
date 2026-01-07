# coding: UTF-8
"""
===========================================
train_eval.py - 训练和评估模块
===========================================

主要功能：
    1. init_network: 模型权重初始化
    2. train: 训练主循环（包含早停、模型保存、TensorBoard日志）
    3. test: 在测试集上评估最佳模型
    4. evaluate: 评估函数（计算loss和准确率）

训练流程：
    for epoch in epochs:
        for batch in train_data:
            1. 前向传播
            2. 计算loss
            3. 反向传播
            4. 更新参数
            5. 每100个batch在验证集评估
            6. 如果验证集loss改善，保存模型
            7. 如果连续1000个batch无改善，早停
    
    训练结束后在测试集上评估最终效果
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter  # 用于TensorBoard可视化


def init_network(model, method='xavier', exclude='embedding', seed=123):
    """
    模型权重初始化
    
    为什么需要初始化？
        - 好的初始化可以加速收敛，避免梯度消失/爆炸
        - Xavier初始化适合tanh/sigmoid激活
        - Kaiming初始化适合ReLU激活
    
    Args:
        model: PyTorch模型
        method: 初始化方法 ('xavier', 'kaiming', 或其他)
        exclude: 排除的层名称（embedding层使用预训练权重，不重新初始化）
        seed: 随机种子
    """
    for name, w in model.named_parameters():
        # 跳过embedding层（保留预训练词向量）
        if exclude not in name:
            if 'weight' in name:
                # 权重初始化
                if method == 'xavier':
                    nn.init.xavier_normal_(w)  # Xavier正态分布初始化
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)  # Kaiming正态分布初始化（适合ReLU）
                else:
                    nn.init.normal_(w)  # 标准正态分布初始化
            elif 'bias' in name:
                # 偏置初始化为0
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, do_test=False):
    """
    训练主函数
    
    Args:
        config: 配置对象（包含所有超参数）
        model: 模型
        train_iter: 训练数据迭代器
        dev_iter: 验证数据迭代器
        test_iter: 测试数据迭代器
        do_test: 是否在训练结束后运行测试集评估（调参期间设为False避免泄漏）
    
    关键超参数（在config中设置）：
        - learning_rate: 学习率 ⭐重要
        - num_epochs: 训练轮数
        - require_improvement: 早停patience（多少batch无改善就停止）
    """
    start_time = time.time()
    
    # 设置为训练模式（启用dropout等）
    model.train()
    
    # ==================== 优化器配置 ====================
    # 使用Adam优化器
    # 🎯 learning_rate 是最重要的超参数！
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率衰减（可选，默认注释掉）
    # 如果启用，每个epoch学习率乘以gamma
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # ==================== 训练状态变量 ====================
    total_batch = 0           # 已训练的batch总数
    dev_best_loss = float('inf')  # 验证集最佳loss
    last_improve = 0          # 上次验证集改善的batch数
    flag = False              # 早停标志
    
    # TensorBoard日志（可用tensorboard --logdir=THUCNews/log 查看）
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    
    # ==================== 训练主循环 ====================
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step()  # 如果使用学习率衰减，取消注释
        
        for i, (trains, labels) in enumerate(train_iter):
            # trains: ((token_ids, seq_len), ) 格式的输入
            # labels: 标签
            
            # ----- 1. 前向传播 -----
            outputs = model(trains)  # [batch_size, num_classes]
            
            # ----- 2. 计算损失 -----
            model.zero_grad()  # 清空梯度
            loss = F.cross_entropy(outputs, labels)  # 交叉熵损失
            
            # ----- 3. 反向传播 -----
            loss.backward()  # 计算梯度
            
            # ----- 4. 参数更新 -----
            optimizer.step()  # 更新权重
            
            # ----- 5. 定期评估（每100个batch）-----
            if total_batch % 100 == 0:
                # 计算当前batch的训练准确率
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()  # 取概率最大的类别
                train_acc = metrics.accuracy_score(true, predic)
                
                # 在验证集上评估
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                
                # 如果验证集loss改善，保存模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)  # 保存最佳模型
                    improve = '*'  # 标记改善
                    last_improve = total_batch
                else:
                    improve = ''
                
                # 打印训练信息
                # 格式：Iter, Train Loss, Train Acc, Val Loss, Val Acc, Time, 是否改善
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                
                # 写入TensorBoard日志
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                
                # 恢复训练模式（evaluate会切换到eval模式）
                model.train()
            
            total_batch += 1
            
            # ----- 6. 早停检查 -----
            # 如果连续 require_improvement 个batch验证集loss没有改善，则停止训练
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        
        if flag:
            break
    
    writer.close()
    
    # ==================== 训练结束，测试最终效果 ====================
    # 调参阶段关闭测试，避免用测试集做决策
    if do_test:
        test(config, model, test_iter)
    else:
        print("=== Tuning mode: skipping test evaluation ===")
        print(f"Best dev_loss: {dev_best_loss:.4f}")


def test(config, model, test_iter):
    """
    在测试集上评估模型
    
    会加载训练过程中保存的最佳模型（验证集loss最低的那个）
    """
    # 加载最佳模型
    model.load_state_dict(torch.load(config.save_path))
    model.eval()  # 设置为评估模式（关闭dropout）
    
    start_time = time.time()
    
    # 在测试集上评估，返回详细报告
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    
    # 打印结果
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    
    # 打印分类报告（每个类别的精确率、召回率、F1）
    print("Precision, Recall and F1-Score...")
    print(test_report)
    
    # 打印混淆矩阵
    print("Confusion Matrix...")
    print(test_confusion)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    """
    评估函数：计算损失和准确率
    
    Args:
        config: 配置对象
        model: 模型
        data_iter: 数据迭代器
        test: 是否是测试模式（True则返回详细报告）
    
    Returns:
        test=False: (准确率, 平均loss)
        test=True: (准确率, 平均loss, 分类报告, 混淆矩阵)
    """
    model.eval()  # 评估模式：关闭dropout等
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    # 不计算梯度（节省内存，加速计算）
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            
            # 收集所有预测和标签
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)
    
    if test:
        # 测试模式：返回详细报告
        # classification_report: 每个类别的precision, recall, f1-score
        report = metrics.classification_report(labels_all, predict_all, 
                                               target_names=config.class_list, digits=4)
        # confusion_matrix: 混淆矩阵
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    
    # 非测试模式：只返回准确率和loss
    return acc, loss_total / len(data_iter)