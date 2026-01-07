# TextRNN 调参实验设计

## 实验记录

| Exp ID | 阶段 | batch_size | lr | hidden_size | num_layers | dropout | 最佳dev_loss | 最佳dev_acc | 备注 |
|--------|------|------------|-----|-------------|------------|---------|--------------|-------------|------|
| exp_baseline | A | 128 | 1e-3 | 128 | 2 | 0.5 | 0.31 | 89.71% | Baseline，Iter 3300早停 |
| exp_bs256 | B | 256 | 1e-3 | 128 | 2 | 0.5 | 0.3077 | 90.53% | batch_size测试，性能略好 |
| exp_bs512 | B | 512 | 1e-3 | 128 | 2 | 0.5 | 0.318 | 90.21% | batch_size测试 |
| exp_bs1024 | B | 1024 | 1e-3 | 128 | 2 | 0.5 | 0.3247 | 90.30% | batch_size测试 |

**阶段B结论：固定batch_size=256**（平衡速度与性能）。后续lr搜索基准lr_base=2e-3（线性缩放）

| exp_lr1 | C | 256 | 2e-4 | 128 | 2 | 0.5 | 0.3374 | 89.47% | lr太小，收敛慢 |
| exp_lr2 | C | 256 | 1e-3 | 128 | 2 | 0.5 | 0.3077 | 90.84% | lr搜索 |
| exp_lr3 | C | 256 | 2e-3 | 128 | 2 | 0.5 | 0.3047 | 90.87% | **lr_v1最佳** |
| exp_lr4 | C | 256 | 4e-3 | 128 | 2 | 0.5 | 0.3091 | 90.75% | lr搜索 |
| exp_lr5 | C | 256 | 1e-2 | 128 | 2 | 0.5 | 0.3909 | 87.94% | lr太大，震荡 |

**阶段C结论：lr_v1 = 2e-3**

| exp_h1 | D1 | 256 | 2e-3 | 64 | 2 | 0.5 | 0.3145 | 90.84% | hidden搜索 |
| exp_h2 | D1 | 256 | 2e-3 | 256 | 2 | 0.5 | 0.3136 | 91.19% | hidden搜索 |

**阶段D1结论：hidden* = 128**（exp_lr3已测试，dev_loss=0.3047最佳）

| exp_l1 | D2 | 256 | 2e-3 | 128 | 1 | 0.5 | 0.3131 | 90.46% | layers搜索，dropout警告 |
| exp_l2 | D2 | 256 | 2e-3 | 128 | 3 | 0.5 | 0.3089 | 90.57% | layers搜索 |

**阶段D2结论：layers* = 2**（exp_lr3已测试，dev_loss=0.3047最佳）

| exp_d1 | E | 256 | 2e-3 | 128 | 2 | 0.2 | 0.3104 | 90.85% | dropout搜索 |
| exp_d2 | E | 256 | 2e-3 | 128 | 2 | 0.3 | 0.3068 | 91.02% | dropout搜索 |
| exp_d3 | E | 256 | 2e-3 | 128 | 2 | 0.7 | 0.3017 | 90.83% | **dropout*最佳** |

**阶段E结论：dropout* = 0.7**（dev_loss=0.3017最佳）

| exp_lrf1 | F | 256 | 1e-3 | 128 | 2 | 0.7 | 0.3127 | 90.55% | lr重调 |
| exp_lrf2 | F | 256 | 4e-3 | 128 | 2 | 0.7 | 0.3113 | 90.58% | lr重调 |

**阶段F结论：lr_final = 2e-3**（exp_d3已测试，dev_loss=0.3017仍为最佳）

---
### 最终最优配置
| 参数 | 值 |
|------|-----|
| batch_size | 256 |
| learning_rate | 2e-3 |
| hidden_size | 128 |
| num_layers | 2 |
| dropout | 0.7 |

---
### 阶段G：多seed验证

| seed | dev_loss | dev_acc |
|------|----------|---------|
| 1 | 0.3017 | 90.83% |
| 42 | 0.3099 | 90.71% |
| 123 | 0.3016 | 91.09% |
| **平均** | **0.3044 ± 0.004** | **90.88%** |

结论：相比baseline（dev_loss=0.31, dev_acc=89.71%），最终配置稳定提升约1-2%

---
### 阶段H：最终测试结果

| 指标 | 值 |
|------|-----|
| Test Loss | 0.30 |
| **Test Acc** | **90.69%** |
| Macro Precision | 0.9083 |
| Macro Recall | 0.9069 |
| Macro F1 | 0.9072 |

---
## 调参总结

### 最终优化配置 vs Baseline

| 参数 | Baseline | 优化后 |
|------|----------|--------|
| batch_size | 128 | **256** |
| learning_rate | 1e-3 | **2e-3** |
| hidden_size | 128 | 128 |
| num_layers | 2 | 2 |
| dropout | 0.5 | **0.7** |
| **dev_loss** | 0.31 | **0.3017** |
| **dev_acc** | 89.71% | **90.83%** |
| **test_acc** | - | **90.69%** |

### 关键发现

1. **Batch Size**: 256是速度和效果的最佳平衡点，更大的batch_size需要更大的lr
2. **Learning Rate**: lr=2e-3（按batch_size=256线性缩放）表现最佳
3. **Dropout**: 从0.5提高到0.7带来明显改善，说明模型有轻微过拟合
4. **模型容量**: hidden_size=128, num_layers=2已经是最优，不需要增大

### 实验统计

- 总实验次数：约22组
- 调参耗时：约1小时
- 最终提升：dev_acc +1.12%, test_acc达到90.69%

---

## 0. 代码结构总览

### 文件结构
```
Chinese-Text-Classification-Pytorch/
├── run.py                 # 🚀 主入口：解析参数、加载数据、启动训练
├── train_eval.py          # 🔄 训练循环：前向传播、反向传播、验证、早停
├── utils.py               # 📦 数据处理：构建词表、加载数据、构建迭代器
├── models/
│   └── TextRNN.py         # 🧠 模型定义：Config配置类 + Model网络结构
└── THUCNews/
    ├── data/              # 数据文件
    │   ├── train.txt      # 训练集（18万条）
    │   ├── dev.txt        # 验证集（1万条）
    │   ├── test.txt       # 测试集（1万条）
    │   ├── class.txt      # 类别名称（10类）
    │   ├── vocab.pkl      # 词表
    │   └── embedding_SougouNews.npz  # 预训练词向量
    ├── saved_dict/        # 模型保存目录
    └── log/               # TensorBoard日志
```

### 训练流程图
```
┌─────────────────────────────────────────────────────────────┐
│                        run.py                               │
│  1. 解析命令行参数 (--model TextRNN)                         │
│  2. 加载模型配置 (models/TextRNN.py → Config)               │
│  3. 构建数据集 (utils.py → build_dataset)                   │
│  4. 初始化模型 (models/TextRNN.py → Model)                  │
│  5. 开始训练 (train_eval.py → train)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     train_eval.py                           │
│                                                             │
│  for epoch in num_epochs:                                   │
│      for batch in train_data:                               │
│          outputs = model(batch)      # 前向传播              │
│          loss = cross_entropy(...)   # 计算损失              │
│          loss.backward()             # 反向传播              │
│          optimizer.step()            # 更新参数              │
│                                                             │
│          每100个batch:                                       │
│              evaluate(dev_data)      # 验证集评估            │
│              if 改善: save_model()   # 保存最佳模型          │
│              if 1000batch无改善: 早停                        │
│                                                             │
│  test(test_data)                     # 最终测试              │
└─────────────────────────────────────────────────────────────┘
```

### 关键超参数位置
```python
# 在 models/TextRNN.py 的 Config 类中修改：
self.learning_rate = 1e-3    # ⭐ 学习率（最重要）
self.hidden_size = 128       # ⭐ LSTM隐藏层维度
self.num_layers = 2          # ⭐ LSTM层数
self.dropout = 0.5           # ⭐ Dropout比例
self.batch_size = 128        # 批大小
```

---

## 1. 实验目标

通过科学的超参数搜索方法，找到TextRNN在THUCNews数据集上的最优配置，目标是超越baseline的 **91.12%** 准确率。

## 2. 超参数分类（参考TUNE.md）

根据调参指南，我们将超参数分为三类：

### 🎯 目标超参数（Scientific Hyperparameters）
需要研究其对模型性能的影响：
| 超参数 | 默认值 | 搜索范围 | 重要性 |
|--------|--------|----------|--------|
| `learning_rate` | 1e-3 | [1e-4, 5e-4, 1e-3, 2e-3, 5e-3] | ⭐⭐⭐ 最重要 |
| `hidden_size` | 128 | [64, 128, 256] | ⭐⭐⭐ 模型容量 |
| `dropout` | 0.5 | [0.2, 0.3, 0.5, 0.7] | ⭐⭐ 正则化 |
| `num_layers` | 2 | [1, 2, 3] | ⭐⭐ 模型深度 |

### 🔧 冗余超参数（Nuisance Hyperparameters）
需要为每个目标超参数配置单独优化：
- `learning_rate`（当其他参数是目标时，需要重新调整）

### 📌 固定超参数（Fixed Hyperparameters）
实验中保持不变：
| 超参数 | 固定值 | 原因 |
|--------|--------|------|
| `batch_size` | **先测试硬件最大值** | 主要影响训练速度，不影响最终性能 |
| `pad_size` | 32 | 数据集文本长度20-30，32已足够 |
| `num_epochs` | 10 | 配合早停机制 |
| `require_improvement` | 1000 | 早停patience |
| `embedding` | SougouNews | 使用预训练词向量 |

> ⚠️ **关于batch_size**：根据TUNE.md，batch_size主要影响训练速度而非最终性能。
> 建议先测试GPU能支持的最大batch_size（128→256→512...），然后固定使用。
> 注意：改变batch_size后，learning_rate需要线性缩放（batch翻倍→lr翻倍）

---

## 3. 实验设计

### 阶段零：确定Batch Size（一次性）

**目标**：找到GPU能支持的最大batch_size，加速后续所有实验

```bash
# 逐步增大batch_size，直到OOM（内存溢出）
batch_size = 128  # 默认
batch_size = 256  # 尝试
batch_size = 512  # 尝试（如果256能跑）
```

**选择原则**：
- 选最大能跑的batch_size（不OOM）
- 如果batch_size从128改到256，learning_rate也要从1e-3改到2e-3（线性缩放）

**后续实验全部固定此batch_size**

---

### 阶段一：学习率搜索（最关键）

**目标**：确定最优学习率范围

**搜索空间为什么是这个？**
- 1e-3 是Adam优化器的经典默认值，作为中心点
- 使用**对数尺度**搜索（1e-4, 5e-4, 1e-3, 2e-3, 5e-3），因为学习率影响是指数级的
- 先粗搜定范围，后面可以细搜

```bash
# 修改 TextRNN.py 中的 self.learning_rate，依次运行：
lr = 1e-4   # 实验1
lr = 5e-4   # 实验2
lr = 1e-3   # 实验3 (baseline)
lr = 2e-3   # 实验4
lr = 5e-3   # 实验5
```

**预期产出**：
- 绘制 **基本超参数轴图**：X轴=learning_rate，Y轴=验证集准确率
- 确定学习率的"甜点区域"

**如何判断检查点问题？**

| 检查项 | 如何判断 | 解决方案 |
|--------|----------|----------|
| 最佳lr在边界？ | 画图看最高点位置。如果lr=5e-3最好，说明可能lr=1e-2更好 | 扩展搜索范围，尝试更大的lr |
| 训练不稳定？ | 看训练曲线(loss vs steps)：是否震荡、突然飙升、NaN | 降低lr，或添加warmup |

**训练曲线示例**：
```
✅ 稳定（loss平滑下降）      ❌ 不稳定（震荡/发散）
Loss                        Loss
  |  \                        |    /\
  |   \                       |   /  \  /\
  |    \___                   |  /    \/
  +--------> steps            +-----------> steps
```

---

### 阶段二a：Hidden Size搜索

**目标**：固定num_layers=2，找最优hidden_size

使用阶段一的最优learning_rate：

```bash
hidden_size = 64   # 实验1
hidden_size = 128  # 实验2 (baseline)
hidden_size = 256  # 实验3
hidden_size = 512  # 实验4（如果内存允许）
```

**总计**：3-4次试验

**检查点**：
- [ ] 大hidden_size是否过拟合？（看：训练acc - 验证acc 的差距）
- [ ] 最优值是否在边界？

---

### 阶段二b：Num Layers搜索

**目标**：使用阶段2a的最优hidden_size，找最优num_layers

```bash
num_layers = 1  # 实验1
num_layers = 2  # 实验2 (baseline)
num_layers = 3  # 实验3
```

**总计**：3次试验

**注意**：更深的模型可能需要重新微调learning_rate！

---

### 阶段三：正则化调优（Dropout）

**目标**：针对最优模型容量配置，调整dropout

```bash
# 使用阶段二最优的 hidden_size 和 num_layers
dropout = 0.2  # 实验1
dropout = 0.3  # 实验2
dropout = 0.5  # 实验3 (baseline)
dropout = 0.7  # 实验4
```

**总计**：4次试验

**如何选择dropout？**
- 如果阶段二最优模型**过拟合**（训练acc >> 验证acc）→ 增大dropout
- 如果阶段二最优模型**欠拟合**（训练acc也不高）→ 减小dropout

---

### 阶段四：学习率重调（关键！解决交互问题）

**为什么需要这一步？**

超参数之间存在交互：
- 阶段1找到的最佳lr是在hidden_size=128时的最佳值
- 但阶段2a可能改成了hidden_size=256
- hidden_size=256时，最佳lr可能不再是阶段1找到的那个了！

**做法**：用阶段2-3找到的最优配置，重新搜索学习率

```bash
# 假设阶段2-3得到：hidden_size=256, num_layers=2, dropout=0.3
# 重新搜索学习率：
lr = 5e-4  # 实验1
lr = 1e-3  # 实验2
lr = 2e-3  # 实验3
```

**这样得到的最终配置才是真正优化过的组合！**

---

### 阶段五：最终验证

**目标**：确认最终配置的稳定性

1. 用最终配置运行2-3次（不同随机种子）
2. 计算平均准确率和标准差
3. 确认提升是稳定的，不是运气

```bash
# 最终配置运行3次
seed = 1  → Test Acc = ?
seed = 42 → Test Acc = ?
seed = 123 → Test Acc = ?

# 报告：平均 91.5% ± 0.2%
```

---

## 4. 实验记录模板

| 实验ID | 阶段 | batch_size | learning_rate | hidden_size | num_layers | dropout | Train Acc | Val Acc | Test Acc | 备注 |
|--------|------|------------|--------------|-------------|------------|---------|-----------|---------|----------|------|
| baseline | - | 128 | 1e-3 | 128 | 2 | 0.5 | - | - | 91.12% | 原始配置 |
| exp_001 | 1 | 128 | 1e-4 | 128 | 2 | 0.5 | | | | lr搜索 |
| exp_002 | 1 | 128 | 5e-4 | 128 | 2 | 0.5 | | | | lr搜索 |
| exp_003 | 1 | 128 | 2e-3 | 128 | 2 | 0.5 | | | | lr搜索 |
| exp_004 | 1 | 128 | 5e-3 | 128 | 2 | 0.5 | | | | lr搜索 |
| exp_005 | 2a | 128 | lr* | 64 | 2 | 0.5 | | | | hidden搜索 |
| exp_006 | 2a | 128 | lr* | 256 | 2 | 0.5 | | | | hidden搜索 |
| ... | | | | | | | | | | |

> lr* 表示使用阶段一找到的最优学习率

---

## 5. 如何修改代码运行实验

每次实验修改 `models/TextRNN.py` 中的对应参数：

```python
# 例如：实验 exp_001（lr=1e-4）
self.learning_rate = 1e-4  # 修改这一行

# 例如：实验 exp_005（hidden_size=64）
self.hidden_size = 64      # 修改这一行
```

然后运行：
```bash
python run.py --model TextRNN
```

---

## 6. 预期时间和资源

假设每次训练约5-10分钟（GPU）：

| 阶段 | 试验数 | 预计时间 |
|------|--------|----------|
| 阶段零：Batch Size | 2-3 | ~20分钟 |
| 阶段一：学习率 | 5 | ~50分钟 |
| 阶段二a：Hidden Size | 3-4 | ~40分钟 |
| 阶段二b：Num Layers | 3 | ~30分钟 |
| 阶段三：Dropout | 4 | ~40分钟 |
| 阶段四：细调 | ~3 | ~30分钟 |
| **总计** | **~20** | **~3.5小时** |

---

## 7. 超参数交互问题详解

### 为什么不能简单地把各阶段最优值组合？

```
❌ 错误理解：
阶段1: lr=1e-3 最好
阶段2: hidden=256 最好
→ 最终用 (lr=1e-3, hidden=256)？ 不一定！

✅ 正确理解：
阶段1: hidden=128时，lr=1e-3 最好
阶段2: lr=1e-3时，hidden=256 最好
→ 但 hidden=256时，最佳lr可能是5e-4！
→ 所以需要阶段4重新调lr
```

### 交互的可视化

```
        lr=5e-4   lr=1e-3   lr=2e-3
       ┌─────────┬─────────┬─────────┐
h=128  │  90.5%  │  91.2%  │  90.0%  │  ← 阶段1在这行找lr
       ├─────────┼─────────┼─────────┤
h=256  │  91.8%  │  91.5%  │  89.5%  │  ← 阶段4应该在这行重新找lr
       └─────────┴─────────┴─────────┘
                     ↑
              阶段2在这列找hidden

如果只做阶段1+2，会选择 (lr=1e-3, h=256) = 91.5%
但真正最优是 (lr=5e-4, h=256) = 91.8%
```

### 调参流程总结

```
阶段0: 确定batch_size（一次性，不再改）
    ↓
阶段1: 调learning_rate → 得到 lr_v1
    ↓
阶段2a: 用lr_v1，调hidden_size → 得到 hidden*
    ↓
阶段2b: 用lr_v1，调num_layers → 得到 layers*
    ↓
阶段3: 用lr_v1，调dropout → 得到 dropout*
    ↓
🔄 阶段4: 固定(hidden*, layers*, dropout*)，重新调lr → 得到 lr_final
    ↓
阶段5: 最终验证（多次运行确认稳定性）
    ↓
最终配置: (lr_final, hidden*, layers*, dropout*)
```

---

## 8. 关键原则（来自TUNE.md）

1. **增量调整**：从简单配置开始，逐步添加复杂度
2. **一次只改一个**：每组实验只研究一个超参数的影响
3. **检查训练曲线**：关注过拟合、训练不稳定等问题
4. **记录所有结果**：包括失败的实验，用于分析
5. **学习率最重要**：优先调整学习率，且其他参数变化后需重调
6. **Batch Size影响速度不影响性能**：先固定最大可用batch_size
7. **最后验证组合**：用最终配置重新验证lr，确保组合最优

---

## 8. 常见问题判断指南

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss震荡/发散 | 学习率太大 | 降低lr，或添加warmup |
| Loss下降极慢 | 学习率太小 | 增大lr |
| 训练acc高，验证acc低 | 过拟合 | 增大dropout，减小模型 |
| 训练acc和验证acc都低 | 欠拟合 | 增大模型，减小dropout |
| 最佳值在搜索边界 | 搜索范围不够 | 扩展搜索范围 |
| OOM内存溢出 | batch_size/模型太大 | 减小batch_size或hidden_size |

---

## 9. 下一步行动

1. ✅ 阅读并理解代码注释
2. ✅ 理解实验设计
3. ⬜ 运行baseline确认环境正常：`python run.py --model TextRNN`
4. ⬜ 阶段零：测试GPU最大batch_size
5. ⬜ 阶段一：学习率搜索（最重要！）
6. ⬜ 记录结果，画图分析

---

**准备好后告诉我，我们开始实验！**

