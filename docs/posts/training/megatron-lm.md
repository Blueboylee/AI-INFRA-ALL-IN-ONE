---
title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
date: 2026-02-15
---

# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-15 &nbsp;·&nbsp; 分布式训练 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
> - **机构**: NVIDIA
> - **发表**: arXiv 2019 (后续被广泛引用)
> - **链接**: [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

## 一句话总结

Megatron-LM 提出了一套针对 Transformer 架构的 **层内张量并行（Intra-layer Tensor Parallelism）** 方案，通过对 MLP 层、Self-Attention 层、Embedding 层和交叉熵损失的精心切分，仅需 **每层 2 次 AllReduce**（前向 + 反向各 1 次）即可实现高效的模型并行，在 512 张 V100 上训练了当时最大的 **83 亿参数** Transformer 语言模型，达到 **76% 的弱扩展效率**。

---

## Introduction：为什么需要 Megatron-LM？

### 1. 大模型的崛起与单卡内存瓶颈

2018-2019 年，预训练语言模型的参数量快速增长：

| 模型 | 发布时间 | 参数量 | 关键突破 |
|------|---------|--------|---------|
| BERT-Large | 2018.10 | 340M | 双向预训练 |
| GPT-2 | 2019.02 | 1.5B | 大规模单向 LM |
| XLNet | 2019.06 | 340M | 排列语言模型 |
| **Megatron-LM** | **2019.09** | **8.3B** | **高效张量并行** |

GPT-2 (1.5B) 的参数在 fp16 下占 3 GB，加上优化器状态和激活值，**单张 V100 (32 GB) 已经非常紧张**。想训练 8B+ 的模型，必须将模型切分到多张 GPU 上。

### 2. 数据并行的局限

[ZeRO 系列](./deepspeed-zero) 解决了数据并行中的内存冗余问题。但在 ZeRO 发表之前（2019 年），标准数据并行的方式是每张卡存完整模型，受限于单卡显存。

即使有了 ZeRO，数据并行仍有一个根本限制：**它不能减少单层的计算/内存开销**。如果模型的一个 Transformer 层就超过了单卡显存，数据并行无能为力——你需要将 **层内的参数切分到多张 GPU** 上。

### 3. 现有模型并行方法的问题

2019 年已有一些模型并行的实现，但都存在明显缺陷：

**流水线并行（Pipeline Parallelism）**：
- 将不同层放到不同 GPU
- 简单直观，但存在严重的 **流水线气泡（Pipeline Bubble）**——当前方 GPU 在算前向传播时，后方 GPU 在空等
- 典型效率只有 50-60%

**朴素张量并行（Naive Tensor Parallelism）**：
- 将每层的参数矩阵简单切分
- 需要大量的 **点对点通信**（Send/Recv），通信模式复杂
- 难以在现有深度学习框架中高效实现

**Mesh-TensorFlow**：
- Google 提出的 TPU 上的张量并行框架
- 需要重新定义整个计算图，**与 PyTorch/TensorFlow 的编程模型不兼容**
- 在 GPU 集群上表现不佳

### 4. Megatron-LM 的核心思想

Megatron-LM 的创新在于：**针对 Transformer 的具体结构，设计精巧的张量切分方式，使得每层只需要 2 次 AllReduce 操作**。

核心原则：
1. **利用矩阵乘法的可分解性**：\(Y = XA\) 中，如果按列切分 \(A = [A_1, A_2]\)，则 \(Y = [XA_1, XA_2]\)——每张卡可以独立计算一部分
2. **最小化同步点**：精心设计切分方式，让需要同步的位置恰好在层的边界，每层只需 1 次前向 AllReduce + 1 次反向 AllReduce
3. **不修改模型代码的语义**：只需在现有 PyTorch 代码中插入少量通信原语

::: tip 张量并行 vs 数据并行 vs 流水线并行
- **数据并行（DP）**：每卡存完整模型，切分数据 → 通信梯度
- **张量并行（TP）**：每卡存 **一层的一部分**，共同计算一个 batch → 通信中间激活值
- **流水线并行（PP）**：每卡存 **不同的层**，流水线执行 → 通信层间激活值

Megatron-LM 专注于 TP，后续论文（Megatron-LM v2, 2021）将三者结合形成 **3D 并行**。
:::

### 5. 论文的主要贡献

1. **Transformer 专用的张量并行方案**：针对 MLP 层和 Self-Attention 层设计了简洁高效的切分策略，每层仅需 2 次 AllReduce

2. **Embedding 层和 Cross-Entropy 的并行化**：对输入/输出 Embedding 和交叉熵损失函数也进行了并行切分，避免在巨大词表上的冗余计算

3. **高效工程实现**：在 PyTorch 中用简洁的自定义算子（`f` 和 `g`）实现，无需修改框架底层

4. **规模化验证**：成功训练了 8.3B 参数的 GPT-2 和 3.9B 参数的 BERT，在 512 张 V100 上达到 76% 的弱扩展效率

5. **下游任务 SOTA**：8.3B GPT-2 在 WikiText-103 上达到困惑度 10.8（当时 SOTA），3.9B BERT 在多项 NLU 基准上超越 RoBERTa

---

## 预备知识：Transformer 的计算结构

在深入 Megatron-LM 的切分方案之前，我们需要回顾 Transformer 的核心计算模块。

### 标准 Transformer 层

一个标准的 Transformer 层由两个子模块组成：

```
Transformer 层的计算流程:

输入 X ∈ R^{b×s×h}  (batch × sequence × hidden)
        │
        ↓
┌───────────────────────┐
│  Multi-Head Attention  │
│                       │
│  Q = XW_Q             │  W_Q, W_K, W_V ∈ R^{h×h}
│  K = XW_K             │
│  V = XW_V             │
│  Attn = softmax(QK^T/√d)V  │
│  Y = Attn · W_O       │  W_O ∈ R^{h×h}
│                       │
│  输出 = LayerNorm(X + Y)  │  ← 残差连接 + 归一化
└───────────┬───────────┘
            │
            ↓
┌───────────────────────┐
│  Feed-Forward (MLP)    │
│                       │
│  H = GeLU(X · W_1)    │  W_1 ∈ R^{h×4h}  (扩展 4 倍)
│  Y = H · W_2          │  W_2 ∈ R^{4h×h}  (缩回)
│                       │
│  输出 = LayerNorm(X + Y)  │  ← 残差连接 + 归一化
└───────────┬───────────┘
            │
            ↓
        输出 X' ∈ R^{b×s×h}
```

### 参数量分析

对于隐藏维度 \(h\) 的 Transformer 层：

| 模块 | 参数矩阵 | 形状 | 参数量 |
|------|---------|------|--------|
| Q 投影 | \(W_Q\) | \(h \times h\) | \(h^2\) |
| K 投影 | \(W_K\) | \(h \times h\) | \(h^2\) |
| V 投影 | \(W_V\) | \(h \times h\) | \(h^2\) |
| 输出投影 | \(W_O\) | \(h \times h\) | \(h^2\) |
| MLP 第一层 | \(W_1\) | \(h \times 4h\) | \(4h^2\) |
| MLP 第二层 | \(W_2\) | \(4h \times h\) | \(4h^2\) |
| **每层合计** | — | — | \(12h^2\) |

对于 8.3B 参数的模型（\(h = 3072, L = 72\) 层）：每层约 \(12 \times 3072^2 \approx 113M\) 参数。

### 核心观察：矩阵乘法可以按列/行切分

这是理解 Megatron-LM 的数学基础。对于矩阵乘法 \(Y = XA\)：

**按列切分（Column Parallelism）**：

$$
A = [A_1, A_2] \implies Y = X[A_1, A_2] = [XA_1, XA_2] = [Y_1, Y_2]
$$

每张卡拿到 \(A\) 的一部分列，用**完整的** \(X\) 做乘法，得到 \(Y\) 的一部分列。**无需通信**就能独立计算。

**按行切分（Row Parallelism）**：

$$
A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix},\; X = [X_1, X_2] \implies Y = X_1 A_1 + X_2 A_2
$$

每张卡拿到 \(A\) 的一部分行和 \(X\) 的对应列，做乘法后需要 **AllReduce 求和**。

---

## MLP 层的张量并行

MLP 层是 Megatron-LM 张量并行方案的核心。论文的切分方式简洁而精妙。

### MLP 的计算结构

标准 MLP 由两个线性变换 + GeLU 激活组成：

$$
Y = \text{GeLU}(XW_1) \cdot W_2
$$

其中 \(W_1 \in \mathbb{R}^{h \times 4h}\)（扩展），\(W_2 \in \mathbb{R}^{4h \times h}\)（收缩）。

### 切分策略：列并行 + 行并行

Megatron-LM 对两个权重矩阵采用互补的切分方式：

- \(W_1\)：**按列切分（Column Parallel）** → 每张卡得到 \(W_1\) 的一部分列
- \(W_2\)：**按行切分（Row Parallel）** → 每张卡得到 \(W_2\) 的一部分行

```
MLP 层的张量并行 (2 张 GPU):

           GPU 0                        GPU 1
     ┌──────────────┐             ┌──────────────┐
     │              │             │              │
X ──→│ W_1 的左半列  │         X ──→│ W_1 的右半列  │
     │ h × 2h       │             │ h × 2h       │
     │      ↓       │             │      ↓       │
     │  GeLU(XW_1₁) │             │  GeLU(XW_1₂) │
     │  [b,s,2h]    │             │  [b,s,2h]    │
     │      ↓       │             │      ↓       │
     │ W_2 的上半行  │             │ W_2 的下半行  │
     │ 2h × h       │             │ 2h × h       │
     │      ↓       │             │      ↓       │
     │    Y₁        │             │    Y₂        │
     └──────┬───────┘             └──────┬───────┘
            │                            │
            └──────── AllReduce ─────────┘
                    Y = Y₁ + Y₂
```

### 为什么这样切分是对的？

让我们用数学严格验证。设 \(t = 2\) 张 GPU，将 \(W_1\) 按列分为 \(W_1 = [W_{1,1}, W_{1,2}]\)，将 \(W_2\) 按行分为 \(W_2 = \begin{bmatrix} W_{2,1} \\ W_{2,2} \end{bmatrix}\)。

**Step 1**：每张卡独立计算第一个线性层 + 激活函数：

$$
H_i = \text{GeLU}(X W_{1,i}), \quad i = 1, 2
$$

**关键问题**：GeLU 是非线性函数，\(\text{GeLU}(XW_1)\) 能否拆成 \([\text{GeLU}(XW_{1,1}), \text{GeLU}(XW_{1,2})]\)？

**答案是可以的！** 因为列切分意味着 \(XW_1 = [XW_{1,1}, XW_{1,2}]\)，GeLU 是逐元素操作，所以：

$$
\text{GeLU}([XW_{1,1}, XW_{1,2}]) = [\text{GeLU}(XW_{1,1}), \text{GeLU}(XW_{1,2})]
$$

这正是列切分与非线性激活函数兼容的原因。

::: warning 如果按行切分 \(W_1\) 呢？
如果 \(W_1\) 按行切分，则 \(XW_1 = X_1 W_{1,1} + X_2 W_{1,2}\)，在 GeLU 之前需要先 AllReduce 求和。因为 \(\text{GeLU}(a+b) \neq \text{GeLU}(a) + \text{GeLU}(b)\)——非线性函数不能分配到加法上。这就多了一次通信！所以 **第一层必须按列切分**。
:::

**Step 2**：每张卡独立计算第二个线性层：

$$
Y_i = H_i \cdot W_{2,i}, \quad i = 1, 2
$$

注意 \(H_i \in \mathbb{R}^{b \times s \times 2h}\) 和 \(W_{2,i} \in \mathbb{R}^{2h \times h}\)，维度刚好匹配。

**Step 3**：AllReduce 求和得到最终结果：

$$
Y = Y_1 + Y_2 = H_1 W_{2,1} + H_2 W_{2,2} = [H_1, H_2] \begin{bmatrix} W_{2,1} \\ W_{2,2} \end{bmatrix} = \text{GeLU}(XW_1) \cdot W_2
$$

**结果与单卡计算完全一致！**

### 通信分析

MLP 层的前向传播只需要 **1 次 AllReduce**（在 \(W_2\) 之后），反向传播也只需要 **1 次 AllReduce**（将梯度传回 \(W_1\) 之前）。

| 方向 | AllReduce 次数 | 数据量 | 位置 |
|------|---------------|--------|------|
| 前向 | 1 | \(b \times s \times h\) | \(W_2\) 输出后 |
| 反向 | 1 | \(b \times s \times h\) | \(W_1\) 梯度前 |

---

## Self-Attention 层的张量并行

### 多头注意力的天然可并行性

多头注意力（Multi-Head Attention）有一个天然的并行结构：**各个头之间是完全独立的计算**。

标准 Multi-Head Attention：

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_k) \cdot W_O
$$

其中 \(\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)\)。

Megatron-LM 的策略是：**将注意力头均匀分配到各 GPU**。

### 切分方式

设有 \(t\) 张 GPU，\(k\) 个注意力头（要求 \(k\) 能被 \(t\) 整除），每张卡分到 \(k/t\) 个头：

```
Self-Attention 的张量并行 (t=2, k=16 个头):

              GPU 0                         GPU 1
        ┌───────────────┐            ┌───────────────┐
        │ 头 0-7 (8个头) │            │ 头 8-15(8个头) │
    X ──→ W_Q₁,W_K₁,W_V₁│        X ──→ W_Q₂,W_K₂,W_V₂│
        │     ↓         │            │     ↓         │
        │ Q₁,K₁,V₁     │            │ Q₂,K₂,V₂     │
        │     ↓         │            │     ↓         │
        │ Attention₁    │            │ Attention₂    │
        │ (8 个头的输出)  │            │ (8 个头的输出)  │
        │     ↓         │            │     ↓         │
        │   × W_O₁      │            │   × W_O₂      │
        │ (行并行切分)    │            │ (行并行切分)    │
        │     ↓         │            │     ↓         │
        │    Y₁         │            │    Y₂         │
        └──────┬────────┘            └──────┬────────┘
               │                            │
               └──────── AllReduce ─────────┘
                       Y = Y₁ + Y₂
```

### 实现细节

投影矩阵 \(W_Q, W_K, W_V\) 的切分方式与 MLP 的 \(W_1\) 相同——**按列切分**：

$$
W_Q = [W_Q^{(1)}, W_Q^{(2)}], \quad W_K = [W_K^{(1)}, W_K^{(2)}], \quad W_V = [W_V^{(1)}, W_V^{(2)}]
$$

每张卡的 \(W_Q^{(i)} \in \mathbb{R}^{h \times (h/t)}\)，对应 \(k/t\) 个头的投影。

输出投影矩阵 \(W_O\) 按行切分（与 MLP 的 \(W_2\) 相同），每张卡计算部分结果后 AllReduce 求和。

**整个 Self-Attention 层也只需要 1 次前向 AllReduce + 1 次反向 AllReduce。**

### 与 MLP 的统一视角

| 模块 | 第一组权重 | 切分方式 | 第二组权重 | 切分方式 | 通信 |
|------|-----------|---------|-----------|---------|------|
| MLP | \(W_1\) (h×4h) | **列并行** | \(W_2\) (4h×h) | **行并行** | 1 AllReduce |
| Attention | \(W_{QKV}\) (h×3h) | **列并行** | \(W_O\) (h×h) | **行并行** | 1 AllReduce |

两者遵循完全相同的模式：**第一个矩阵列切分 → 各卡独立计算 → 第二个矩阵行切分 → AllReduce 求和**。

---

## 通信原语 \(f\) 和 \(g\)

论文定义了两个简洁的通信原语来封装张量并行的通信逻辑。

### 定义

$$
f: \text{前向 = Identity（恒等）}, \quad \text{反向 = AllReduce}
$$

$$
g: \text{前向 = AllReduce}, \quad \text{反向 = Identity（恒等）}
$$

### 含义

- \(f\) 放在层的**输入端**：前向传播时，每张卡直接使用完整的输入（因为上一层的 \(g\) 已经 AllReduce 过了）；反向传播时，需要 AllReduce 收集梯度
- \(g\) 放在层的**输出端**：前向传播时，AllReduce 汇总各卡的部分结果；反向传播时，梯度直接分发给各卡

### 在 Transformer 层中的应用

```
一个完整 Transformer 层的通信模式:

输入 X (所有 GPU 相同)
    │
    f (前向: Identity, 反向: AllReduce)
    │
┌───┴───────────────────────────┐
│ Self-Attention (列并行 → 行并行) │
└───┬───────────────────────────┘
    │
    g (前向: AllReduce, 反向: Identity)
    │
    + (残差连接)
    │
    LayerNorm
    │
    f (前向: Identity, 反向: AllReduce)
    │
┌───┴───────────────────────────┐
│ MLP (列并行 → 行并行)           │
└───┬───────────────────────────┘
    │
    g (前向: AllReduce, 反向: Identity)
    │
    + (残差连接)
    │
    LayerNorm
    │
输出 X' (所有 GPU 相同)

总通信: 前向 2 次 AllReduce (两个 g)
       反向 2 次 AllReduce (两个 f)
```

### PyTorch 实现

论文的 \(f\) 和 \(g\) 在 PyTorch 中只需几行自定义 `autograd.Function`：

```python
class f(torch.autograd.Function):
    """输入端通信原语: 前向=Identity, 反向=AllReduce"""
    @staticmethod
    def forward(ctx, x):
        return x  # 前向不通信

    @staticmethod
    def backward(ctx, grad):
        # 反向 AllReduce: 收集所有卡的梯度
        torch.distributed.all_reduce(grad)
        return grad


class g(torch.autograd.Function):
    """输出端通信原语: 前向=AllReduce, 反向=Identity"""
    @staticmethod
    def forward(ctx, x):
        # 前向 AllReduce: 汇总所有卡的部分结果
        torch.distributed.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad  # 反向不通信
```

::: tip 优雅的设计
通过 \(f\) 和 \(g\)，Megatron-LM 将所有通信逻辑封装成两个 **即插即用** 的原语。在现有 PyTorch 模型代码中，只需在每个并行子模块的输入端插入 `f`、输出端插入 `g`，就完成了张量并行的改造——无需修改任何计算逻辑。
:::

---

## Embedding 层的并行化

### 输入 Embedding

输入 Embedding 矩阵 \(E \in \mathbb{R}^{V \times h}\)（\(V\) 为词表大小，\(h\) 为隐藏维度）可能非常大。以 GPT-2 为例：\(V = 50257, h = 3072\)，则 \(E\) 有约 154M 参数。

Megatron-LM **按行切分** Embedding 矩阵（即按词表维度切分）：

$$
E = \begin{bmatrix} E_1 \\ E_2 \\ \vdots \\ E_t \end{bmatrix}, \quad E_i \in \mathbb{R}^{(V/t) \times h}
$$

每张 GPU 负责词表中 \(V/t\) 个 token 的 Embedding 查找。对于不在本卡词表范围内的 token，输出全零向量。最后通过 **AllReduce** 求和得到完整结果。

```
Embedding 并行 (t=2, V=50000):

输入 token_ids: [103, 25001, 7, 30000]

GPU 0 (词表 0-24999):                GPU 1 (词表 25000-49999):
  103    → E[103]                      103    → [0, 0, ..., 0]
  25001  → [0, 0, ..., 0]             25001  → E[25001]
  7      → E[7]                        7      → [0, 0, ..., 0]
  30000  → [0, 0, ..., 0]             30000  → E[30000]

         AllReduce (求和)
              ↓
  结果: [E[103], E[25001], E[7], E[30000]]  ← 完整的 Embedding 输出
```

### 输出 Embedding（语言模型头）

语言模型的输出层需要将隐藏状态投影到词表维度：\(\text{logits} = X W_{\text{out}}^\top\)，其中 \(W_{\text{out}} \in \mathbb{R}^{V \times h}\)。

为了避免内存浪费，Megatron-LM 与输入 Embedding **共享权重**（tied weights），即 \(W_{\text{out}} = E\)。输出层采用同样的切分方式，每张卡计算 \(V/t\) 个 token 的 logits。

---

## Cross-Entropy 损失的并行化

### 问题：Softmax 需要完整的 logits

语言模型的 Cross-Entropy 损失需要对整个词表做 Softmax：

$$
\text{loss} = -\log \frac{e^{z_y}}{\sum_{j=1}^{V} e^{z_j}}
$$

其中 \(z_j\) 是第 \(j\) 个 token 的 logit。Softmax 的分母是 **对所有 \(V\) 个 logit 求和**，这要求所有 logit 在同一个设备上。

如果先 AllGather 所有 logit 到每张卡（\(b \times s \times V\) 的张量），当 \(V\) 很大时（如 50K），这个张量会非常大——**比模型参数还大**。

### Megatron-LM 的解决方案

论文的做法是 **在分布式下直接计算 Cross-Entropy**，避免 AllGather 完整 logit：

**Step 1**：每张卡在本地 logit 上计算局部最大值和局部 exp 之和

$$
m_i = \max_{j \in \text{local}} z_j, \quad s_i = \sum_{j \in \text{local}} e^{z_j - m_i}
$$

**Step 2**：AllReduce 求全局最大值和全局 exp 之和

$$
m = \max_i m_i, \quad s = \sum_i s_i \cdot e^{m_i - m}
$$

**Step 3**：每张卡在本地计算梯度，只对自己负责的 logit 部分计算

这种方式只需要 AllReduce 2 个标量（最大值和 exp 和），比 AllGather 整个 logit 张量高效得多。

```
Cross-Entropy 并行计算:

GPU 0 (logits 0-24999)         GPU 1 (logits 25000-49999)
  local_max₀ = max(z₀..z₂₄₉₉₉)    local_max₁ = max(z₂₅₀₀₀..z₄₉₉₉₉)
  local_sum₀ = Σ exp(z-local_max₀)  local_sum₁ = Σ exp(z-local_max₁)
         │                                  │
         └──── AllReduce max,sum ───────────┘
         │                                  │
  global_max = max(local_max₀, local_max₁)
  global_sum = sum₀·e^{m₀-m} + sum₁·e^{m₁-m}
         │                                  │
  loss = -z_target + global_max + log(global_sum)
  ↓ 各卡只对自己的 logit 分片计算梯度

通信量: 2 个标量 (max 和 sum), 而非 b×s×V 的完整 logit!
```

::: tip 与 FlashAttention 的联系
这种"在线 Softmax"的思路与 [FlashAttention](../inference/flash-attention) 中的在线 Softmax 技巧如出一辙——都是将全局的 max 和 sum 通过增量更新的方式分布式计算，避免实体化完整的 Softmax 输入。
:::

---

## 完整 Transformer 的张量并行

将所有模块组合起来，我们可以看到 Megatron-LM 如何并行化一个完整的 GPT 模型：

```
GPT 模型的完整张量并行 (t=2 张 GPU):

                    GPU 0              GPU 1
                    ┌────┐             ┌────┐
  Input Token IDs ──→ E₁ │             │ E₂ ←── Input Token IDs
                    └──┬─┘             └──┬─┘
                       └── AllReduce ─────┘
                              │
                    ┌─────────┴─────────┐
                    │   + Position Emb  │
                    │   Dropout         │
                    └─────────┬─────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │              × L 层 Transformer           │
         │                                          │
         │  ┌──── f ────┐          ┌──── f ────┐   │
         │  │ Attn 头0-7 │          │ Attn 头8-15│   │
         │  │ → W_O₁    │          │ → W_O₂    │   │
         │  └──── g ────┘          └──── g ────┘   │
         │        │ AllReduce            │          │
         │        └──────────┬───────────┘          │
         │                   │ + Residual + LN      │
         │  ┌──── f ────┐   │     ┌──── f ────┐   │
         │  │ MLP W₁₁   │   │     │ MLP W₁₂   │   │
         │  │ GeLU       │   │     │ GeLU       │   │
         │  │ W₂₁       │   │     │ W₂₂       │   │
         │  └──── g ────┘   │     └──── g ────┘   │
         │        │ AllReduce│           │          │
         │        └──────────┬───────────┘          │
         │                   │ + Residual + LN      │
         └───────────────────┴──────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Final LayerNorm  │
                    └─────────┬─────────┘
                              │
                    ┌──── 分片 logit 计算 ────┐
                    │ logit₁   │   logit₂     │
                    └──── 并行 CrossEntropy ───┘
                              │
                            Loss
```

### 每层通信量汇总

| 模块 | 前向 AllReduce | 反向 AllReduce | 数据形状 |
|------|---------------|---------------|---------|
| Self-Attention | 1 次 | 1 次 | \(b \times s \times h\) |
| MLP | 1 次 | 1 次 | \(b \times s \times h\) |
| **每层合计** | **2 次** | **2 次** | — |

对于 \(L\) 层的模型，总共 \(4L\) 次 AllReduce。每次 AllReduce 的数据量为 \(b \times s \times h\) 个元素。

---

## 通信效率的深入分析

### AllReduce 的成本模型

对于 \(t\) 张 GPU 之间的 AllReduce，使用 Ring AllReduce 算法的通信量为：

$$
\text{通信量} = \frac{2(t-1)}{t} \cdot n \approx 2n \text{ （当 } t \text{ 较大时）}
$$

其中 \(n\) 是要归约的元素数。

### Megatron-LM 的总通信量

每步训练的前向传播通信量：

$$
C_{\text{forward}} = 2L \times 2bsh = 4Lbsh \text{ 元素}
$$

反向传播类似，总通信量约 \(8Lbsh\) 元素。

### 与数据并行的对比

| 并行策略 | 通信内容 | 通信量 | 通信频率 |
|---------|---------|--------|---------|
| 数据并行 | 梯度（模型参数） | \(2 \times 12Lh^2\) | 每步 1 次 |
| 张量并行 | 激活值 | \(8Lbsh\) | 每步 \(4L\) 次 |

关键区别：
- **数据并行**：通信量与 **模型参数量** 成正比，通信次数少（每步 1 次 AllReduce）
- **张量并行**：通信量与 **激活值大小** 成正比，通信次数多（每层 4 次 AllReduce）

当 batch size \(b\) 和序列长度 \(s\) 较大时，张量并行的通信量可能超过数据并行。这就是为什么 **张量并行适合节点内（NVLink 高带宽）**，**数据并行适合跨节点（以太网/InfiniBand）**。

### NVLink 的重要性

V100 DGX-2 节点内 8 张 GPU 通过 NVSwitch 互连，提供每对 GPU 之间 **300 GB/s** 的双向带宽，比 PCIe 3.0 (32 GB/s) 快约 10 倍。

```
通信带宽与并行策略的匹配:

节点内 (NVLink/NVSwitch):
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  │GPU0 │GPU1 │GPU2 │GPU3 │GPU4 │GPU5 │GPU6 │GPU7 │
  │←──────── 300 GB/s 全互连 (NVSwitch) ────────→│
  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  → 张量并行 (TP): 频繁通信, 需要高带宽 ✓

跨节点 (InfiniBand):
  ┌──────────┐     100 Gb/s     ┌──────────┐
  │  节点 0   │ ←───────────→  │  节点 1   │
  └──────────┘                  └──────────┘
  → 数据并行 (DP): 通信少, 带宽要求低 ✓
```

---

## 混合精度训练

Megatron-LM 使用 **混合精度训练** 来加速计算和减少内存：

### 策略

| 操作 | 精度 | 原因 |
|------|------|------|
| 前向/反向矩阵乘法 | fp16 | 利用 V100 Tensor Core (125 TFLOPS) |
| 权重存储 | fp16 (计算用) + fp32 (主副本) | fp16 计算快，fp32 保证更新精度 |
| 激活值 | fp16 | 减少内存和通信带宽 |
| 损失缩放（Loss Scaling） | 动态 | 防止 fp16 下溢 |
| AllReduce | fp16 | 减少通信量 |
| 优化器状态 | fp32 | 数值稳定性 |

### 动态 Loss Scaling

fp16 的可表示范围有限（最小正数约 \(6 \times 10^{-8}\)），小梯度值可能被截断为 0（下溢）。动态 Loss Scaling 的策略：

1. 用一个 scale factor \(S\) 乘以 loss（初始值通常为 \(2^{16}\)）
2. 反向传播的梯度也被放大 \(S\) 倍（链式法则自动实现）
3. 优化器更新前将梯度除以 \(S\)
4. 如果出现 inf/NaN，跳过这步更新并减小 \(S\)
5. 如果连续多步没有 inf/NaN，增大 \(S\)

---

## 激活值内存优化

### 激活值检查点（Activation Checkpointing）

对于大模型，激活值的内存消耗可能远超参数本身。Megatron-LM 使用选择性的 **激活值检查点** 策略：

- 只在每个 Transformer 层的 **输入处** 保存检查点
- 反向传播时从检查点重新计算该层内部的激活值
- 牺牲约 33% 的计算时间（该层要算两遍），换来大幅内存节省

### 张量并行对激活值内存的影响

张量并行不仅切分了参数，还 **自然地切分了部分激活值**：

| 激活值 | 切分情况 | 大小（每卡） |
|--------|---------|-------------|
| Attention 的 QKV | 切分到各卡 | \(b \times s \times (h/t)\) |
| MLP 中间层 | 切分到各卡 | \(b \times s \times (4h/t)\) |
| 层间激活值 | 所有卡相同 | \(b \times s \times h\) |

MLP 中间层（\(4h\) 维度）是激活值的大头，它被 \(t\) 等分后每卡只存 \(1/t\)，显著减少了激活值内存。

---

## 实验结果与关键发现

### 实验设置

- **硬件**：最多 32 台 DGX-2H（每台 16 张 V100 32GB），共 512 张 GPU
- **互连**：节点内 NVSwitch (300 GB/s)，节点间 InfiniBand (8× 100 Gb/s)
- **模型**：GPT-2 和 BERT 架构，参数量从 355M 到 8.3B

### 模型配置

| 模型 | 参数量 | 层数 \(L\) | 隐藏维度 \(h\) | 注意力头数 | TP 度 |
|------|--------|-----------|---------------|-----------|------|
| GPT-2 355M | 355M | 24 | 1024 | 16 | 1 |
| GPT-2 2.5B | 2.5B | 54 | 1920 | 24 | 2 |
| GPT-2 4.2B | 4.2B | 72 | 2304 | 24 | 4 |
| **GPT-2 8.3B** | **8.3B** | **72** | **3072** | **24** | **8** |
| BERT 3.9B | 3.9B | 48 | 2560 | 32 | 8 |

### 扩展效率

**单节点内的强扩展（Strong Scaling）**：

| GPU 数量 (TP 度) | 8.3B 模型 TFLOPS/GPU | 相对效率 |
|-----------------|---------------------|---------|
| 1 | OOM | — |
| 2 | 37.5 | — |
| 4 | 35.6 | 95% |
| 8 | 32.3 | 86% |

在 8 卡 NVLink 互连下，效率保持在 **86%** 以上。从 2 卡到 8 卡仅下降 14%，说明 NVLink 带宽足以支撑张量并行的通信需求。

**多节点的弱扩展（Weak Scaling）**：

| 节点数 × GPU数 | 模型参数量 | 总 TFLOPS | 效率 |
|----------------|-----------|-----------|------|
| 1 × 8 (8) | 1.2B | 236 | 100% (基线) |
| 2 × 16 (32) | 2.5B | 450 | 95% |
| 4 × 32 (64) | 4.2B | 860 | 91% |
| 8 × 64 (128) | 8.3B | 1590 | 84% |
| 32 × 512 (512) | 8.3B | 5530 | **76%** |

512 GPU 的弱扩展效率达到 **76%**，这在 2019 年是非常出色的。

### 语言模型质量

**GPT-2 8.3B 在 WikiText-103 上的困惑度（Perplexity）**：

| 模型 | 参数量 | PPL |
|------|--------|-----|
| GPT-2 (OpenAI) | 1.5B | 17.48 |
| **Megatron GPT-2** | **8.3B** | **10.81** |

参数量从 1.5B 提升到 8.3B，困惑度从 17.48 下降到 **10.81**——证明了更大的模型确实带来更好的语言建模能力。

**BERT 3.9B 在下游任务上**：

| 任务 | RoBERTa-Large (355M) | Megatron BERT 3.9B |
|------|---------------------|-------------------|
| RACE-h | 83.2% | **89.5%** |
| MNLI | 90.2% | **91.4%** |
| QQP | 92.2% | **92.6%** |

在多项 NLU 基准上超越当时的 SOTA（RoBERTa）。

::: warning 效率随 TP 度的下降
虽然 86% 的单节点效率很好，但值得注意的是：当 TP 扩展到 **跨节点** 时，效率会急剧下降（因为跨节点带宽远低于 NVLink）。这就是为什么 Megatron-LM 强调 **TP 只在节点内使用**，跨节点用数据并行。这个经验法则后来成为大模型训练的标准实践。
:::

---

## 与 ZeRO 的互补关系

Megatron-LM 的张量并行和 ZeRO 的数据并行是 **正交互补** 的两种技术：

| 维度 | Megatron-LM (TP) | ZeRO (DP) |
|------|------------------|-----------|
| 切分对象 | 层内的参数矩阵 | 跨所有层的模型状态 |
| 通信内容 | 激活值（\(bsh\)） | 梯度/参数（\(\Psi\)） |
| 通信频率 | 每层 4 次 | 每步 1-2 次 |
| 通信带宽需求 | **高**（需要 NVLink） | **中**（AllGather/RS 效率高） |
| 适合场景 | 节点内（高带宽） | 跨节点（低带宽也可） |
| 解决的问题 | 单层参数超过单卡 | 总模型状态超过单卡 |

### 3D 并行：最佳实践

在后续工作（Megatron-LM v2, 2021）中，NVIDIA 将 TP + DP + PP 组合成 **3D 并行**，这成为 GPT-3、PaLM 等超大模型训练的标准范式：

```
3D 并行架构示例 (64 GPUs):

张量并行 (TP=8): 节点内 NVLink
  ┌─GPU0─GPU1─GPU2─GPU3─GPU4─GPU5─GPU6─GPU7─┐ ← 1 个 TP 组
  └─────────────────────────────────────────┘

流水线并行 (PP=4): 跨节点
  TP组₀ (层1-6) → TP组₁ (层7-12) → TP组₂ (层13-18) → TP组₃ (层19-24)
  节点0           节点1             节点2             节点3

数据并行 (DP=2): 跨 PP 阶段的副本
  TP组₀ᵃ → TP组₁ᵃ → TP组₂ᵃ → TP组₃ᵃ   ← 数据分片 A
  TP组₀ᵇ → TP组₁ᵇ → TP组₂ᵇ → TP组₃ᵇ   ← 数据分片 B

总 GPU = TP × PP × DP = 8 × 4 × 2 = 64
```

每种并行度的选择经验：
- **TP**：等于节点内 GPU 数（通常 8），利用 NVLink
- **PP**：根据模型层数和节点数调整，1-8 之间
- **DP**：用剩余的 GPU 数做数据并行，越大越好（吞吐越高）

---

## 关键技术点汇总

下面的代码模拟 Megatron-LM 张量并行的通信量和内存分布：

```cpp-run title="Megatron-LM 张量并行通信量与内存分析"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>

int main() {
    std::cout << "=============================================================\n";
    std::cout << "    Megatron-LM 张量并行通信量与内存分析\n";
    std::cout << "=============================================================\n\n";

    // ---- 1. 模型配置 ----
    struct ModelConfig {
        const char* name;
        long long params;
        int layers;
        int hidden;
        int heads;
        int tp;
    };

    ModelConfig models[] = {
        {"GPT-2 355M",  355000000LL,  24, 1024, 16, 1},
        {"GPT-2 2.5B",  2500000000LL, 54, 1920, 24, 2},
        {"GPT-2 4.2B",  4200000000LL, 72, 2304, 24, 4},
        {"GPT-2 8.3B",  8300000000LL, 72, 3072, 24, 8},
    };

    // ---- 2. 每层参数量与分配 ----
    std::cout << "1. 每层参数量与张量并行分配\n\n";
    std::cout << std::setw(14) << "模型"
              << std::setw(12) << "h"
              << std::setw(14) << "每层参数"
              << std::setw(8)  << "TP"
              << std::setw(14) << "每卡/层参数" << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (auto& m : models) {
        long long per_layer = 12LL * m.hidden * m.hidden;
        long long per_card_layer = per_layer / m.tp;
        std::cout << std::setw(14) << m.name
                  << std::setw(12) << m.hidden
                  << std::setw(10) << per_layer / 1000000 << " M"
                  << std::setw(8)  << m.tp
                  << std::setw(10) << per_card_layer / 1000000 << " M\n";
    }

    std::cout << "\n";

    // ---- 3. 通信量分析 ----
    std::cout << "=============================================================\n";
    std::cout << "2. 每步训练的通信量 (前向 + 反向)\n";
    std::cout << "   batch=8, seq_len=1024\n";
    std::cout << "=============================================================\n\n";

    int batch = 8, seq = 1024;

    std::cout << std::setw(14) << "模型"
              << std::setw(8)  << "TP"
              << std::setw(8)  << "层数"
              << std::setw(16) << "TP通信总量"
              << std::setw(16) << "DP通信总量"
              << std::setw(10) << "TP/DP" << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (auto& m : models) {
        // TP 通信: 每层 4 次 AllReduce, 每次 b*s*h 元素 (fp16 = 2B)
        double tp_per_layer = 4.0 * batch * seq * m.hidden * 2;  // bytes
        double tp_total = tp_per_layer * m.layers;

        // DP 通信: 2 * 参数量 元素 (fp16 = 2B)
        double dp_total = 2.0 * m.params * 2;

        double ratio = (m.tp > 1) ? tp_total / dp_total : 0;

        std::cout << std::setw(14) << m.name
                  << std::setw(8)  << m.tp
                  << std::setw(8)  << m.layers
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << tp_total / 1e9 << " GB"
                  << std::setw(12) << dp_total / 1e9 << " GB"
                  << std::setw(10) << std::setprecision(2)
                  << (m.tp > 1 ? std::to_string(ratio).substr(0,4) + "x" : "N/A")
                  << "\n";
    }

    std::cout << "\n  关键洞察:\n";
    std::cout << "  TP 通信量 ∝ batch × seq × hidden × layers (激活值大小)\n";
    std::cout << "  DP 通信量 ∝ 模型参数量\n";
    std::cout << "  TP 通信更频繁 (每层4次) 但每次量小 → 需要高带宽低延迟 (NVLink)\n";
    std::cout << "  DP 通信少 (每步1次) 但每次量大 → 可以用较低带宽 (InfiniBand)\n\n";

    // ---- 4. 内存分析: TP 对激活值内存的影响 ----
    std::cout << "=============================================================\n";
    std::cout << "3. 张量并行对激活值内存的影响 (8.3B模型, b=8, s=1024)\n";
    std::cout << "=============================================================\n\n";

    int h = 3072, L = 72;

    std::cout << "  每层激活值组成 (不使用检查点):\n\n";

    struct ActivationItem {
        const char* name;
        double elements_per_tp;  // 每卡元素数 (以 b*s 为单位, 乘以维度)
        bool sharded;
    };

    ActivationItem items[] = {
        {"QKV 投影输入",     (double)h,      false},
        {"Q, K, V",          3.0*h,          true},   // 按头切分
        {"Attention scores", (double)(seq*24), true},  // heads/tp
        {"Attention output", (double)h,       true},
        {"W_O 输出 (AllReduce后)", (double)h, false},
        {"MLP W1 输入",      (double)h,       false},
        {"MLP 中间层 (GeLU)", 4.0*h,          true},   // 按列切分
        {"MLP W2 输出 (AllReduce后)", (double)h, false},
    };

    double total_no_tp = 0, total_tp8 = 0;
    std::cout << std::setw(30) << "激活值"
              << std::setw(16) << "无TP(每卡)"
              << std::setw(16) << "TP=8(每卡)"
              << std::setw(10) << "切分?" << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (auto& item : items) {
        double no_tp = batch * seq * item.elements_per_tp * 2 / 1e9; // GB, fp16
        double tp8 = item.sharded ? no_tp / 8 : no_tp;
        total_no_tp += no_tp;
        total_tp8 += tp8;
        std::cout << std::setw(30) << item.name
                  << std::setw(12) << std::setprecision(3) << no_tp << " GB"
                  << std::setw(12) << tp8 << " GB"
                  << std::setw(10) << (item.sharded ? "YES" : "no") << "\n";
    }
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::setw(30) << "每层合计"
              << std::setw(12) << std::setprecision(3) << total_no_tp << " GB"
              << std::setw(12) << total_tp8 << " GB"
              << std::setw(10) << std::setprecision(1)
              << (1.0 - total_tp8/total_no_tp)*100 << "% 节省\n";
    std::cout << std::setw(30) << "72 层合计"
              << std::setw(12) << total_no_tp * L << " GB"
              << std::setw(12) << total_tp8 * L << " GB" << "\n";

    std::cout << "\n";

    // ---- 5. 扩展效率模型 ----
    std::cout << "=============================================================\n";
    std::cout << "4. NVLink vs PCIe: TP 效率预估 (8.3B 模型)\n";
    std::cout << "=============================================================\n\n";

    // 每步 TP 通信量
    double tp_comm_bytes = 4.0 * L * batch * seq * h * 2;  // bytes

    double nvlink_bw = 300e9;   // 300 GB/s
    double pcie_bw   = 16e9;    // 16 GB/s

    double tp_time_nvlink = tp_comm_bytes / nvlink_bw;
    double tp_time_pcie   = tp_comm_bytes / pcie_bw;

    // 粗估 GPU 计算时间 (8.3B model, fp16)
    double flops = 2.0 * 8.3e9 * batch * seq;  // ~2*Params*tokens
    double gpu_tflops = 125e12;
    double compute_time = flops / gpu_tflops;

    std::cout << "  TP 通信量: " << std::setprecision(1) << tp_comm_bytes / 1e9 << " GB\n";
    std::cout << "  GPU 计算时间 (粗估): " << std::setprecision(3) << compute_time << " s\n\n";

    std::cout << "  NVLink (300 GB/s):\n";
    std::cout << "    通信时间: " << tp_time_nvlink << " s\n";
    std::cout << "    计算效率: " << std::setprecision(1)
              << compute_time / (compute_time + tp_time_nvlink) * 100 << "%\n\n";

    std::cout << "  PCIe 3.0 (16 GB/s):\n";
    std::cout << "    通信时间: " << tp_time_pcie << " s\n";
    std::cout << "    计算效率: "
              << compute_time / (compute_time + tp_time_pcie) * 100 << "%\n\n";

    std::cout << "  结论: NVLink 下 TP 高效, PCIe 下 TP 效率大幅下降\n";
    std::cout << "        → TP 只应在节点内 (NVLink 互连) 使用!\n";

    return 0;
}
```

---

## 总结与启示

### Megatron-LM 的核心贡献

1. **精巧的张量切分方案**：列并行 + 行并行的组合，使 Transformer 每层只需 2 次 AllReduce，通信最小化

2. **\(f\) 和 \(g\) 通信原语**：将张量并行的通信逻辑封装为即插即用的算子，不侵入模型计算代码

3. **端到端的并行化**：不仅是 Attention 和 MLP，还包括 Embedding 层和 Cross-Entropy 损失的并行化，消除所有冗余

4. **工程落地**：证明了在 NVLink 互连下，张量并行可以达到 **86%** 的单节点效率，是大模型训练的核心技术

### 深层设计原则

Megatron-LM 的成功体现了两个重要的系统设计原则：

**原则 1：利用计算图的结构特性**

Megatron-LM 没有设计通用的张量并行框架，而是 **针对 Transformer 的具体结构** 手工设计切分方案。正是因为了解 MLP 的"扩展-收缩"模式和多头注意力的"独立头"结构，才找到了只需 2 次 AllReduce 的最优方案。

**原则 2：匹配通信模式与硬件拓扑**

张量并行的高频通信天然匹配 NVLink 的高带宽低延迟特性。论文明确建议 TP 只在节点内使用——这不是妥协，而是 **将正确的并行策略放在正确的硬件层级**。

::: tip 从 Megatron-LM 到 ZeRO：互补的全景
两条技术路线解决了大模型训练的两个不同维度：

- **Megatron-LM（张量并行）**：解决 **"一层放不下单卡"** 的问题 → 切分层内参数
- **ZeRO（数据并行优化）**：解决 **"所有层的状态放不下单卡"** 的问题 → 切分跨层状态

两者组合（TP + ZeRO-DP），再加上流水线并行（PP），就形成了 **3D 并行**——当今训练 GPT-4、PaLM、LLaMA 等万亿参数模型的标准方法。
:::

---

## 参考文献

1. Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

2. Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., ... & Catanzaro, B. (2021). **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**. SC 2021. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)

3. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

4. Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, D., Chen, M., ... & Wu, Y. (2019). **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**. NeurIPS 2019. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)

5. Shazeer, N., Cheng, Y., Parmar, N., Tran, D., Vaswani, A., Koanantakool, P., ... & Hawkins, J. (2018). **Mesh-TensorFlow: Deep Learning for Supercomputers**. NeurIPS 2018. [arXiv:1811.02084](https://arxiv.org/abs/1811.02084)

6. Smith, S., et al. (2022). **Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model**. [arXiv:2201.11990](https://arxiv.org/abs/2201.11990)
