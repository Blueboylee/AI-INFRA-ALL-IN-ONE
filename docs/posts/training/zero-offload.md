---
title: "ZeRO-Offload: Democratizing Billion-Scale Model Training"
date: 2026-02-15
---

# ZeRO-Offload: Democratizing Billion-Scale Model Training

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-15 &nbsp;·&nbsp; 分布式训练 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He
> - **机构**: Microsoft, University of California Merced
> - **发表**: USENIX ATC 2021
> - **链接**: [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)

## 一句话总结

ZeRO-Offload 将 ZeRO-2 的分片梯度和优化器状态 **卸载（Offload）到 CPU 内存和 CPU 计算**，通过精心设计的 **数据流图分析** 找到最优卸载策略，使得 **单张 GPU** 就能训练超过 **100 亿参数** 的模型，同时维持接近纯 GPU 训练的吞吐量。

---

## Introduction：为什么需要 ZeRO-Offload？

### 1. ZeRO 的遗留问题：GPU 数量依赖

在上一篇 [ZeRO 论文精读](./deepspeed-zero) 中，我们看到 ZeRO 通过分片模型状态在 \(N_d\) 张 GPU 之间消除冗余。但这意味着：**内存节省与 GPU 数量成正比**。

- ZeRO-3 在 64 张 GPU 上可以将每卡内存降到标准 DP 的 1/64
- 但如果你只有 **1 张 GPU**，ZeRO 完全帮不上忙——\(N_d = 1\) 时没有冗余可消除

这对广大研究者和小团队而言是一个巨大的障碍。2020 年的现实是：

| 场景 | 典型硬件 | 可训练最大模型 |
|------|---------|---------------|
| 顶级实验室 | 数百张 V100/A100 | 100B+ |
| 大学实验室 | 1-8 张 GPU | ~1B |
| 个人研究者 | 1 张 GPU | ~0.3B |

**核心问题**：能否让拥有单张 GPU 的研究者也能训练 10B 级别的模型？

### 2. 一个被忽视的资源：CPU

现代服务器的 CPU 内存通常有 **数百 GB 到 TB 级别**，远超 GPU 显存。以一台典型的深度学习工作站为例：

```
典型的深度学习工作站资源对比:

┌──────────────────────────────────────────────┐
│                  GPU (V100)                    │
│  ┌──────────────────────────────────────┐     │
│  │ 显存: 32 GB    算力: 125 TFLOPS (fp16)│     │
│  │ 带宽: 900 GB/s (HBM2)               │     │
│  └──────────────────────────────────────┘     │
│                     │ PCIe 3.0: 32 GB/s       │
│                     ↕                         │
│  ┌──────────────────────────────────────┐     │
│  │ CPU 内存: 256 GB    算力: ~1 TFLOPS   │     │
│  │ 带宽: 50-100 GB/s (DDR4)             │     │
│  └──────────────────────────────────────┘     │
└──────────────────────────────────────────────┘

GPU 算力是 CPU 的 ~100x, 但 CPU 内存是 GPU 显存的 ~8x!
```

CPU 内存大，但 CPU 算力弱、PCIe 带宽低。这意味着我们不能简单地把所有东西搬到 CPU 上算——**卸载策略的设计至关重要**。

### 3. 朴素异构卸载的失败

一些先前工作尝试了将模型训练卸载到 CPU：

- **L2L (Layer-to-Layer)**：将模型参数按层在 GPU ↔ CPU 之间搬运
- **SwapAdvisor**：使用强化学习寻找最优的张量交换策略

但这些方法有两个致命问题：

1. **吞吐量暴跌**：朴素卸载导致 GPU 大量时间在等待 PCIe 传输，利用率可能低于 **10%**
2. **缺乏理论保证**：启发式搜索不能保证找到最优解，且针对不同模型需要重新搜索

### 4. ZeRO-Offload 的核心思想

ZeRO-Offload 的创新在于 **将卸载问题形式化为图优化问题**，通过分析训练过程的数据流图（Data-flow Graph），找到一个同时满足以下条件的最优策略：

1. **最大化 GPU 内存节省**：尽可能多地将模型状态卸载到 CPU
2. **最小化通信开销**：减少 GPU ↔ CPU 之间的 PCIe 传输量
3. **最大化 GPU 计算效率**：将计算密集型操作留在 GPU，只把内存密集但计算轻量的操作放到 CPU

最终方案：**将梯度和优化器状态卸载到 CPU，参数保留在 GPU，参数更新在 CPU 上执行**。

::: tip 与 ZeRO 的关系
ZeRO-Offload 建立在 ZeRO-2 之上。ZeRO-2 将优化器状态和梯度分片到多张 GPU；ZeRO-Offload 更进一步，将这些分片卸载到 CPU。可以理解为：**ZeRO 是 GPU 间的分片，ZeRO-Offload 是 GPU-CPU 间的卸载**。
:::

### 5. 论文的主要贡献

1. **最优卸载策略的形式化推导**：通过数据流图分析，证明将梯度和优化器状态卸载到 CPU 是保持 GPU 计算效率的最优方案

2. **高效的 CPU 执行引擎**：实现了基于 SIMD 向量化的 CPU Adam 优化器，吞吐量比 PyTorch CPU 实现快 **6 倍**

3. **一步延迟参数更新（One-Step Delayed Parameter Update）**：通过将参数更新与下一步的前向传播重叠，进一步隐藏 CPU 计算和通信延迟

4. **单卡可训 10B 模型**：在单张 V100 32GB 上训练最多 **130 亿参数** 的模型，比不使用 Offload 的上限（~14 亿）提升约 **10 倍**

5. **多卡可扩展**：与 ZeRO-2 的多卡数据并行无缝结合

---

## 核心方法：最优卸载策略的推导

ZeRO-Offload 的核心贡献是通过 **数据流图分析（Data-flow Graph Analysis）** 系统性地推导出最优卸载策略。这不是拍脑袋的启发式设计，而是有严格理论支撑的最优解。

### 训练过程的数据流图

论文首先将混合精度训练的完整过程抽象为一个数据流图。图中的 **节点** 代表数据（张量），**边** 代表计算操作。

```
混合精度训练的数据流图:

          ┌─────────────────────────────────────────────────┐
          │                                                 │
          ↓                                                 │
   ┌──────────┐   forward    ┌──────────┐   backward   ┌───┴──────┐
   │ fp16 参数 │ ──────────→ │  激活值    │ ──────────→ │ fp16 梯度 │
   │  (2Ψ B)  │             │           │             │  (2Ψ B)  │
   └──────────┘             └──────────┘             └──────────┘
        ↑                                                │
        │ fp32→fp16                                      │
        │ 转换                                            ↓
   ┌──────────┐   参数更新    ┌──────────┐   float2half  ┌──────────┐
   │ fp32 参数 │ ←────────── │ fp32 梯度 │ ←──────────  │ fp16 梯度 │
   │  (4Ψ B)  │   (Adam)    │  (4Ψ B)  │  half2float  │  (2Ψ B)  │
   └──────────┘             └──────────┘              └──────────┘
        ↑                        ↑
        │                        │
   ┌──────────┐             ┌──────────┐
   │ Adam 动量 │             │ Adam 方差 │
   │  (4Ψ B)  │             │  (4Ψ B)  │
   └──────────┘             └──────────┘

数据节点 (圆角矩形): fp16参数, fp16梯度, fp32梯度, fp32参数, Adam动量, Adam方差
计算边: forward, backward, half2float, Adam更新, float2half
```

### 将图切分为 GPU 子图和 CPU 子图

卸载策略的本质就是：**在数据流图上画一条切割线，将节点和边分配到 GPU 或 CPU**。

每条"切割"产生的跨设备边就是 PCIe 通信。论文的优化目标是：

1. **最小化跨设备通信量**（切割边的数据量最小）
2. **将计算密集型操作保留在 GPU**（forward 和 backward 必须在 GPU）
3. **最大化 CPU 端的模型状态存储**（内存节省最大）

### 关键约束分析

论文通过分析数据流图中各条边的计算特性，得出以下关键约束：

**约束 1：Forward 和 Backward 必须在 GPU**

这两个操作是计算密集型的（涉及大量矩阵乘法），将它们放在 CPU 上会导致训练速度下降 **100 倍以上**。

**约束 2：fp16 参数必须在 GPU**

因为 forward 和 backward 需要读取 fp16 参数，如果参数在 CPU 上，每步训练都要通过 PCIe 传输 \(2\Psi\) 字节，这个开销太大。

**约束 3：fp16 梯度由 backward 产生**

backward 在 GPU 上执行，所以 fp16 梯度首先出现在 GPU 上。

基于这些约束，剩下要决定的是：**fp16 梯度后续的处理（half2float + Adam 更新 + float2half）放在哪里？**

### 穷举所有合法策略

论文枚举了所有满足约束的卸载方案，发现只有 **3 种** 是可行的：

```
策略 A: 全部留在 GPU (基线)
┌─────────────────────────────────────────────────────┐
│ GPU                                                   │
│ fp16参数 → forward → backward → fp16梯度              │
│ → half2float → fp32梯度 → Adam更新 → fp32参数         │
│ → float2half → fp16参数                               │
│ + Adam 动量, Adam 方差                                │
└─────────────────────────────────────────────────────┘
│ CPU: (无)
│ 通信量: 0
│ GPU 内存: 16Ψ (无节省)


策略 B: 卸载 Adam 更新 + 优化器状态到 CPU ← ZeRO-Offload 选择!
┌─────────────────────────────────────────────────────┐
│ GPU                                                   │
│ fp16参数 → forward → backward → fp16梯度              │
│ ← float2half ←───────────────── fp16梯度 ──→ CPU     │
└────────────────────────────────────────────↕─────────┘
                                        PCIe │ 2Ψ (梯度)
┌────────────────────────────────────────────↕─────────┐
│ CPU                                                   │
│ fp16梯度 → half2float → fp32梯度                      │
│ → Adam更新(fp32参数, 动量, 方差) → float2half          │
│ → fp16 参数更新 ──→ GPU                                │
└─────────────────────────────────────────────────────┘
│ 通信量: 2Ψ×2 = 4Ψ (梯度下行 + 参数更新上行), 单方向同时只有 2Ψ
│ GPU 内存: 2Ψ (仅 fp16 参数), 节省 ~87.5%!


策略 C: 仅卸载优化器状态, Adam 更新在 GPU
┌─────────────────────────────────────────────────────┐
│ GPU                                                   │
│ fp16参数 → forward → backward → fp16梯度              │
│ → half2float → fp32梯度 → Adam更新                    │
│ ← Adam 状态 ←── CPU     ──→ Adam 状态 ──→ CPU        │
└────────────────────────────────────────────↕─────────┘
                                    PCIe │ 12Ψ×2 (优化器状态来回)
┌────────────────────────────────────────────↕─────────┐
│ CPU                                                   │
│ Adam 动量, Adam 方差, fp32 参数副本                     │
└─────────────────────────────────────────────────────┘
│ 通信量: 24Ψ (优化器状态来回传输!)  ← 太大了!
│ GPU 内存: 4Ψ (fp16 参数 + fp16 梯度)
```

### 为什么选择策略 B？

| 策略 | GPU 内存 | PCIe 通信量 | CPU 计算 |
|------|---------|------------|---------|
| A (全 GPU) | \(16\Psi\) | 0 | 无 |
| **B (卸载梯度+优化器+更新)** | \(\mathbf{2\Psi}\) | \(\mathbf{4\Psi}\) | **Adam (轻量)** |
| C (仅卸载优化器状态) | \(4\Psi\) | \(24\Psi\) | 无 |

策略 B 是唯一同时满足 **高内存节省** 和 **低通信量** 的方案：

- **内存节省最大**：GPU 上只保留 fp16 参数（\(2\Psi\)），节省 \(14\Psi\)（87.5%）
- **通信量最小**：只需传输 fp16 梯度到 CPU（\(2\Psi\)）和 fp16 参数更新回 GPU（\(2\Psi\)）
- **CPU 计算合理**：Adam 更新是逐元素操作（element-wise），计算密度低，CPU 完全胜任

::: warning 策略 C 为什么不行？
策略 C 虽然内存节省也不错（\(4\Psi\)），但通信量高达 \(24\Psi\)——是策略 B 的 **6 倍**！因为优化器状态（fp32 参数副本 + 动量 + 方差 = \(12\Psi\)）需要在每一步从 CPU 传到 GPU（做 Adam 更新），更新后再传回去。这会让 PCIe 成为严重瓶颈。
:::

---

## ZeRO-Offload 架构详解

### 整体数据流

基于策略 B，ZeRO-Offload 的完整训练流程如下：

```
ZeRO-Offload 单步训练流程:

       GPU                              CPU
  ┌─────────────┐                 ┌─────────────────┐
  │             │                 │                 │
  │ ① 前向传播   │                 │                 │
  │ fp16参数→激活 │                 │                 │
  │             │                 │                 │
  │ ② 反向传播   │                 │                 │
  │ 激活→fp16梯度 │                 │                 │
  │             │   ③ PCIe传输     │                 │
  │ fp16 梯度  ──┼──────────────→  │ ④ half2float    │
  │             │   (2Ψ bytes)    │ fp16梯度→fp32梯度 │
  │             │                 │                 │
  │             │                 │ ⑤ Adam 更新      │
  │             │                 │ fp32 参数 +=     │
  │             │                 │   f(fp32梯度,    │
  │             │   ⑥ PCIe传输     │    动量, 方差)   │
  │ fp16 参数  ←─┼──────────────── │                 │
  │ (更新后)    │   (2Ψ bytes)    │ ⑥ float2half    │
  │             │                 │ fp32参数→fp16参数 │
  └─────────────┘                 └─────────────────┘

GPU 持久存储: fp16 参数 (2Ψ bytes)
CPU 持久存储: fp32 参数 + fp32 动量 + fp32 方差 (12Ψ bytes)
```

### 关键优化 1：高效 CPU Adam 优化器

Adam 更新虽然是逐元素操作，但参数量很大（10B 级别 = 100 亿个元素）。朴素的 CPU 实现会成为瓶颈。

论文实现了一个 **高度优化的 CPU Adam**：

- **SIMD 向量化**：使用 AVX-512 指令，一次处理 16 个 fp32 元素
- **循环展开**：减少分支预测开销
- **NUMA 感知**：在多 socket 系统上利用本地内存带宽
- **多线程并行**：利用 CPU 的多核心

性能对比：

| 实现 | 10B 参数 Adam 更新耗时 |
|------|---------------------|
| PyTorch CPU Adam | ~1.2 秒 |
| **ZeRO-Offload CPU Adam** | **~0.2 秒** |
| GPU Adam (V100) | ~0.03 秒 |

虽然比 GPU 慢约 7 倍，但通过下面的延迟隐藏技术，这个开销可以被有效掩盖。

### 关键优化 2：一步延迟参数更新

这是 ZeRO-Offload 最精巧的优化。核心思想：**当 CPU 在执行第 \(t\) 步的 Adam 更新时，GPU 可以用稍旧的参数先开始第 \(t+1\) 步的前向传播**。

```
流水线时间线 (无延迟 vs 有延迟):

无延迟 (串行执行):
  GPU: [Forward₁][Backward₁][ 等待CPU ][ Forward₂][Backward₂][ 等待CPU ]
  CPU:                      [Adam₁+传输]                    [Adam₂+传输]
  ────────────────────────────────────────────────────────────────→ 时间


一步延迟 (流水线执行):
  GPU: [Forward₁][Backward₁][Forward₂ *][Backward₂][Forward₃ *][Backward₃]
  CPU:           ↓梯度₁     [Adam₁ + 传输↑]  ↓梯度₂  [Adam₂ + 传输↑]
  ────────────────────────────────────────────────────────────────→ 时间

  * Forward₂ 使用的是第 0 步的参数 (延迟 1 步)
    Forward₃ 使用的是第 1 步的参数 (延迟 1 步)

关键: CPU 的 Adam 更新与 GPU 的下一步 Forward 重叠!
      只要 Adam + PCIe 传输时间 < Forward + Backward 时间,
      CPU 开销就完全被隐藏!
```

**收敛性分析**：一步延迟意味着使用"过时"一步的参数做前向传播。论文引用了异步 SGD 的理论分析，证明在学习率适当的情况下，一步延迟不影响收敛性。实验也验证了这一点——有无延迟的训练 loss 曲线几乎完全重合。

::: tip 延迟条件
一步延迟的前提是 CPU 计算时间 + PCIe 传输时间 ≤ GPU 前向 + 反向传播时间。对于大模型（参数量大 → 计算量大），这个条件几乎总是满足的——因为模型越大，GPU 上的计算时间增长更快。
:::

### 关键优化 3：CPU-GPU 通信重叠

除了参数更新的延迟执行，通信本身也可以与计算重叠：

1. **梯度传输与反向传播重叠**：反向传播是逐层进行的。当第 \(l\) 层的梯度计算完毕后，可以立即开始传输到 CPU，同时 GPU 继续计算第 \(l-1\) 层的梯度
2. **参数更新传输与前向传播重叠**：CPU 更新完参数后传回 GPU 时，GPU 可能已经在做下一步的前向传播

```
通信-计算重叠 (反向传播阶段):

GPU 计算:  [Grad_L][Grad_L-1][Grad_L-2]...[Grad_1]
PCIe 传输:      [Transfer Grad_L][Transfer Grad_L-1]...
CPU:                              [开始接收并处理...]

反向传播的梯度 "流式" 传输到 CPU, 而非等全部算完再传!
```

---

## 与多 GPU ZeRO-2 的结合

ZeRO-Offload 不仅适用于单 GPU 场景，还可以与 ZeRO-2 的多卡数据并行无缝结合。

### 多卡场景下的数据流

在 \(N_d\) 张 GPU 的场景中，ZeRO-Offload 的工作方式为：

1. 每张 GPU 上保留 **完整的 fp16 参数**（用于前向/反向）
2. 反向传播后，执行 GPU 间的 **Reduce-Scatter**（与 ZeRO-2 相同），每卡只保留 \(1/N_d\) 的聚合梯度
3. 将分片的聚合梯度卸载到 **对应的 CPU 分区**
4. CPU 上执行分片的 Adam 更新
5. 更新后的分片参数传回 GPU，执行 **AllGather** 得到完整参数

```
ZeRO-Offload + 多卡 (N_d = 4):

        GPU 0          GPU 1          GPU 2          GPU 3
       ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
       │完整   │       │完整   │       │完整   │       │完整   │
       │fp16   │       │fp16   │       │fp16   │       │fp16   │
       │参数   │       │参数   │       │参数   │       │参数   │
       └──┬───┘       └──┬───┘       └──┬───┘       └──┬───┘
          │  ReduceScatter (GPU 间)      │              │
          ↓              ↓              ↓              ↓
       ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
       │梯度   │       │梯度   │       │梯度   │       │梯度   │
       │分片1/4│       │分片2/4│       │分片3/4│       │分片4/4│
       └──┬───┘       └──┬───┘       └──┬───┘       └──┬───┘
          │ PCIe          │ PCIe         │ PCIe         │ PCIe
          ↓              ↓              ↓              ↓
       ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
  CPU: │Adam   │       │Adam   │       │Adam   │       │Adam   │
       │更新   │       │更新   │       │更新   │       │更新   │
       │分片1/4│       │分片2/4│       │分片3/4│       │分片4/4│
       └──┬───┘       └──┬───┘       └──┬───┘       └──┬───┘
          │ PCIe          │ PCIe         │ PCIe         │ PCIe
          ↓              ↓              ↓              ↓
       ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
       │参数   │       │参数   │       │参数   │       │参数   │
       │分片1/4│       │分片2/4│       │分片3/4│       │分片4/4│
       └──┬───┘       └──┬───┘       └──┬───┘       └──┬───┘
          │    AllGather (GPU 间)        │              │
          ↓              ↓              ↓              ↓
       ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
       │完整   │       │完整   │       │完整   │       │完整   │
       │fp16   │       │fp16   │       │fp16   │       │fp16   │
       │参数   │       │参数   │       │参数   │       │参数   │
       └──────┘       └──────┘       └──────┘       └──────┘

每张 GPU 显存: 2Ψ (仅完整 fp16 参数)
每张 GPU 对应的 CPU 内存: 12Ψ/N_d (分片的优化器状态)
```

### 通信量分析

| 通信类型 | 数据量 | 路径 |
|---------|--------|------|
| Reduce-Scatter 梯度 | \(\Psi\) | GPU ↔ GPU |
| 梯度卸载 | \(2\Psi / N_d\) | GPU → CPU |
| 参数更新回传 | \(2\Psi / N_d\) | CPU → GPU |
| AllGather 参数 | \(\Psi\) | GPU ↔ GPU |

GPU 间通信量仍为 \(2\Psi\)（与标准 DP 相同），PCIe 通信量为 \(4\Psi / N_d\)（随 GPU 数量线性减少）。

---

## 内存消耗的精确计算

让我们精确计算 ZeRO-Offload 在不同配置下的内存消耗。

### 单 GPU 场景

```
单 GPU + ZeRO-Offload 的内存布局:

GPU 显存:
┌───────────────────────────────────────────┐
│ fp16 参数:                 2Ψ bytes        │
│ fp16 梯度 (临时):          2Ψ bytes        │ ← 反向传播时产生,
│                                           │   传输到 CPU 后可释放
│ 激活值:                    变量            │
├───────────────────────────────────────────┤
│ 总计: ~2Ψ + 激活值                        │
│ (临时梯度可流式传输, 不需全部驻留)          │
└───────────────────────────────────────────┘

CPU 内存:
┌───────────────────────────────────────────┐
│ fp32 参数副本:             4Ψ bytes        │
│ fp32 Adam 动量:            4Ψ bytes        │
│ fp32 Adam 方差:            4Ψ bytes        │
│ fp32 梯度 (half2float后):  4Ψ bytes        │
├───────────────────────────────────────────┤
│ 总计: 16Ψ bytes                           │
└───────────────────────────────────────────┘
```

关键洞察：GPU 显存中只有 \(2\Psi\) 是参数的持久存储。梯度通过流式传输（边算边传），不需要全部同时驻留在 GPU 上。

### 可训练模型大小

以 V100 32 GB 为例，假设激活值检查点后需要约 **2-4 GB**：

| 配置 | GPU 可用显存 | 最大 fp16 参数量 | 最大模型参数 |
|------|------------|-----------------|-------------|
| 无 Offload | 32 GB | \(16\Psi \leq 32\text{GB}\) | ~2B |
| **ZeRO-Offload** | ~28 GB (留给激活) | \(2\Psi \leq 28\text{GB}\) | **~13B** |
| ZeRO-Offload + 激活卸载 | ~32 GB | \(2\Psi \leq 32\text{GB}\) | **~15B** |

提升了约 **6.5-7.5 倍**！

下面的代码可以精确计算不同场景下的内存分布：

```cpp-run title="ZeRO-Offload 内存分布计算器"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>

struct OffloadMemory {
    double gpu_params_gb;      // GPU: fp16 参数
    double gpu_grads_gb;       // GPU: fp16 梯度 (峰值)
    double gpu_activations_gb; // GPU: 激活值 (估算)
    double gpu_total_gb;
    double cpu_params_gb;      // CPU: fp32 参数副本
    double cpu_momentum_gb;    // CPU: Adam 动量
    double cpu_variance_gb;    // CPU: Adam 方差
    double cpu_grads_gb;       // CPU: fp32 梯度
    double cpu_total_gb;
};

OffloadMemory calc_offload(double psi_billions, int Nd, double act_gb) {
    double psi = psi_billions * 1e9;
    double psi_gb = psi * 1.0 / (1024.0 * 1024 * 1024);  // Ψ 个参数 = Ψ bytes

    OffloadMemory m;
    // GPU: fp16 参数 (完整) + fp16 梯度 (峰值, 可流式释放) + 激活值
    m.gpu_params_gb = 2.0 * psi_gb;  // fp16 = 2 bytes/param
    m.gpu_grads_gb  = 2.0 * psi_gb / Nd;  // ZeRO-2 分片后的梯度峰值
    m.gpu_activations_gb = act_gb;
    m.gpu_total_gb = m.gpu_params_gb + m.gpu_grads_gb + m.gpu_activations_gb;

    // CPU: fp32 参数 + 动量 + 方差 + 梯度, 各分片到 Nd
    m.cpu_params_gb   = 4.0 * psi_gb / Nd;
    m.cpu_momentum_gb = 4.0 * psi_gb / Nd;
    m.cpu_variance_gb = 4.0 * psi_gb / Nd;
    m.cpu_grads_gb    = 4.0 * psi_gb / Nd;
    m.cpu_total_gb = m.cpu_params_gb + m.cpu_momentum_gb
                   + m.cpu_variance_gb + m.cpu_grads_gb;
    return m;
}

struct StandardDPMemory {
    double gpu_total_gb;
};

StandardDPMemory calc_standard(double psi_billions, double act_gb) {
    double psi_gb = psi_billions * 1e9 / (1024.0 * 1024 * 1024);
    return { 16.0 * psi_gb + act_gb };
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << "        ZeRO-Offload 内存分布精确计算器\n";
    std::cout << "=============================================================\n\n";

    // ---- 场景 1: 单 GPU, 不同模型大小 ----
    std::cout << "场景 1: 单 GPU (V100 32GB), 不同模型大小\n";
    std::cout << "  激活值估算: 2 GB (使用激活检查点)\n\n";

    double act_gb = 2.0;
    double gpu_mem = 32.0;

    std::cout << std::setw(10) << "模型"
              << std::setw(14) << "标准DP GPU"
              << std::setw(14) << "Offload GPU"
              << std::setw(14) << "Offload CPU"
              << std::setw(10) << "可训?"  << "\n";
    std::cout << std::string(62, '-') << "\n";

    double models[] = {1.0, 2.0, 5.0, 7.5, 10.0, 13.0, 15.0};
    const char* names[] = {"1B","2B","5B","7.5B","10B","13B","15B"};

    for (int i = 0; i < 7; i++) {
        auto std_m = calc_standard(models[i], act_gb);
        auto off_m = calc_offload(models[i], 1, act_gb);

        std::string fit_std = std_m.gpu_total_gb <= gpu_mem ? "YES" : "NO";
        std::string fit_off = off_m.gpu_total_gb <= gpu_mem ? "YES" : "NO";
        std::string fit = fit_off;

        std::cout << std::setw(10) << names[i]
                  << std::setw(10) << std::fixed << std::setprecision(1)
                  << std_m.gpu_total_gb << " GB"
                  << std::setw(10) << off_m.gpu_total_gb << " GB"
                  << std::setw(10) << off_m.cpu_total_gb << " GB"
                  << std::setw(10) << (fit_off == "YES" ? "YES" : "NO")
                  << (fit_std == "YES" ? "" : (fit_off == "YES" ? " ← Offload!" : ""))
                  << "\n";
    }

    std::cout << "\n";

    // ---- 场景 2: 多 GPU + Offload ----
    std::cout << "=============================================================\n";
    std::cout << "场景 2: 10B 模型, 不同 GPU 数量 + ZeRO-Offload\n";
    std::cout << "=============================================================\n\n";

    double psi = 10.0;  // 10B

    std::cout << std::setw(6) << "N_d"
              << std::setw(14) << "GPU 内存"
              << std::setw(14) << "CPU/卡内存"
              << std::setw(16) << "PCIe/卡/步"
              << std::setw(14) << "GPU间通信" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (int Nd : {1, 2, 4, 8, 16, 32, 64}) {
        auto m = calc_offload(psi, Nd, act_gb);
        double psi_gb = psi * 1e9 / (1024.0 * 1024 * 1024);
        double pcie_gb = 4.0 * psi_gb / Nd;  // 梯度下+参数上, fp16
        double gpu_comm_gb = 2.0 * psi_gb;

        std::cout << std::setw(6) << Nd
                  << std::setw(10) << std::fixed << std::setprecision(1)
                  << m.gpu_total_gb << " GB"
                  << std::setw(10) << m.cpu_total_gb << " GB"
                  << std::setw(12) << pcie_gb << " GB"
                  << std::setw(10) << (Nd > 1 ? std::to_string((int)std::round(gpu_comm_gb)) + " GB" : "N/A")
                  << "\n";
    }

    std::cout << "\n";

    // ---- 场景 3: 详细内存组成 ----
    std::cout << "=============================================================\n";
    std::cout << "场景 3: 10B 模型, 单 GPU — 详细内存组成\n";
    std::cout << "=============================================================\n\n";

    auto m = calc_offload(10.0, 1, 2.0);

    std::cout << "  GPU 显存:\n";
    std::cout << "    fp16 参数:       " << std::fixed << std::setprecision(1)
              << m.gpu_params_gb << " GB\n";
    std::cout << "    fp16 梯度(峰值): " << m.gpu_grads_gb << " GB  "
              << "(流式传输时远小于此)\n";
    std::cout << "    激活值:          " << m.gpu_activations_gb << " GB\n";
    std::cout << "    ─────────────────────\n";
    std::cout << "    合计:            " << m.gpu_total_gb << " GB"
              << "  ← V100 32GB " << (m.gpu_total_gb <= 32 ? "可放下!" : "放不下!") << "\n\n";

    std::cout << "  CPU 内存:\n";
    std::cout << "    fp32 参数副本:   " << m.cpu_params_gb << " GB\n";
    std::cout << "    fp32 Adam 动量:  " << m.cpu_momentum_gb << " GB\n";
    std::cout << "    fp32 Adam 方差:  " << m.cpu_variance_gb << " GB\n";
    std::cout << "    fp32 梯度:       " << m.cpu_grads_gb << " GB\n";
    std::cout << "    ─────────────────────\n";
    std::cout << "    合计:            " << m.cpu_total_gb << " GB"
              << "  ← 典型工作站 256GB 轻松放下\n\n";

    auto std_m = calc_standard(10.0, 2.0);
    std::cout << "  对比标准 DP: " << std_m.gpu_total_gb << " GB GPU 内存\n";
    std::cout << "  节省: " << std::setprecision(1)
              << (1.0 - m.gpu_total_gb / std_m.gpu_total_gb) * 100
              << "% GPU 内存\n";

    // ---- 场景 4: PCIe 带宽分析 ----
    std::cout << "\n=============================================================\n";
    std::cout << "场景 4: PCIe 带宽瓶颈分析 (10B 模型, 单 GPU)\n";
    std::cout << "=============================================================\n\n";

    double psi_bytes = 10.0e9;
    double grad_transfer = 2.0 * psi_bytes;  // fp16 梯度
    double param_transfer = 2.0 * psi_bytes;  // fp16 更新后参数

    double pcie3_bw = 16.0e9;  // PCIe 3.0 x16: ~16 GB/s 实际
    double pcie4_bw = 32.0e9;  // PCIe 4.0 x16: ~32 GB/s 实际

    std::cout << "  数据传输量:\n";
    std::cout << "    梯度 GPU→CPU: " << std::setprecision(1)
              << grad_transfer / 1e9 << " GB (fp16)\n";
    std::cout << "    参数 CPU→GPU: " << param_transfer / 1e9 << " GB (fp16)\n";
    std::cout << "    合计: " << (grad_transfer + param_transfer) / 1e9 << " GB\n\n";

    std::cout << "  传输时间 (假设全带宽):\n";
    std::cout << "    PCIe 3.0 x16 (~16 GB/s): "
              << std::setprecision(2)
              << (grad_transfer + param_transfer) / pcie3_bw << " 秒\n";
    std::cout << "    PCIe 4.0 x16 (~32 GB/s): "
              << (grad_transfer + param_transfer) / pcie4_bw << " 秒\n\n";

    // V100 的 forward+backward 时间估算
    double flops_per_step = 6.0 * psi_bytes * 1024;  // ~6*Ψ*B FLOPs, B=1024 tokens
    double v100_tflops = 125e12;  // fp16
    double gpu_time = flops_per_step / v100_tflops;
    std::cout << "  GPU 计算时间 (粗估, bs=1024):\n";
    std::cout << "    Forward+Backward: " << std::setprecision(2)
              << gpu_time << " 秒\n\n";

    std::cout << "  结论: ";
    double pcie_time = (grad_transfer + param_transfer) / pcie3_bw;
    if (pcie_time < gpu_time) {
        std::cout << "PCIe 传输时间 < GPU 计算时间 → "
                  << "通信可被计算完全隐藏!\n";
    } else {
        std::cout << "PCIe 可能成为瓶颈, 需要更大 batch size 或 PCIe 4.0\n";
    }

    return 0;
}
```

---

## 吞吐量分析：开销在哪里？

ZeRO-Offload 的吞吐量取决于几个关键因素。让我们分析每一个。

### 时间分解

一步训练的总时间可以分解为：

$$
T_{\text{total}} = \max\!\big(T_{\text{GPU}},\; T_{\text{CPU}} + T_{\text{PCIe}}\big)
$$

其中：
- \(T_{\text{GPU}} = T_{\text{forward}} + T_{\text{backward}}\)：GPU 上的前向 + 反向传播时间
- \(T_{\text{CPU}} = T_{\text{Adam}}\)：CPU 上的 Adam 更新时间
- \(T_{\text{PCIe}} = T_{\text{grad\_transfer}} + T_{\text{param\_transfer}}\)：PCIe 传输时间

使用一步延迟后：

$$
T_{\text{total}} \approx T_{\text{GPU}} \quad \text{(当 } T_{\text{CPU}} + T_{\text{PCIe}} \leq T_{\text{GPU}} \text{)}
$$

也就是说，**当模型足够大时，CPU 和 PCIe 的开销完全被 GPU 计算隐藏**，吞吐量接近纯 GPU 方案。

### 什么时候 PCIe 成为瓶颈？

当模型较小而 batch size 也较小时，\(T_{\text{GPU}}\) 可能小于 \(T_{\text{CPU}} + T_{\text{PCIe}}\)，此时 PCIe 成为瓶颈。解决方法：

1. **增大 batch size**：更大的 batch → 更长的 GPU 计算时间 → 更容易隐藏 CPU 开销
2. **使用梯度累积**：多个 micro-batch 的梯度在 GPU 上累积后再一次性传输，减少 PCIe 传输频率
3. **PCIe 4.0/5.0**：更高的 PCIe 带宽直接减少传输时间

---

## 实验结果与关键发现

### 单 GPU 实验

**硬件**：单张 V100 32GB，CPU 内存 256GB

| 模型参数量 | 无 Offload 可训？ | ZeRO-Offload 可训？ | 吞吐量 (TFLOPS) |
|-----------|-----------------|-------------------|----------------|
| 1.4B | YES | YES | 40 |
| 4B | NO | **YES** | 37 |
| 7.5B | NO | **YES** | 35 |
| 10B | NO | **YES** | 30 |
| 13B | NO | **YES** | 28 |

关键数据点：
- **可训模型大小提升 ~10 倍**：从 1.4B 到 13B
- **吞吐量保持高水平**：即使是 13B 模型，仍有 28 TFLOPS（V100 fp16 峰值 125 TFLOPS 的 22%，考虑到内存受限训练场景，这是很好的利用率）
- **小模型几乎无损**：1.4B 模型的吞吐量与纯 GPU 方案几乎相同

### 多 GPU 实验

在 4 张 V100 上使用 ZeRO-Offload + ZeRO-2：

| 配置 | 最大可训模型 | 吞吐量 |
|------|------------|--------|
| 4× V100, 标准 DP | ~6B | 基线 |
| 4× V100, ZeRO-2 | ~14B | ~基线 |
| 4× V100, ZeRO-2 + Offload | **~50B** | ~0.9× 基线 |

多卡场景下，PCIe 通信量被 \(N_d\) 分摊，因此吞吐量损失更小。

### 与其他方法的对比

| 方法 | 最大模型 (单 V100) | 10B 模型吞吐量 |
|------|-------------------|---------------|
| L2L (Layer-to-Layer) | 10B | 0.3 TFLOPS |
| SwapAdvisor | 10B | 2.1 TFLOPS |
| **ZeRO-Offload** | **13B** | **30 TFLOPS** |

ZeRO-Offload 的吞吐量比 L2L 快 **100 倍**，比 SwapAdvisor 快 **14 倍**。这正是因为 ZeRO-Offload 的卸载策略是最优的——最少的通信，最高效的计算分配。

::: tip 为什么 L2L 这么慢？
L2L 将整个模型参数按层在 GPU ↔ CPU 之间搬运，意味着前向传播的每一层都要经历一次 PCIe 传输。对于 100 层的 Transformer，这就是 200 次 PCIe round-trip。而 ZeRO-Offload 只在梯度计算完成后做 **一次** 批量传输，通信效率天壤之别。
:::

### 收敛性验证

论文在 GPT-2 (1.5B) 上验证了使用一步延迟参数更新的收敛性：

- 训练 loss 曲线与无延迟版本 **几乎完全重合**
- 在 WikiText-103 上的困惑度（Perplexity）差异 < 0.1%
- 证明一步延迟在实践中不影响模型质量

---

## DeepSpeed 中的使用方式

在 DeepSpeed 中启用 ZeRO-Offload 非常简单，只需在配置中设置 `offload_optimizer` 和 `offload_param`：

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true    # 使用锁页内存加速 PCIe 传输
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### 关键配置项

| 配置项 | 含义 | 推荐值 |
|--------|------|--------|
| `offload_optimizer.device` | 优化器状态卸载目标 | `"cpu"` |
| `offload_param.device` | 参数卸载目标（ZeRO-3） | `"cpu"` |
| `pin_memory` | 使用锁页内存 | `true`（提升 PCIe 传输速度 ~2x） |
| `overlap_comm` | 通信-计算重叠 | `true` |

::: warning pin_memory 的作用
锁页内存（Pinned Memory）可以绕过操作系统的页面交换机制，允许 GPU 通过 DMA 直接访问 CPU 内存。这可以将 PCIe 传输速度从 ~8 GB/s 提升到 ~16 GB/s。强烈建议开启。
:::

---

## 总结与启示

### 核心贡献

1. **将卸载问题形式化为图优化**：不是拍脑袋决定"什么该卸载"，而是通过数据流图分析，严格推导出最优方案

2. **最优策略的直觉**：将 **内存大户**（优化器状态 12Ψ）和 **计算轻量操作**（Adam 逐元素更新）一起卸载到 CPU，而将 **计算密集操作**（forward/backward）和 **频繁读取的数据**（参数）保留在 GPU

3. **系统级优化**：CPU Adam SIMD 加速、一步延迟流水线、通信-计算重叠——每一个都是工程上的关键优化

4. **民主化大模型训练**：将 10B 级模型训练的硬件门槛从"数十张 GPU"降到"一张 GPU + 足够的 CPU 内存"

### 与 ZeRO 系列的关系

```
ZeRO 系列的演进:

ZeRO (SC 2020)
  └─ 消除多 GPU 间的模型状态冗余
  └─ 内存节省 ∝ GPU 数量 (N_d)
      │
      ↓
ZeRO-Offload (ATC 2021) ← 本文
  └─ 将模型状态卸载到 CPU 内存
  └─ 单 GPU 可训 10B+ 模型
  └─ 利用 CPU 内存扩展有效容量
      │
      ↓
ZeRO-Infinity (SC 2021)
  └─ 进一步卸载到 NVMe SSD
  └─ 理论上可训任意大模型
  └─ 利用整个存储层级: GPU → CPU → SSD
```

### 深层启示

ZeRO-Offload 体现了一个系统设计的核心哲学：**异构计算系统的关键不在于某一个组件有多强，而在于如何最优地利用每个组件的优势**。

| 组件 | 优势 | ZeRO-Offload 的利用 |
|------|------|-------------------|
| GPU | 计算强（125 TFLOPS fp16） | 前向传播 + 反向传播 |
| CPU | 内存大（256+ GB） | 存储优化器状态 + Adam 更新 |
| PCIe | 连接 GPU 和 CPU | 流式传输梯度和参数更新 |

每个组件都在做它最擅长的事，没有一个组件是闲置的——这就是 ZeRO-Offload 高效的根本原因。

::: tip 与 FlashAttention 的异曲同工
FlashAttention 利用了 GPU 内部的存储层次（SRAM vs HBM），ZeRO 利用了 GPU 之间的并行性，而 ZeRO-Offload 利用了 GPU-CPU 的异构存储层次。**三者的共同哲学：不要把所有数据都放在最贵的存储里，而是让数据在正确的时间出现在正确的位置。**
:::

---

## 参考文献

1. Ren, J., Rajbhandari, S., Aminabadi, R.Y., Ruwase, O., Yang, S., Zhang, M., Li, D., & He, Y. (2021). **ZeRO-Offload: Democratizing Billion-Scale Model Training**. USENIX ATC 2021. [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)

2. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

3. Rajbhandari, S., et al. (2021). **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning**. SC 2021. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)

4. Pudipeddi, B., Mesmakhosroshahi, M., Xi, J., & Bharadwaj, S. (2020). **Training Large Neural Networks with Constant Memory using a New Execution Order**. [arXiv:2002.05645](https://arxiv.org/abs/2002.05645)

5. Huang, C., Jin, S., & Li, J. (2020). **SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping**. ASPLOS 2020.
