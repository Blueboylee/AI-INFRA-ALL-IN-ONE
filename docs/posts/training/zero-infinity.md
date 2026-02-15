---
title: "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning"
date: 2026-02-15
---

# ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-15 &nbsp;·&nbsp; 分布式训练 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He
> - **机构**: Microsoft
> - **发表**: SC 2021 (The International Conference for High Performance Computing)
> - **链接**: [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)

## 一句话总结

ZeRO-Infinity 将 ZeRO 的内存卸载从 CPU 进一步扩展到 **NVMe SSD**，构建了一套完整的 **GPU → CPU → NVMe 三级存储卸载引擎**，辅以带宽中心化分片（Bandwidth-Centric Partitioning）和重叠通信机制，使得 **有限的 GPU 集群可以训练理论上无穷大的模型**——论文在 512 张 V100 上成功训练了 **32 万亿（32T）参数** 的模型。

---

## Introduction：为什么需要 ZeRO-Infinity？

### 1. ZeRO 系列的演进与瓶颈

在前两篇论文精读中，我们看到了 ZeRO 系列的逐步演进：

| 方案 | 发表 | 核心思想 | 限制 |
|------|------|---------|------|
| [ZeRO](./deepspeed-zero) | SC 2020 | GPU 间分片模型状态 | 需要大量 GPU |
| [ZeRO-Offload](./zero-offload) | ATC 2021 | 卸载到 CPU 内存 | 受限于 CPU 内存容量 |
| **ZeRO-Infinity** | **SC 2021** | **卸载到 NVMe SSD** | **？** |

ZeRO-Offload 使单卡训练 13B 模型成为可能，但 CPU 内存仍然是有限的（通常 256 GB - 1 TB）。对于 100B+ 的模型，即使是 CPU 内存也不够用：

```
100B 模型的优化器状态内存需求:

fp32 参数副本:   100B × 4 bytes = 400 GB
fp32 Adam 动量:  100B × 4 bytes = 400 GB
fp32 Adam 方差:  100B × 4 bytes = 400 GB
────────────────────────────────────────
合计:                            1.2 TB

→ 单节点 CPU 内存可能不够!
→ 更大的模型 (1T+) 完全放不下!
```

但如果往下看一层存储层级——**NVMe SSD** 的容量通常是 **数 TB 到数十 TB**，价格低廉且易于扩展。

### 2. 完整的存储层级

现代深度学习服务器拥有一个完整的存储层级：

```
存储层级金字塔 (单节点, 8×V100 DGX-2):

            ┌──────────┐
            │ GPU HBM  │  8 × 32 GB = 256 GB
            │  ~900    │  带宽: 900 GB/s/卡
            │  GB/s/卡 │
            ├──────────┤
            │ CPU DRAM │  1.5 TB
            │  ~100    │  带宽: 100 GB/s
            │  GB/s    │
            ├──────────┤
            │ NVMe SSD │  16 × 1.6 TB = 25.6 TB
            │  ~50     │  带宽: 50 GB/s (理论)
            │  GB/s    │  (16 块 NVMe RAID-0)
            └──────────┘

容量:    256 GB  →  1.5 TB  →  25.6 TB
带宽:    900 GB/s  →  100 GB/s  →  50 GB/s
倍率:       6x 容量    ~17x 容量
```

**ZeRO-Infinity 的核心洞察**：NVMe SSD 提供了 **比 CPU 内存大一个数量级的容量**，带宽虽然低于 CPU 但仍然可观（多盘 RAID-0 可达 50+ GB/s）。如果能高效利用这一层，**可训练模型的参数量将不再受限于任何单一存储层级**。

### 3. 朴素 NVMe 卸载的问题

直接把 ZeRO-Offload 的策略照搬到 NVMe 上不可行，因为：

**问题 1：带宽鸿沟**

| 通路 | 带宽 | 延迟 |
|------|------|------|
| GPU HBM 内部 | 900 GB/s | ~ns |
| GPU ↔ CPU (PCIe 3.0) | 32 GB/s | ~μs |
| CPU ↔ NVMe (PCIe) | 3-6 GB/s/盘 | ~μs-ms |

从 GPU 到 NVMe 的有效带宽比 GPU 内部低 **100-300 倍**。如果不做通信优化，SSD I/O 将严重拖慢训练。

**问题 2：ZeRO-Offload 的设计假设不再成立**

ZeRO-Offload 假设 CPU 有足够带宽完成 Adam 更新和数据传输。但如果数据来自 NVMe，CPU 需要先从 SSD 读取数据，执行计算，再写回 SSD——这个流水线更长、更容易成为瓶颈。

**问题 3：多层卸载的调度复杂度**

GPU → CPU → NVMe 三级存储意味着数据可以在任意两层之间移动。如何决定"什么数据放在哪一层、什么时候移动"是一个组合优化问题。

### 4. ZeRO-Infinity 的核心创新

论文提出了五个关键技术来解决上述问题：

1. **Infinity Offload Engine**：统一的 GPU-CPU-NVMe 三级卸载引擎，支持将模型状态（参数、梯度、优化器状态）和激活值卸载到任意存储层级

2. **带宽中心化分片（Bandwidth-Centric Partitioning）**：确保每个数据分片的大小足够大，从而最大化 NVMe 的顺序读写带宽（避免小块随机 I/O）

3. **重叠中心化设计（Overlap-Centric Design）**：通过预取（Prefetch）和异步写回，将 NVMe I/O 与 GPU 计算和网络通信重叠

4. **数据感知的内存管理（Data-Aware Memory Management）**：动态管理 CPU 内存作为 GPU 和 NVMe 之间的缓存层

5. **高性能 NVMe 访问库**：基于 Linux AIO 的深度优化 I/O 库，充分利用多盘并行

::: tip 命名含义
"Infinity" 意味着可训练模型的参数量理论上没有上限——只要你能加更多的 NVMe SSD。存储层级的扩展性使得 GPU 内存不再是硬性瓶颈。
:::

---

## 技术核心 1：Infinity Offload Engine

Infinity Offload Engine 是 ZeRO-Infinity 的核心组件，负责统一管理 GPU、CPU 和 NVMe 三级存储。

### 设计哲学

传统的卸载方案将每种模型状态绑定到特定设备（如 ZeRO-Offload 将优化器状态固定在 CPU）。ZeRO-Infinity 采用了更灵活的设计——**任何模型状态都可以放在任何存储层级**，由系统根据容量和带宽自动决定最优分配。

```
Infinity Offload Engine 的三级存储管理:

                    ┌──────────────────┐
                    │   Offload Engine  │
                    │   (调度中心)       │
                    └────┬─────┬────┬──┘
                         │     │    │
            ┌────────────┘     │    └────────────┐
            ↓                  ↓                 ↓
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   GPU HBM    │  │   CPU DRAM   │  │   NVMe SSD   │
    │              │  │              │  │              │
    │ • 当前层参数  │  │ • 参数缓存    │  │ • 优化器状态  │
    │   (临时)     │  │ • 梯度缓冲    │  │ • 参数分片    │
    │ • 当前激活值  │  │ • CPU计算空间 │  │ • 梯度分片    │
    │ • 梯度(临时)  │  │              │  │ • 激活检查点  │
    │              │  │              │  │              │
    │ 容量: 小      │  │ 容量: 中      │  │ 容量: 大      │
    │ 带宽: 极高    │  │ 带宽: 中      │  │ 带宽: 较低    │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### 各类模型状态的卸载策略

| 模型状态 | 默认存储 | 原因 |
|---------|---------|------|
| **参数** (fp16) | NVMe → CPU (缓存) → GPU (预取) | 前向/反向时需要完整参数，逐层预取到 GPU |
| **梯度** (fp16) | GPU → CPU → NVMe | 反向时产生在 GPU，聚合后卸载 |
| **优化器状态** (fp32) | NVMe | 最大内存消耗者，仅在更新时读取 |
| **激活值检查点** | NVMe (可选) | 最大的残余内存消耗者 |

### 参数的完整生命周期

以 ZeRO-3 + NVMe 卸载为例，一个参数分片在一步训练中的完整旅程：

```
参数分片的生命周期 (以第 l 层为例):

时间 ────────────────────────────────────────────────────→

1. 休眠态 (大部分时间)
   NVMe: [参数分片 l/N_d] (持久存储)

2. 前向传播预取
   NVMe ──读取──→ CPU ──预取──→ GPU
   [分片 l/N_d]    [缓存]      [AllGather → 完整第 l 层参数]

3. 前向计算
   GPU: 使用完整第 l 层参数执行前向传播
        计算完毕 → 释放非本卡分片

4. 反向传播预取 (再次)
   NVMe ──读取──→ CPU ──预取──→ GPU
   [分片 l/N_d]    [缓存]      [AllGather → 完整第 l 层参数]

5. 反向计算
   GPU: 使用完整参数计算梯度
        梯度 ReduceScatter → 每卡只保留 1/N_d 梯度
        释放参数

6. 梯度卸载
   GPU ──传输──→ CPU ──写回──→ NVMe
   [梯度分片]    [缓冲]      [持久存储]

7. 参数更新 (异步, CPU 上执行)
   NVMe ──读取──→ CPU: Adam(参数, 梯度, 动量, 方差)
   CPU ──写回──→ NVMe: [更新后的参数分片]
```

---

## 技术核心 2：带宽中心化分片

### 问题：小块 I/O 的带宽灾难

NVMe SSD 的带宽特性与 HBM/DRAM 有本质不同：

| 特性 | GPU HBM | CPU DRAM | NVMe SSD |
|------|---------|----------|----------|
| 随机读写 | 几乎无损 | 轻微损失 | **严重损失** |
| 最优块大小 | 任意 | 任意 | **≥ 512 KB** |
| 小块惩罚 | 无 | 小 | **10-100 倍降速** |

如果按照 ZeRO-3 的朴素分片方式（按参数平均分配到 \(N_d\) 个进程），当参数量较小或 \(N_d\) 很大时，每个分片可能只有几 KB——这会导致 NVMe 的带宽利用率暴跌。

### 解决方案：确保最小分片大小

ZeRO-Infinity 的带宽中心化分片策略保证 **每个 NVMe I/O 请求的数据量足够大**：

$$
\text{分片大小} = \max\!\left(\frac{\text{参数量}}{N_d},\; B_{\min}\right)
$$

其中 \(B_{\min}\) 是保证 NVMe 达到接近峰值带宽的最小块大小（通常 ≥ 512 KB）。

如果某个参数张量太小无法满足最小块大小要求，则多个小张量 **打包（Pack）** 成一个大块后再进行 I/O。

```
带宽中心化分片:

朴素分片 (可能产生小块):
┌─────────────────────────────────────────┐
│ 参数张量 A (100MB)                       │
│ [分片0: 1.5MB][分片1: 1.5MB]...[分片63]  │ ← OK, 每块 1.5MB > 512KB
└─────────────────────────────────────────┘

┌───────────────────┐
│ 参数张量 B (1MB)   │
│ [分0:16KB][分1:16KB]...[分63:16KB] │ ← 灾难! 16KB 远小于最优块大小
└───────────────────┘

带宽中心化分片 (打包小张量):
┌──────────────────────────────────────────────┐
│ 打包块: [张量B + 张量C + 张量D + ...]           │
│ 总大小: 2MB → [分片0: 32KB]...[分片63: 32KB]   │ ← 仍然太小?
│ → 进一步打包或减少分片数                        │
└──────────────────────────────────────────────┘

→ 保证每个 NVMe I/O ≥ B_min, 带宽利用率接近峰值
```

### AllGather 中的带宽优化

ZeRO-3 的 AllGather 操作需要从各个节点收集参数分片。在 ZeRO-Infinity 中，这涉及 NVMe → CPU → GPU → 网络的多级数据搬运。论文确保每一级的数据块都足够大：

- **NVMe 读取**：块大小 ≥ 512 KB
- **CPU → GPU 传输**：攒够一个批次再通过 PCIe 传输
- **GPU 间 AllGather**：使用大消息的集合通信

---

## 技术核心 3：重叠中心化设计

NVMe I/O 的绝对速度不如 HBM 和 DRAM，因此 **隐藏 I/O 延迟** 是 ZeRO-Infinity 性能的关键。

### 四路重叠

ZeRO-Infinity 同时维护四条并行的数据流水线：

```
四路重叠时间线 (前向传播阶段):

GPU 计算:    [Forward 层l  ][Forward 层l+1][Forward 层l+2]
GPU↔GPU通信: [AllGather l+1][AllGather l+2][AllGather l+3]  ← 网络预取
PCIe 传输:   [CPU→GPU  l+1 ][CPU→GPU  l+2 ][CPU→GPU  l+3 ]
NVMe 读取:   [SSD→CPU  l+2 ][SSD→CPU  l+3 ][SSD→CPU  l+4 ]  ← NVMe 预取
─────────────────────────────────────────────────────────→ 时间

关键: 四条流水线交错执行, 每条都在为未来的计算做准备!

当 GPU 在计算第 l 层时:
  - 网络正在 AllGather 第 l+1 层的参数
  - PCIe 正在将第 l+1 层的本地分片从 CPU 传到 GPU
  - NVMe 正在将第 l+2 层的分片从 SSD 读到 CPU

如果流水线足够深, NVMe 的 I/O 延迟完全被隐藏!
```

### 预取深度的选择

预取的层数（深度）决定了 I/O 隐藏的效果：

- **深度 = 0**：无预取，等数据到了才开始算 → NVMe I/O 完全暴露
- **深度 = 1**：预取下一层 → 部分隐藏
- **深度 = 2-3**：预取未来 2-3 层 → 大部分 I/O 被隐藏
- **深度过大**：CPU 内存中缓存太多层 → 可能 OOM

论文的经验法则：**预取深度 2-3 层** 在大多数配置下能达到最优平衡。

### 反向传播的异步写回

反向传播产生的梯度需要写回 NVMe。类似地，写回操作也与计算重叠：

```
反向传播的异步写回:

GPU 计算:    [Backward 层l  ][Backward 层l-1][Backward 层l-2]
梯度ReduceScatter:[RS 层l     ][RS 层l-1     ][RS 层l-2      ]
PCIe 传输:   [GPU→CPU 梯度l  ][GPU→CPU 梯度l-1]
NVMe 写入:                    [CPU→SSD 梯度l   ][CPU→SSD 梯度l-1]
─────────────────────────────────────────────────────────→ 时间

梯度的 "流式写回": 算完一层就开始传, 不等全部算完
```

---

## 技术核心 4：CPU 内存作为缓存

在三级存储体系中，CPU 内存扮演着 **NVMe 和 GPU 之间的缓存层** 角色。

### 内存管理策略

```
CPU 内存的缓存角色:

┌──────────────────────────────────────────┐
│              CPU DRAM (缓存层)             │
│                                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ 预取缓冲 │  │ 写回缓冲 │  │ 计算空间 │  │
│  │          │  │          │  │          │  │
│  │ NVMe→GPU │  │ GPU→NVMe │  │ Adam更新 │  │
│  │ 的中转站  │  │ 的中转站  │  │ 工作区   │  │
│  └─────────┘  └─────────┘  └─────────┘  │
│                                          │
│  管理策略:                                │
│  1. 预取缓冲: 固定大小, 容纳 2-3 层参数    │
│  2. 写回缓冲: 双缓冲, 一个在写SSD,         │
│     一个在接收新梯度                       │
│  3. 计算空间: Adam 更新所需的临时空间       │
└──────────────────────────────────────────┘
```

### 智能淘汰策略

当 CPU 内存缓存满时，需要决定 **淘汰哪些数据**。ZeRO-Infinity 使用基于训练计算图的确定性淘汰：

- **参数**：前向传播时按层顺序访问，用完立即淘汰
- **梯度**：写入 NVMe 后立即淘汰
- **优化器状态**：更新完一个分片后淘汰，加载下一个分片

这与传统的 LRU/LFU 缓存策略不同——训练过程的访问模式是 **完全可预测的**，因此可以做到零缓存未命中。

---

## 技术核心 5：高性能 NVMe 访问

### Linux AIO 与深度队列

传统的文件 I/O（read/write 系统调用）是同步的——发出请求后线程被阻塞直到 I/O 完成。这无法充分利用 NVMe 的并行性。

ZeRO-Infinity 使用 **Linux Asynchronous I/O (AIO)**：

- 一次提交多个 I/O 请求（深度队列）
- NVMe 控制器可以并行处理这些请求
- 应用线程在 I/O 期间继续执行其他工作

```
同步 I/O vs 异步 I/O (AIO):

同步 I/O:
  线程: [发请求1][等待...][完成1][发请求2][等待...][完成2][发请求3][等待...]
  NVMe:          [处理1]                [处理2]                [处理3]
  利用率: ~30%

异步 I/O (AIO):
  线程: [发请求1,2,3,4...][继续工作...][收割完成的请求]
  NVMe: [处理1][处理2][处理3][处理4]...
  利用率: ~90%+
```

### 多盘并行

现代 DGX 节点配备 **16 块 NVMe SSD**。通过 RAID-0 或软件层面的并行 I/O，可以聚合带宽：

| 配置 | 顺序读带宽 | 顺序写带宽 |
|------|-----------|-----------|
| 单盘 NVMe | ~3 GB/s | ~2 GB/s |
| 4 盘并行 | ~12 GB/s | ~8 GB/s |
| 16 盘并行 | ~48 GB/s | ~32 GB/s |

16 盘并行的 48 GB/s 读带宽已经接近 PCIe 3.0 x16 的 32 GB/s——**NVMe 聚合带宽可能超过 PCIe 到 GPU 的带宽**，此时 PCIe 反而成为瓶颈。

---

## 内存与带宽的定量分析

### 可训练模型大小

设集群有 \(N\) 个节点，每节点 8 张 GPU，CPU 内存 \(M_{\text{cpu}}\)，NVMe 总容量 \(M_{\text{nvme}}\)：

$$
\Psi_{\max} = \frac{M_{\text{nvme}} \times N}{16} \text{ 参数}
$$

（分母 16 是因为混合精度 Adam 需要每参数 16 字节。）

| 集群规模 | NVMe/节点 | 最大可训模型 |
|---------|-----------|------------|
| 1 节点 (8 GPU) | 25.6 TB | **1.6 T** |
| 8 节点 (64 GPU) | 25.6 TB × 8 | **12.8 T** |
| 64 节点 (512 GPU) | 25.6 TB × 64 | **102 T** |

相比之下：
- ZeRO-3（仅 GPU）在 512 张 V100 上最多约 2T 参数
- ZeRO-Offload（+ CPU）在 512 张 V100 + CPU 上最多约 10T 参数
- ZeRO-Infinity（+ NVMe）在同样硬件上理论可达 **100T+ 参数**

### 带宽瓶颈分析

训练一步的关键在于数据能否及时从 NVMe 到达 GPU。以下分析单节点的瓶颈：

```
数据流瓶颈分析 (单节点, 8×V100, 16×NVMe):

NVMe → CPU: 16 盘 × 3 GB/s = 48 GB/s  ─┐
                                         ├→ 瓶颈: min(48, 32) = 32 GB/s
CPU → GPU:  PCIe 3.0 × 8 GPU = 32 GB/s ─┘
                                         (PCIe 是瓶颈!)

每步训练需传输的数据 (Ψ 参数模型):
  前向 AllGather 参数:  2Ψ bytes (fp16)
  反向 AllGather 参数:  2Ψ bytes (fp16)
  反向 ReduceScatter:  跨 GPU, 不经过 NVMe
  梯度写回 NVMe:       2Ψ/N_d bytes
  优化器状态读写:       16Ψ/N_d bytes (每张卡的分片)

总 NVMe I/O ≈ (2×2Ψ + 18Ψ/N_d) bytes (与流水线深度有关)
```

下面的代码可以计算不同配置下的理论性能上限：

```cpp-run title="ZeRO-Infinity 存储层级与带宽分析"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>

struct ClusterConfig {
    int num_nodes;
    int gpus_per_node;
    double gpu_mem_gb;       // 每 GPU 显存
    double cpu_mem_gb;       // 每节点 CPU 内存
    int nvme_per_node;       // 每节点 NVMe 盘数
    double nvme_cap_tb;      // 每盘容量 TB
    double nvme_read_gbps;   // 每盘读带宽 GB/s
    double nvme_write_gbps;  // 每盘写带宽 GB/s
    double pcie_bw_gbps;     // 单 GPU PCIe 带宽 GB/s
    double gpu_tflops;       // 单 GPU fp16 TFLOPS
};

int main() {
    // DGX-2 配置
    ClusterConfig dgx2 = {
        .num_nodes = 1,
        .gpus_per_node = 8,
        .gpu_mem_gb = 32,
        .cpu_mem_gb = 1536,   // 1.5 TB
        .nvme_per_node = 16,
        .nvme_cap_tb = 1.6,
        .nvme_read_gbps = 3.0,
        .nvme_write_gbps = 2.0,
        .pcie_bw_gbps = 16.0,  // PCIe 3.0 x16 实际
        .gpu_tflops = 125.0,
    };

    std::cout << "=============================================================\n";
    std::cout << "    ZeRO-Infinity 存储层级与可训练模型大小分析\n";
    std::cout << "=============================================================\n\n";

    // ---- 1. 不同方案的最大可训模型 ----
    std::cout << "1. 不同方案的最大可训练模型参数量\n";
    std::cout << "   (混合精度 + Adam, 每参数 16 字节模型状态)\n\n";

    std::cout << std::setw(8) << "节点数"
              << std::setw(10) << "GPU数"
              << std::setw(14) << "ZeRO-3"
              << std::setw(14) << "+CPU卸载"
              << std::setw(14) << "+NVMe卸载" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int nodes : {1, 4, 8, 32, 64}) {
        int total_gpus = nodes * dgx2.gpus_per_node;
        // ZeRO-3: 仅 GPU 内存
        double zero3_bytes = (double)total_gpus * dgx2.gpu_mem_gb * 1e9;
        double zero3_params = zero3_bytes / 16.0;
        // ZeRO-Offload: GPU + CPU
        double offload_bytes = zero3_bytes + (double)nodes * dgx2.cpu_mem_gb * 1e9;
        double offload_params = offload_bytes / 16.0;
        // ZeRO-Infinity: GPU + CPU + NVMe
        double nvme_bytes = (double)nodes * dgx2.nvme_per_node * dgx2.nvme_cap_tb * 1e12;
        double infinity_bytes = offload_bytes + nvme_bytes;
        double infinity_params = infinity_bytes / 16.0;

        auto fmt = [](double params) -> std::string {
            if (params >= 1e12) return std::to_string((int)(params/1e12)) + "T";
            if (params >= 1e9)  return std::to_string((int)(params/1e9)) + "B";
            return std::to_string((int)(params/1e6)) + "M";
        };

        std::cout << std::setw(8) << nodes
                  << std::setw(10) << total_gpus
                  << std::setw(14) << fmt(zero3_params)
                  << std::setw(14) << fmt(offload_params)
                  << std::setw(14) << fmt(infinity_params) << "\n";
    }

    std::cout << "\n";

    // ---- 2. 单节点带宽瓶颈分析 ----
    std::cout << "=============================================================\n";
    std::cout << "2. 单节点 (8×V100) 带宽瓶颈分析\n";
    std::cout << "=============================================================\n\n";

    double nvme_agg_read = dgx2.nvme_per_node * dgx2.nvme_read_gbps;
    double nvme_agg_write = dgx2.nvme_per_node * dgx2.nvme_write_gbps;
    double pcie_agg = dgx2.gpus_per_node * dgx2.pcie_bw_gbps;

    std::cout << "  NVMe 聚合读带宽:  " << std::fixed << std::setprecision(0)
              << nvme_agg_read << " GB/s (" << dgx2.nvme_per_node << " 盘)\n";
    std::cout << "  NVMe 聚合写带宽:  " << nvme_agg_write << " GB/s\n";
    std::cout << "  PCIe 聚合带宽:    " << pcie_agg << " GB/s ("
              << dgx2.gpus_per_node << " GPU)\n";
    std::cout << "  有效瓶颈带宽:     "
              << std::min({nvme_agg_read, pcie_agg}) << " GB/s\n\n";

    // ---- 3. 不同模型大小的 I/O 时间估算 ----
    std::cout << "=============================================================\n";
    std::cout << "3. 不同模型大小的 I/O 与计算时间估算 (单节点)\n";
    std::cout << "=============================================================\n\n";

    double eff_bw = std::min(nvme_agg_read, pcie_agg);  // 有效带宽
    int Nd = dgx2.gpus_per_node;

    std::cout << std::setw(10) << "模型"
              << std::setw(16) << "参数传输(2次)"
              << std::setw(14) << "I/O 时间"
              << std::setw(14) << "GPU计算时间"
              << std::setw(12) << "瓶颈?" << "\n";
    std::cout << std::string(66, '-') << "\n";

    struct Model { const char* name; double params; int seq_len; };
    Model models[] = {
        {"10B",   10e9,  1024},
        {"50B",   50e9,  1024},
        {"100B",  100e9, 1024},
        {"500B",  500e9, 1024},
        {"1T",    1e12,  1024},
    };

    for (auto& m : models) {
        // 参数传输: 前向+反向各一次 AllGather, 每次 2Ψ bytes (fp16)
        double param_transfer_bytes = 4.0 * m.params;  // 2×2Ψ
        double io_time = param_transfer_bytes / (eff_bw * 1e9);

        // GPU 计算时间: ~6ΨB FLOPs (B = micro_batch_size × seq_len)
        double micro_bs = 4;
        double flops = 6.0 * m.params * micro_bs * m.seq_len;
        double gpu_time = flops / (dgx2.gpu_tflops * 1e12 * Nd);

        const char* bottleneck = io_time > gpu_time ? "I/O" : "计算";

        std::cout << std::setw(10) << m.name
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << param_transfer_bytes / 1e9 << " GB"
                  << std::setw(11) << std::setprecision(2) << io_time << " s"
                  << std::setw(11) << gpu_time << " s"
                  << std::setw(12) << bottleneck << "\n";
    }

    std::cout << "\n  说明: I/O 时间假设无重叠; 实际通过预取可隐藏大部分 I/O\n";
    std::cout << "        GPU 计算时间基于 micro_bs=4, seq_len=1024 粗估\n\n";

    // ---- 4. 预取的 I/O 隐藏效果 ----
    std::cout << "=============================================================\n";
    std::cout << "4. 预取深度对 I/O 隐藏的影响 (100B 模型, 单节点)\n";
    std::cout << "=============================================================\n\n";

    double psi = 100e9;
    int num_layers = 96;  // 100B 模型约 96 层
    double per_layer_params = psi / num_layers;
    double per_layer_transfer = 4.0 * per_layer_params;  // 前向+反向
    double per_layer_io = per_layer_transfer / (eff_bw * 1e9);
    double per_layer_compute = (6.0 * per_layer_params * 4 * 1024) /
                               (dgx2.gpu_tflops * 1e12 * Nd);

    std::cout << "  每层参数: " << std::setprecision(1) << per_layer_params / 1e9
              << "B (" << per_layer_transfer / 1e9 << " GB 传输)\n";
    std::cout << "  每层 I/O 时间: " << std::setprecision(3) << per_layer_io << " s\n";
    std::cout << "  每层计算时间: " << per_layer_compute << " s\n\n";

    for (int depth : {0, 1, 2, 3, 4}) {
        double exposed_io;
        if (depth == 0) {
            exposed_io = per_layer_io * num_layers;
        } else {
            // 预取可以隐藏 min(depth × per_layer_compute, per_layer_io) 的 I/O
            double hidden = std::min((double)depth * per_layer_compute, per_layer_io);
            double exposed_per_layer = std::max(0.0, per_layer_io - hidden);
            exposed_io = exposed_per_layer * num_layers;
        }
        double total_compute = per_layer_compute * num_layers;
        double total = total_compute + exposed_io;
        double efficiency = total_compute / total * 100;

        std::cout << "  预取深度 " << depth << ": 暴露I/O="
                  << std::setprecision(2) << exposed_io << "s"
                  << "  总时间=" << total << "s"
                  << "  GPU效率=" << std::setprecision(1) << efficiency << "%\n";
    }

    return 0;
}
```

---

## ZeRO-Infinity 的完整架构

将所有技术组合在一起，ZeRO-Infinity 的完整训练架构如下：

```
ZeRO-Infinity 完整架构 (单节点视角):

                    ┌────────────────────────────────┐
                    │        GPU 0 ~ GPU 7            │
                    │                                │
                    │  ┌─────────────────────────┐   │
                    │  │ 当前层完整参数 (AllGather)│   │
                    │  │ Forward / Backward 计算  │   │
                    │  │ 当前层梯度 (ReduceScatter)│   │
                    │  └────────┬───────┬────────┘   │
                    │           │       │            │
                    │    AllGather  ReduceScatter     │
                    │    (GPU↔GPU)  (GPU↔GPU)        │
                    └────────┬──────────┬────────────┘
                             │ PCIe     │ PCIe
                    ┌────────┴──────────┴────────────┐
                    │          CPU DRAM               │
                    │                                │
                    │  ┌──────┐ ┌──────┐ ┌────────┐ │
                    │  │预取   │ │写回   │ │Adam    │ │
                    │  │缓冲   │ │缓冲   │ │更新    │ │
                    │  │(参数) │ │(梯度) │ │(分片)  │ │
                    │  └──┬───┘ └──┬───┘ └──┬─────┘ │
                    └─────┼────────┼────────┼───────┘
                          │ AIO    │ AIO    │ AIO
                    ┌─────┴────────┴────────┴───────┐
                    │        NVMe SSD Array           │
                    │  ┌────┐┌────┐┌────┐┌────┐     │
                    │  │参数││梯度││动量 ││方差 │ ... │
                    │  │分片││分片││分片 ││分片 │     │
                    │  └────┘└────┘└────┘└────┘     │
                    │  (所有模型状态的持久存储)        │
                    └───────────────────────────────┘
```

---

## 实验结果与关键发现

### 实验设置

- **硬件**：最多 512 张 V100 32GB（64 个 DGX-2 节点）
- **每节点**：8 GPU + 1.5 TB CPU 内存 + 16× NVMe SSD（25.6 TB）
- **模型**：基于 Transformer 架构，参数量从 1B 到 32T

### 关键结果 1：可训练模型大小

| 方案 | 512 GPU 最大可训模型 | 备注 |
|------|-------------------|------|
| 3D 并行 (Megatron + Pipeline) | ~2T | 纯 GPU 内存 |
| ZeRO-3 | ~2T | 纯 GPU 内存 |
| ZeRO-Offload | ~10T | + CPU 内存 |
| **ZeRO-Infinity** | **>32T** | + NVMe SSD |

论文在 512 张 V100 上成功运行了 **32 万亿参数** 的模型——这在当时是一个里程碑式的数字。

### 关键结果 2：吞吐量

| 模型大小 | 方案 | 吞吐量 (TFLOPS/GPU) | 峰值效率 |
|---------|------|---------------------|---------|
| 1T | ZeRO-Infinity (512 GPU) | 39 | 31% |
| 10T | ZeRO-Infinity (512 GPU) | 36 | 29% |
| 32T | ZeRO-Infinity (512 GPU) | 33 | 26% |

即使是 32T 这样极端的模型，仍然保持了约 **26% 的 GPU 峰值效率**。这意味着 NVMe I/O 的开销被成功控制在了合理范围内。

### 关键结果 3：与 3D 并行的对比

在 GPU 内存能容纳的模型大小范围内（~1T），ZeRO-Infinity 与 3D 并行的性能可比：

- **1T 模型，512 GPU**：ZeRO-Infinity 达到 39 TFLOPS/GPU，3D 并行约 36 TFLOPS/GPU
- ZeRO-Infinity 不需要复杂的流水线并行配置（pipeline stage 划分、micro-batch 数量调优等），**易用性更好**

### 关键结果 4：单节点大模型

| 配置 | 最大可训模型 |
|------|------------|
| 1 节点, 8 GPU, 无卸载 | ~14B |
| 1 节点, 8 GPU, CPU 卸载 | ~100B |
| 1 节点, 8 GPU, **NVMe 卸载** | **~1.6T** |

单节点从 14B 到 1.6T——**提升超过 100 倍**！

::: warning 吞吐量与模型大小的权衡
ZeRO-Infinity 使"能训"和"训得快"解耦。对于超大模型，它保证了"能训"，但吞吐量会随着模型大小增加而下降（因为 NVMe I/O 越来越难被完全隐藏）。在实际生产中，需要在模型大小和训练速度之间找到平衡。
:::

---

## DeepSpeed 中的使用方式

ZeRO-Infinity 在 DeepSpeed 中通过 ZeRO Stage 3 + offload 配置启用：

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",                    # 优化器状态卸载到 NVMe
            "nvme_path": "/local_nvme",          # NVMe 挂载路径
            "pin_memory": true,
            "buffer_count": 5,                   # 异步 I/O 缓冲区数量
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",                    # 参数也卸载到 NVMe
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,                  # 每个缓冲区大小 (bytes)
            "max_in_cpu": 1e9                    # CPU 内存中最多缓存参数量
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,                   # 分片粒度
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "fp16": {
        "enabled": true
    },
    "aio": {
        "block_size": 1048576,                   # 1MB I/O 块大小
        "queue_depth": 8,                        # I/O 队列深度
        "thread_count": 1,                       # I/O 线程数
        "single_submit": false,                  # 批量提交
        "overlap_events": true                   # 允许 I/O 重叠
    }
}
```

### 关键 NVMe 配置项

| 配置项 | 含义 | 调优建议 |
|--------|------|---------|
| `nvme_path` | NVMe 挂载路径 | 使用最快的 NVMe 盘，建议 RAID-0 |
| `buffer_count` | 异步 I/O 缓冲区数量 | 3-5，更多 = 更好的重叠但更多 CPU 内存 |
| `buffer_size` | 每个缓冲区大小 | ≥ 100MB 保证 NVMe 带宽 |
| `max_in_cpu` | CPU 缓存的最大参数量 | 根据 CPU 内存调整 |
| `aio.block_size` | AIO 块大小 | 1MB 通常是好的起点 |
| `aio.queue_depth` | AIO 队列深度 | 8-32，更深 = 更好的 NVMe 利用率 |

### NVMe 准备

```bash
# 1. 检查 NVMe 设备
lsblk | grep nvme

# 2. 确保挂载点可用且有足够空间
df -h /local_nvme

# 3. 安装 DeepSpeed 的 AIO 依赖
pip install deepspeed
ds_report  # 检查 aio 是否可用

# 4. 如果 aio 不可用, 安装 libaio
sudo apt install libaio-dev  # Ubuntu
sudo yum install libaio-devel  # CentOS

# 5. 重新编译 DeepSpeed ops
DS_BUILD_AIO=1 pip install deepspeed --force-reinstall
```

---

## ZeRO 系列全景回顾

三篇论文构成了一个完整的 **内存优化演进路线**：

```
ZeRO 系列演进全景:

  可训练参数量
  (log scale)
       │
  100T ┤                                          ★ ZeRO-Infinity
       │                                        ╱   (GPU+CPU+NVMe)
   10T ┤                                      ╱
       │                                    ╱
    1T ┤                            ★─────╱── ZeRO-Offload
       │                          ╱              (GPU+CPU)
  100B ┤                        ╱
       │                      ╱
   10B ┤              ★─────╱── ZeRO-3
       │            ╱              (GPU 间分片)
    1B ┤    ★─────╱── 标准 DP
       │          (每卡完整副本)
       └───┬──────┬──────┬──────┬──────┬──→ GPU 数量
           1      8     64    512    4096
```

### 核心思想的一致性

| 维度 | ZeRO | ZeRO-Offload | ZeRO-Infinity |
|------|------|-------------|---------------|
| **消除冗余** | GPU 间 | GPU-CPU 间 | GPU-CPU-NVMe 间 |
| **分片对象** | 模型状态 | 模型状态 | 模型状态 + 激活值 |
| **存储利用** | GPU HBM | GPU + CPU DRAM | GPU + CPU + NVMe |
| **通信优化** | AllGather/RS | PCIe 流水线 | AIO + 预取 + 重叠 |
| **设计哲学** | 不重复存储 | 用对的设备做对的事 | 利用一切可用存储 |

三篇论文的哲学一脉相承：**让数据在正确的时间、正确的位置出现**。

### 实际选择指南

```
如何选择 ZeRO 配置?

模型放得下单卡?
  ├── YES → 标准 DP 或 ZeRO-1 (最快)
  └── NO
       ├── 有多张 GPU?
       │    ├── YES → ZeRO-2 或 ZeRO-3
       │    │         (模型状态超 GPU 总内存? → ZeRO-3)
       │    └── NO (单卡)
       │         └── ZeRO-Offload (CPU 内存够? → YES → 搞定)
       │                          (CPU 内存不够? → ZeRO-Infinity)
       └── 多卡但模型极大 (>1T)?
            └── ZeRO-Infinity + 多卡
                (NVMe 容量足够? → YES → 搞定)
                (NVMe 不够? → 加节点!)
```

---

## 总结与启示

### ZeRO-Infinity 的核心贡献

1. **打破 GPU 内存墙**：将可训练模型的参数量从"受限于 GPU 内存"提升到"受限于 NVMe SSD 容量"——后者便宜且易扩展

2. **系统级的存储层级利用**：GPU HBM（速度）→ CPU DRAM（缓存）→ NVMe SSD（容量），每一层都被充分利用

3. **带宽中心化设计**：确保每一级存储都以接近峰值的带宽工作，避免小块 I/O 的性能陷阱

4. **四路重叠的流水线**：NVMe I/O、PCIe 传输、GPU 间通信、GPU 计算四条流水线并行，最大化硬件利用率

### 深层启示

ZeRO-Infinity 展示了一个强大的系统设计原则：**当单一资源成为瓶颈时，不要执着于优化它，而是引入下一级资源并设计高效的层级协作机制**。

这与计算机系统中无处不在的 **缓存层级** 思想完全一致：
- CPU：L1 → L2 → L3 → DRAM → SSD → HDD
- GPU：寄存器 → 共享内存/L1 → L2 → HBM
- ZeRO-Infinity：GPU HBM → CPU DRAM → NVMe SSD

::: tip 从 FlashAttention 到 ZeRO-Infinity 的统一视角
纵观这个系列的所有论文，核心都是同一件事——**内存层级的最优利用**：

- **FlashAttention**：优化 GPU 内部 SRAM ↔ HBM 的数据搬运
- **ZeRO**：优化 GPU ↔ GPU 之间的冗余数据
- **ZeRO-Offload**：优化 GPU ↔ CPU 之间的数据放置
- **ZeRO-Infinity**：优化 GPU ↔ CPU ↔ NVMe 的完整存储层级

**AI 系统的性能优化，本质上是一场与存储层级的博弈。**
:::

---

## 参考文献

1. Rajbhandari, S., Ruwase, O., Rasley, J., Smith, S., & He, Y. (2021). **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning**. SC 2021. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)

2. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

3. Ren, J., et al. (2021). **ZeRO-Offload: Democratizing Billion-Scale Model Training**. USENIX ATC 2021. [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)

4. Rasley, J., et al. (2020). **DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters**. KDD 2020.

5. Narayanan, D., et al. (2021). **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**. SC 2021. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)

6. Smith, S., et al. (2022). **Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model**. [arXiv:2201.11990](https://arxiv.org/abs/2201.11990)
