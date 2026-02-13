---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
date: 2026-02-14
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-14 &nbsp;·&nbsp; 推理引擎 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
> - **机构**: Stanford University, University at Buffalo
> - **发表**: NeurIPS 2022
> - **链接**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

## 一句话总结

FlashAttention 提出了一种 **IO 感知（IO-Aware）** 的精确注意力算法，通过 **分块计算（Tiling）** 和 **核函数融合（Kernel Fusion）** 避免在 GPU 高带宽内存（HBM）中实体化巨大的注意力矩阵，将注意力计算的内存复杂度从 \(O(N^2)\) 降至 \(O(N)\)，同时在墙钟时间上比标准注意力快 **2-4 倍**。

---

## Introduction：为什么需要 FlashAttention？

### 1. Transformer 的核心瓶颈：Self-Attention

Transformer 已成为 NLP、CV、语音等领域的基础架构。然而，自注意力机制（Self-Attention）在序列长度 \(N\) 上具有 **\(O(N^2)\) 的时间和空间复杂度**，这成为 Transformer 处理长序列的根本瓶颈。

**标准注意力的计算流程**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

在标准实现中，这个过程需要：
1. 计算 \(S = QK^\top\)，生成一个 \(N \times N\) 的注意力分数矩阵
2. 对 \(S\) 施加 softmax，得到 \(P = \text{softmax}(S)\)
3. 计算输出 \(O = PV\)

**问题在于**：中间矩阵 \(S\) 和 \(P\) 都是 \(N \times N\) 的，当序列长度 \(N = 8192\) 时，仅这两个矩阵就需要约 **512 MB** 显存（fp32）。这不仅占用大量内存，更关键的是需要频繁在 GPU 的不同存储层级之间搬运数据。

### 2. GPU 内存层次结构：被忽视的瓶颈

论文指出，现有工作几乎都以 **FLOP 数（浮点运算次数）** 作为优化目标，但这忽略了一个关键事实：**现代 GPU 的计算速度远超内存读写速度**。

以 A100 GPU 为例：

| 存储层级 | 容量 | 带宽 |
|---------|------|------|
| **SRAM**（片上缓存，每个 SM） | ~20 MB（总计） | ~19 TB/s |
| **HBM**（高带宽内存） | 40/80 GB | ~2 TB/s |

可以看到，SRAM 的带宽是 HBM 的近 **10 倍**，但容量却小得多。标准注意力实现的问题在于：

```
标准 Attention 的内存访问模式：

  HBM (慢)                    SRAM (快)                  计算单元
┌──────────┐   读取 Q,K    ┌──────────┐   矩阵乘    ┌──────────┐
│  Q, K, V │ ──────────►  │          │ ──────────► │  S=QK^T  │
│          │              │          │             │          │
│  S (N²)  │ ◄────────── │  结果 S  │ ◄────────── │          │
│          │   写回 S     │          │             │          │
│          │   读取 S     │          │   softmax   │          │
│  P (N²)  │ ◄────────── │  结果 P  │ ◄────────── │  P=sm(S) │
│          │   写回 P     │          │             │          │
│          │   读取 P,V   │          │   矩阵乘    │          │
│  O       │ ◄────────── │  结果 O  │ ◄────────── │  O=PV    │
└──────────┘   写回 O     └──────────┘             └──────────┘

问题：S 和 P 各 N² 大小，反复在 HBM ↔ SRAM 之间搬运！
```

**核心洞察**：注意力计算其实是 **内存带宽受限（Memory-bound）** 的操作，而非计算受限。瓶颈不是"算不过来"，而是"数据搬不过来"。

### 3. 现有的近似注意力方法的困境

为了突破 \(O(N^2)\) 的限制，研究社区提出了大量的 **近似注意力（Approximate Attention）** 方法，包括：

- **稀疏注意力（Sparse Attention）**：只计算部分位置对的注意力（如 Longformer、BigBird）
- **低秩近似（Low-rank Approximation）**：用低秩矩阵近似完整注意力矩阵（如 Linformer、Performer）
- **线性注意力（Linear Attention）**：通过核方法将 softmax 近似为可分解形式，实现线性复杂度

然而，论文指出这些方法存在两个共性问题：

1. **精度损失**：近似方法在长序列上经常出现质量退化，尤其是在需要精确建模长距离依赖的任务中
2. **墙钟时间并未真正加速**：虽然 FLOP 数降低了，但由于这些方法往往引入了更多的内存访问开销（如稀疏索引、额外的矩阵变换），在实际 GPU 上跑起来并没有标准注意力快。论文中的实验表明，很多近似方法在序列长度达到 512-2048 之前甚至比标准注意力更慢

::: warning 一个反直觉的事实
减少 FLOP ≠ 减少运行时间。在 GPU 上，如果一个算法 FLOP 更少但内存访问更多，它完全可能比 FLOP 更多但内存访问模式更优的算法更慢。这就是 FlashAttention 的出发点。
:::

### 4. FlashAttention 的核心思路

FlashAttention 不走近似路线，而是从 **IO 复杂度（IO Complexity）** 的角度重新审视标准注意力，通过优化内存访问模式来实现加速，同时保持结果的 **数值精确性**。

核心策略包括两点：

**（1）分块计算（Tiling）**：将 Q、K、V 分成小块，每次只加载一小块到 SRAM 中进行计算，避免实体化完整的 \(N \times N\) 注意力矩阵。

**（2）在线 Softmax（Online Softmax）**：传统 softmax 需要先遍历整行求最大值和求和，再做归一化——这要求整行数据同时在内存中。FlashAttention 采用了 Milakov & Gimelshein (2018) 提出的在线 softmax 技巧，在分块流式处理的过程中 **增量更新** softmax 统计量（running max 和 running sum），无需回头修正。

```
FlashAttention 的内存访问模式：

  HBM (慢)                    SRAM (快)                  计算单元
┌──────────┐   读取 Q块,   ┌──────────┐   一次性     ┌──────────┐
│  Q, K, V │ ──────────►  │ Q块,K块, │ ──────────► │ 分块计算  │
│          │   K块, V块    │  V块     │             │ S块→P块→ │
│          │              │          │   融合计算   │  O块累加  │
│  O       │ ◄────────── │  O块     │ ◄────────── │          │
└──────────┘   只写最终O   └──────────┘             └──────────┘

优势：
  - 中间矩阵 S、P 从不写回 HBM
  - HBM 读写量从 O(N²) 降至 O(N²d²M⁻¹)（M 为 SRAM 大小）
  - 结果与标准注意力完全一致（精确算法）
```

### 5. 论文的主要贡献

论文总结了以下关键贡献：

1. **FlashAttention 算法**：一种 IO 感知的精确注意力实现，通过 Tiling 和在线 Softmax 将 HBM 访问量减少为 \(O(N^2 d^2 M^{-1})\)，其中 \(d\) 是头维度、\(M\) 是 SRAM 大小。论文还证明了在所有精确注意力算法中，这是 **HBM 访问次数的渐近最优下界**

2. **Kernel 融合的扩展**：将 FlashAttention 扩展到支持常用的注意力变体，包括 **带 Mask 的注意力**（如因果掩码）和 **Dropout**，这些操作都在同一个 CUDA Kernel 中完成，避免了额外的内存读写

3. **长序列建模的实际收益**：基于 FlashAttention 的高效实现，论文展示了在多个基准任务上的显著提升：
   - GPT-2 训练速度提升至标准 HuggingFace 实现的 **3 倍**
   - 支持的序列长度从 1K-2K 拓展到 **4K-16K**，使 Transformer 首次在长文档分类（如 MIMIC-III）和长序列生成任务上取得 SOTA 表现
   - Path-X（16K 序列长度的合成任务）上首次达到 **超过随机水平的准确率**

4. **IO 复杂度的理论分析**：论文给出了精确注意力的 HBM 访问下界证明，并分析了常见近似/稀疏注意力的 IO 复杂度，为后续注意力优化研究提供了理论基础

::: tip 为什么叫"Flash"？
Flash 一语双关：既指速度极快（如闪存 Flash Memory），也暗示了算法的核心思想——像闪存一样 **感知和优化 IO 访问模式**，让数据在正确的存储层级被高效处理。
:::

---

## 标准注意力实现：Algorithm 0

在深入 FlashAttention 之前，我们必须先彻底理解"标准实现到底做了什么、代价几何"。论文将其命名为 **Algorithm 0**，作为后续优化的基线。

### 形式化定义

给定输入矩阵 \(Q, K, V \in \mathbb{R}^{N \times d}\)（\(N\) 为序列长度，\(d\) 为头维度），标准注意力的计算分为三步：

$$
S = QK^\top \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = PV \in \mathbb{R}^{N \times d}
$$

其中 softmax 按行应用。

### Algorithm 0 的执行流程

论文给出的标准实现伪代码如下：

```
Algorithm 0: Standard Attention Implementation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: Q, K, V ∈ R^{N×d}，存储在 HBM 中

Step 1: 从 HBM 分块加载 Q, K，计算 S = QK^T，将 S 写回 HBM
Step 2: 从 HBM 读取 S，计算 P = softmax(S)，将 P 写回 HBM
Step 3: 从 HBM 分块加载 P 和 V，计算 O = PV，将 O 写回 HBM

返回: O
```

### 逐步拆解 HBM 访问量

我们来精确计算每一步对 HBM 的读写量：

| 步骤 | 操作 | HBM 读取 | HBM 写入 | 说明 |
|------|------|---------|---------|------|
| **Step 1** | \(S = QK^\top\) | \(Q\): \(Nd\) + \(K\): \(Nd\) | \(S\): \(N^2\) | 矩阵乘法，结果必须写回 HBM |
| **Step 2** | \(P = \text{softmax}(S)\) | \(S\): \(N^2\) | \(P\): \(N^2\) | 逐元素操作，但 S 太大放不进 SRAM |
| **Step 3** | \(O = PV\) | \(P\): \(N^2\) + \(V\): \(Nd\) | \(O\): \(Nd\) | 矩阵乘法 |
| **合计** | — | \(3Nd + 2N^2\) | \(Nd + 2N^2\) | 总 HBM 访问 = \(4Nd + 4N^2\) |

**关键观察**：当 \(N \gg d\) 时（GPT-2: \(N = 1024, d = 64\)），HBM 访问量被 \(4N^2\) 项主导，即 **\(O(N^2)\)**。

而中间矩阵 \(S\) 和 \(P\) 各占 \(N^2\) 的空间——它们的唯一作用是作为"中转站"：被写入 HBM 后马上又被读出来。这种"写了就读、读了就扔"的模式是巨大的浪费。

### 为什么标准实现是 Memory-bound？

我们可以算一笔帐，用 **算术强度（Arithmetic Intensity）** 来判断操作是计算密集还是内存密集：

$$
\text{算术强度} = \frac{\text{FLOPs}}{\text{Bytes accessed}}
$$

对于 Step 2（softmax）：
- **FLOPs**：\(O(N^2)\)（每个元素做 exp、加法、除法）
- **HBM 访问**：读 \(N^2\) + 写 \(N^2\) = \(2N^2\) 个元素 = \(8N^2\) bytes（fp32）
- **算术强度** ≈ \(O(1)\)

A100 的算术强度平衡点约为 \(\frac{312 \text{ TFLOPS}}{2 \text{ TB/s}} = 156\) FLOP/byte。softmax 的算术强度远低于这个值，因此它是一个典型的 **内存带宽受限** 操作——GPU 的计算单元大部分时间在"等数据"。

::: warning 即使矩阵乘法也受影响
Step 1 和 Step 3 的矩阵乘法本身是计算密集型操作，但因为 \(S\) 和 \(P\) 必须经过 HBM 这个"中转站"，整个流水线的实际吞吐被 Step 2 的内存瓶颈拖慢了。Fuse 不掉 softmax，前后的 matmul 也快不起来。
:::

### Masking 和 Dropout 雪上加霜

论文特别指出，实际应用中注意力矩阵上还要叠加额外的逐元素操作：

- **Masking**：因果注意力需要对 \(S\) 施加下三角掩码，即 \(S_{ij} = -\infty\) for \(j > i\)
- **Dropout**：训练时对 \(P\) 施加随机置零

每增加一个逐元素操作，就多一轮 \(N^2\) 的 HBM 读写。虽然社区已尝试将 masking 和 softmax 融合进同一个 kernel，但 **只要 \(S\) 和 \(P\) 还存在于 HBM 中**，根本问题就没有解决。

### 动手试一试

下面的 C++ 代码完整模拟了 Algorithm 0 的三步流程，并 **精确追踪每一步的 HBM 读写量**。运行后你可以直观看到：中间矩阵 \(S\) 和 \(P\) 如何主导了内存访问开销。

```cpp-run title="Algorithm 0: 标准注意力实现 — 逐步追踪 HBM 读写"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

// ===== Algorithm 0: Standard Attention Implementation =====
// 完整模拟三步流程, 精确追踪每一步的 HBM 读写量

struct HBMTracker {
    long long reads = 0, writes = 0;
    void read(long long n)  { reads  += n; }
    void write(long long n) { writes += n; }
    long long total() const { return reads + writes; }
    void report(const char* step) const {
        std::cout << "    " << step
                  << ": 读=" << reads << ", 写=" << writes
                  << ", 小计=" << total() << " 个元素\n";
    }
};

using Mat = std::vector<std::vector<float>>;
using Vec = std::vector<float>;

Mat make_mat(int r, int c, float v = 0) {
    return Mat(r, Vec(c, v));
}

void print_mat(const char* name, const Mat& M, int max_rows = 4) {
    int rows = std::min((int)M.size(), max_rows);
    int cols = M[0].size();
    for (int i = 0; i < rows; i++) {
        std::cout << "  " << name << "[" << i << "] = [";
        for (int j = 0; j < cols; j++)
            std::cout << (j ? ", " : "") << std::fixed
                      << std::setprecision(4) << M[i][j];
        std::cout << "]\n";
    }
    if ((int)M.size() > max_rows)
        std::cout << "  ... (共 " << M.size() << " 行)\n";
}

int main() {
    const int N = 8;   // 序列长度
    const int d = 4;   // 头维度

    std::cout << "====== Algorithm 0: Standard Attention ======\n";
    std::cout << "参数: N=" << N << " (序列长度), d=" << d << " (头维度)\n";
    std::cout << "矩阵大小: Q,K,V ∈ R^{" << N << "x" << d << "}\n\n";

    // ---------- 初始化 Q, K, V (存储在 HBM 中) ----------
    Mat Q = make_mat(N, d), K = make_mat(N, d), V = make_mat(N, d);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++) {
            Q[i][j] = sinf(i * 0.5f + j * 0.3f);
            K[i][j] = cosf(i * 0.4f + j * 0.2f);
            V[i][j] = sinf(i * 0.3f + j * 0.5f);
        }

    // 用三个独立的 tracker 分别统计每一步
    HBMTracker hbm1, hbm2, hbm3;

    // ========== Step 1: S = QK^T ==========
    // 从 HBM 分块加载 Q, K，计算 S，将 S 写回 HBM
    std::cout << "--- Step 1: S = QK^T / sqrt(d) ---\n";
    Mat S = make_mat(N, N);

    hbm1.read(N * d);   // 从 HBM 读取 Q (N×d 个元素)
    hbm1.read(N * d);   // 从 HBM 读取 K (N×d 个元素)

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < d; k++)
                s += Q[i][k] * K[j][k];
            S[i][j] = s / sqrtf((float)d);
        }

    hbm1.write(N * N);  // 将 S (N×N 个元素) 写回 HBM
    hbm1.report("Step 1");
    std::cout << "  S 矩阵大小 = " << N << "x" << N
              << " = " << N * N << " 个元素\n";
    print_mat("S", S);
    std::cout << "\n";

    // ========== Step 2: P = softmax(S) ==========
    // 从 HBM 读取 S，逐行做 softmax，将 P 写回 HBM
    std::cout << "--- Step 2: P = softmax(S) ---\n";
    Mat P = make_mat(N, N);

    hbm2.read(N * N);   // 从 HBM 读取 S (N×N 个元素)

    for (int i = 0; i < N; i++) {
        // 数值稳定 softmax: 先减最大值
        float row_max = *std::max_element(S[i].begin(), S[i].end());
        float row_sum = 0;
        for (int j = 0; j < N; j++) {
            P[i][j] = expf(S[i][j] - row_max);
            row_sum += P[i][j];
        }
        for (int j = 0; j < N; j++)
            P[i][j] /= row_sum;
    }

    hbm2.write(N * N);  // 将 P (N×N 个元素) 写回 HBM
    hbm2.report("Step 2");
    std::cout << "  P 矩阵大小 = " << N << "x" << N
              << " = " << N * N << " 个元素\n";
    // 验证每行概率和为 1
    std::cout << "  每行概率和: ";
    for (int i = 0; i < std::min(N, 4); i++) {
        float s = 0;
        for (int j = 0; j < N; j++) s += P[i][j];
        std::cout << std::fixed << std::setprecision(4) << s << " ";
    }
    std::cout << " (均为 1.0)\n";
    print_mat("P", P);
    std::cout << "\n";

    // ========== Step 3: O = PV ==========
    // 从 HBM 分块加载 P 和 V，计算 O，将 O 写回 HBM
    std::cout << "--- Step 3: O = PV ---\n";
    Mat O = make_mat(N, d);

    hbm3.read(N * N);   // 从 HBM 读取 P (N×N 个元素)
    hbm3.read(N * d);   // 从 HBM 读取 V (N×d 个元素)

    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++) {
            float s = 0;
            for (int k = 0; k < N; k++)
                s += P[i][k] * V[k][j];
            O[i][j] = s;
        }

    hbm3.write(N * d);  // 将 O (N×d 个元素) 写回 HBM
    hbm3.report("Step 3");
    print_mat("O", O);
    std::cout << "\n";

    // ========== HBM 总量分析 ==========
    long long total_reads  = hbm1.reads  + hbm2.reads  + hbm3.reads;
    long long total_writes = hbm1.writes + hbm2.writes + hbm3.writes;
    long long grand_total  = total_reads + total_writes;

    std::cout << "========== HBM 访问量汇总 ==========\n";
    std::cout << "  总读取: " << total_reads << " 个元素"
              << " (其中 N² 相关: " << (hbm2.reads + hbm3.reads - N * d)
              << ")\n";
    std::cout << "  总写入: " << total_writes << " 个元素"
              << " (其中 N² 相关: " << (hbm1.writes + hbm2.writes)
              << ")\n";
    std::cout << "  总 HBM 访问: " << grand_total << " 个元素\n\n";

    // 理论值验证
    long long theory_Nd   = 4LL * N * d;   // 3Nd 读 + Nd 写
    long long theory_N2   = 4LL * N * N;   // 2N² 读 + 2N² 写
    long long theory_total = theory_Nd + theory_N2;

    std::cout << "  理论公式: 4Nd + 4N² = "
              << theory_Nd << " + " << theory_N2
              << " = " << theory_total << "\n";
    std::cout << "  实际统计: " << grand_total << "\n";
    std::cout << "  匹配: " << (grand_total == theory_total ? "PASS" : "FAIL")
              << "\n\n";

    // 占比分析: N² 项 vs Nd 项
    double pct_N2 = 100.0 * theory_N2 / theory_total;
    std::cout << "  N² 项占比: " << std::fixed << std::setprecision(1)
              << pct_N2 << "% ← S 和 P 主导了 HBM 访问!\n";
    std::cout << "  Nd 项占比: " << std::fixed << std::setprecision(1)
              << (100.0 - pct_N2) << "%\n\n";

    // 展示 N 增长时 N² 如何主导
    std::cout << "========== N 增长时的 HBM 访问量 ==========\n";
    std::cout << std::setw(8) << "N"
              << std::setw(12) << "4Nd"
              << std::setw(12) << "4N²"
              << std::setw(14) << "总计"
              << std::setw(10) << "N²占比" << "\n";

    for (int n : {8, 32, 128, 512, 1024, 4096}) {
        long long nd_part = 4LL * n * d;
        long long n2_part = 4LL * n * n;
        long long tot     = nd_part + n2_part;
        double pct        = 100.0 * n2_part / tot;
        std::cout << std::setw(8) << n
                  << std::setw(12) << nd_part
                  << std::setw(12) << n2_part
                  << std::setw(14) << tot
                  << std::setw(8) << std::fixed << std::setprecision(1)
                  << pct << "%\n";
    }

    std::cout << "\n[Key Insight]\n"
              << "  当 N >> d 时, HBM 访问量被 4N² 项完全主导.\n"
              << "  中间矩阵 S 和 P 各 N×N, 写入后立刻被读出, 然后再也不用\n"
              << "  —— 它们是昂贵的\"一次性中转站\".\n"
              << "  这正是 FlashAttention 要解决的核心问题!\n";

    return 0;
}
```

### 小结

标准注意力实现的核心问题可以用一句话概括：

> **中间矩阵 \(S\) 和 \(P\) 是 \(N^2\) 大小的"一次性中转站"，它们被写入 HBM 后马上被读出，读出后再也不用——但它们主导了整个算法的内存访问开销。**

FlashAttention 的解法正是从这里出发：如果我们能在 SRAM 中分块完成 \(S \to P \to O\) 的全部计算，让 \(S\) 和 \(P\) **永远不触碰 HBM**，那么 HBM 的访问量就可以从 \(O(N^2)\) 大幅下降。但这要求我们解决一个技术难题：**softmax 是全局操作，如何在只看到一小块数据的情况下正确计算？** 这就是下一节 FlashAttention 算法的核心挑战。

---

## FlashAttention 算法：Tiling + Online Softmax

上一节我们看到，标准注意力的瓶颈在于中间矩阵 \(S, P \in \mathbb{R}^{N \times N}\) 必须在 HBM 中实体化。FlashAttention 的目标很明确：**在不实体化 \(S\) 和 \(P\) 的前提下，精确计算 \(\text{softmax}(QK^\top)V\)**。

这里我们只讨论前向传播（Forward Pass），反向传播的细节见论文 Appendix B。

### 核心挑战：Softmax 是"全局操作"

分块计算矩阵乘法很简单——把大矩阵切成小块逐块相乘再累加就行。但 softmax 不同：

$$
\text{softmax}(S_{i,:}) = \frac{e^{S_{i,j}}}{\sum_{k=1}^{N} e^{S_{i,k}}}
$$

分母是对 **整行** 求和，这意味着要计算第 \(i\) 行的 softmax，你需要知道 \(S\) 的整行 \(N\) 个值。如果我们把 \(K\) 分成多个块，每次只算出 \(S\) 的一部分列，怎么做 softmax？

FlashAttention 的答案是：**在线 Softmax（Online Softmax）**——一种增量式的分块 softmax 算法。

### 分块 Softmax 的数学推导

#### 单块的 Softmax 统计量

对于一个向量 \(x \in \mathbb{R}^{B}\)，数值稳定的 softmax 需要三个统计量：

$$
m(x) := \max_i x_i, \quad f(x) := \begin{bmatrix} e^{x_1 - m(x)} \\ \vdots \\ e^{x_B - m(x)} \end{bmatrix}, \quad \ell(x) := \sum_i f(x)_i
$$

最终 \(\text{softmax}(x) = \frac{f(x)}{\ell(x)}\)。

#### 两块合并：关键递推公式

现在假设我们有两个分块 \(x^{(1)}, x^{(2)} \in \mathbb{R}^{B}\)，要计算拼接向量 \(x = [x^{(1)}, x^{(2)}] \in \mathbb{R}^{2B}\) 的 softmax。核心递推公式为：

$$
m(x) = \max\!\big(m(x^{(1)}),\; m(x^{(2)})\big)
$$

$$
f(x) = \Big[\; e^{m(x^{(1)}) - m(x)} \cdot f(x^{(1)}), \quad e^{m(x^{(2)}) - m(x)} \cdot f(x^{(2)}) \;\Big]
$$

$$
\ell(x) = e^{m(x^{(1)}) - m(x)} \cdot \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \cdot \ell(x^{(2)})
$$

**直觉**：当新块的最大值更大时，旧块的 exp 值需要"缩小"（乘以 \(e^{m_{\text{old}} - m_{\text{new}}} < 1\)）；当旧块最大值更大时，新块被缩小。这个缩放因子保证了数值的正确性。

::: tip 关键洞察
这个递推公式意味着：我们可以 **逐块处理 K 的列方向**，每处理一块就更新 \(m\) 和 \(\ell\)，最终得到的 softmax 结果与一次性处理整行 **完全一致**——没有任何近似！
:::

### 从 Softmax 到 Attention 输出的增量更新

分块 softmax 解决了归一化的问题，但注意力的最终输出是 \(O = PV\)，即 softmax 的结果还要和 \(V\) 做矩阵乘法。我们不能先算完所有 softmax 再乘 \(V\)——那又回到了实体化 \(P\) 的老路。

FlashAttention 的做法是 **边算 softmax 边累加 \(O\)**。当处理第 \(j\) 个 K/V 块时：

1. 计算当前块的局部注意力分数 \(\tilde{S}_{ij} = Q_i K_j^\top\)
2. 计算局部统计量 \(\tilde{m}_{ij}, \tilde{P}_{ij}, \tilde{\ell}_{ij}\)
3. 更新全局统计量 \(m_i^{\text{new}}, \ell_i^{\text{new}}\)
4. **修正并累加输出**：

$$
O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1} \Big( \underbrace{\text{diag}(\ell_i) \cdot e^{m_i - m_i^{\text{new}}} \cdot O_i}_{\text{旧输出修正}} + \underbrace{e^{\tilde{m}_{ij} - m_i^{\text{new}}} \cdot \tilde{P}_{ij} V_j}_{\text{新块贡献}} \Big)
$$

这个公式的含义是：
- **旧输出修正**：之前累加的 \(O_i\) 是基于旧的最大值 \(m_i\) 计算的，现在最大值更新为 \(m_i^{\text{new}}\)，需要乘以修正因子 \(e^{m_i - m_i^{\text{new}}}\)
- **新块贡献**：当前块的 softmax 值乘以 \(V_j\)，同样调整到新的最大值尺度
- **重新归一化**：除以新的 \(\ell_i^{\text{new}}\) 确保概率和为 1

### Algorithm 1：FlashAttention 完整算法

将上述思路系统化，就得到了论文中的 Algorithm 1：

```
Algorithm 1: FlashAttention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: Q, K, V ∈ R^{N×d} (在 HBM 中), 片上 SRAM 大小 M

1: 设置块大小 Bc = ⌈M/(4d)⌉, Br = min(⌈M/(4d)⌉, d)
2: 在 HBM 中初始化 O = 0^{N×d}, ℓ = 0^N, m = (-∞)^N

3: 将 Q 分为 Tr = ⌈N/Br⌉ 个块: Q₁, ..., Q_Tr  (每块 Br×d)
   将 K, V 分为 Tc = ⌈N/Bc⌉ 个块: K₁,...,K_Tc 和 V₁,...,V_Tc  (每块 Bc×d)
4: 将 O 分为 Tr 个块, ℓ 和 m 也分为 Tr 个块

5: for j = 1 to Tc do                    ← 外层循环: 遍历 K/V 块
6:     从 HBM 加载 Kⱼ, Vⱼ 到 SRAM
7:     for i = 1 to Tr do                ← 内层循环: 遍历 Q 块
8:         从 HBM 加载 Qᵢ, Oᵢ, ℓᵢ, mᵢ 到 SRAM
9:         在 SRAM 中计算 Sᵢⱼ = Qᵢ Kⱼᵀ ∈ R^{Br×Bc}
10:        在 SRAM 中计算:
             m̃ᵢⱼ = rowmax(Sᵢⱼ)          ∈ R^{Br}
             P̃ᵢⱼ = exp(Sᵢⱼ - m̃ᵢⱼ)      ∈ R^{Br×Bc}
             ℓ̃ᵢⱼ = rowsum(P̃ᵢⱼ)          ∈ R^{Br}
11:        在 SRAM 中计算:
             mᵢⁿᵉʷ = max(mᵢ, m̃ᵢⱼ)
             ℓᵢⁿᵉʷ = e^{mᵢ - mᵢⁿᵉʷ} · ℓᵢ + e^{m̃ᵢⱼ - mᵢⁿᵉʷ} · ℓ̃ᵢⱼ
12:        写回 HBM:
             Oᵢ ← diag(ℓᵢⁿᵉʷ)⁻¹ (diag(ℓᵢ)·e^{mᵢ-mᵢⁿᵉʷ}·Oᵢ + e^{m̃ᵢⱼ-mᵢⁿᵉʷ}·P̃ᵢⱼ·Vⱼ)
13:        写回 HBM: ℓᵢ ← ℓᵢⁿᵉʷ, mᵢ ← mᵢⁿᵉʷ
14:    end for
15: end for
16: 返回 O
```

### 循环结构的直觉理解

算法采用 **双层循环**，可以用下面的图来理解：

```
        K₁    K₂    K₃    K₄     (Tc 个 K/V 块, 外层循环 j)
      ┌─────┬─────┬─────┬─────┐
  Q₁  │ S₁₁ │ S₁₂ │ S₁₃ │ S₁₄ │  ← 第 i=1 轮: 逐块更新 O₁
      ├─────┼─────┼─────┼─────┤
  Q₂  │ S₂₁ │ S₂₂ │ S₂₃ │ S₂₄ │  ← 第 i=2 轮: 逐块更新 O₂
      ├─────┼─────┼─────┼─────┤
  Q₃  │ S₃₁ │ S₃₂ │ S₃₃ │ S₃₄ │  ← 第 i=3 轮: 逐块更新 O₃
      └─────┴─────┴─────┴─────┘

每个小块 Sᵢⱼ 的大小只有 Br × Bc, 完全放得进 SRAM!
整个 N×N 的 S 矩阵从未被完整构造出来.
```

外层遍历 K/V 块（列方向），内层遍历 Q 块（行方向）。对于每个 \((i, j)\) 组合：
- 计算一个小块 \(S_{ij} \in \mathbb{R}^{B_r \times B_c}\)（**在 SRAM 中，不写回 HBM**）
- 更新第 \(i\) 行的 softmax 统计量和输出 \(O_i\)

### 块大小的选择

Algorithm 1 第 1 行给出了块大小的设定：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B_r = \min\!\left(\left\lceil \frac{M}{4d} \right\rceil,\; d\right)
$$

这是为了保证 **每次循环需要的数据都能放进 SRAM**。一次内层迭代需要同时在 SRAM 中保存：
- \(K_j\): \(B_c \times d\)
- \(V_j\): \(B_c \times d\)
- \(Q_i\): \(B_r \times d\)
- \(O_i\): \(B_r \times d\)
- \(S_{ij}\): \(B_r \times B_c\)（中间计算结果）

总计约 \(2B_c d + 2B_r d + B_r B_c\) 个浮点数，需要不超过 \(M\) 个元素。

### 重计算（Recomputation）：反向传播的优化

标准反向传播需要保存 \(S, P \in \mathbb{R}^{N \times N}\) 用于梯度计算。FlashAttention 的策略是：**只保存 \(O\)、\(m\)、\(\ell\)，在反向传播时重新计算 \(S\) 和 \(P\)**。

| 方案 | 需要保存 | 额外内存 | 代价 |
|------|---------|---------|------|
| 标准反向传播 | \(S, P \in \mathbb{R}^{N \times N}\) | \(O(N^2)\) | 无 |
| 梯度检查点 | 不保存，全部重算 | \(O(N)\) | 速度慢（重复计算） |
| **FlashAttention** | \(O, m, \ell\) | \(O(N)\) | **反而更快**（减少 HBM 访问） |

这看似矛盾——重新计算不是增加了 FLOP 吗？确实，FlashAttention 的总 FLOP 略多于标准方法。但由于重计算发生在 SRAM 中（带宽 ~19 TB/s），而标准方法需要从 HBM 中读取 \(S\) 和 \(P\)（带宽 ~2 TB/s），**减少的 HBM 访问远比多出的 FLOP 划算**。

::: warning 反直觉的结论
更多的 FLOP + 更少的 HBM 访问 = 更快的实际速度。这再次印证了 FlashAttention 的核心哲学：**在 memory-bound 场景下，优化 IO 比优化计算更重要。**
:::

### Kernel 融合：一个 CUDA Kernel 搞定一切

Tiling 使得 FlashAttention 可以将所有计算步骤融合到 **一个 CUDA Kernel** 中：

```
一个 FlashAttention Kernel 内部的完整流程:

┌─────────────────────────────────────────────┐
│  CUDA Kernel (一次启动, 一次执行)            │
│                                             │
│  1. 从 HBM 加载 Q块, K块, V块 → SRAM       │
│  2. SRAM 中: S块 = Q块 × K块ᵀ              │
│  3. SRAM 中: Masking (可选, 因果掩码)        │
│  4. SRAM 中: 在线 Softmax (更新 m, ℓ)       │
│  5. SRAM 中: Dropout (可选, 训练时)          │
│  6. SRAM 中: O块 累加 = P̃块 × V块           │
│  7. 写回 O块, m, ℓ → HBM                    │
│                                             │
│  中间结果 S块, P̃块 全程留在 SRAM, 从不触碰 HBM │
└─────────────────────────────────────────────┘
```

相比标准实现需要多个 Kernel（matmul → mask → softmax → dropout → matmul），每个 Kernel 之间都要经过 HBM 中转，FlashAttention 只有 **一次 HBM 读入 + 一次 HBM 写出**。

### 正确性与复杂度（Theorem 1）

论文给出了严格证明（详见 Appendix C）：

> **Theorem 1**: Algorithm 1 返回 \(O = \text{softmax}(QK^\top)V\)，所需 FLOPs 为 \(O(N^2 d)\)，额外内存（输入输出之外）为 \(O(N)\)。

- **精确性**：输出与标准注意力 **完全一致**（不是近似），因为在线 softmax 的递推公式是数学恒等式
- **FLOPs**：与标准注意力相同，都是 \(O(N^2 d)\)——FlashAttention 没有减少计算量
- **额外内存**：只需要 \(O(N)\) 存储 \(m\) 和 \(\ell\)（而非标准方法的 \(O(N^2)\) 存储 \(S\) 和 \(P\)）

### 动手试一试

下面的 C++ 代码完整实现了 Algorithm 1，你可以对比上一节的 Algorithm 0，观察两者的输出 **完全一致**，但 HBM 访问量大幅下降。

```cpp-run title="Algorithm 1: FlashAttention — 分块 Tiling + 在线 Softmax"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cassert>

// ===== FlashAttention Algorithm 1 完整实现 =====
// 对比 Algorithm 0, 展示 HBM 访问量的显著下降

struct HBMTracker {
    long long reads = 0, writes = 0;
    void read(long long n)  { reads  += n; }
    void write(long long n) { writes += n; }
    long long total() const { return reads + writes; }
};

using Mat = std::vector<std::vector<float>>;
using Vec = std::vector<float>;

Mat make_mat(int r, int c, float v = 0) {
    return Mat(r, Vec(c, v));
}

// ========== Algorithm 0: 标准注意力 (基线) ==========
Mat standard_attention(const Mat& Q, const Mat& K, const Mat& V,
                       int N, int d, HBMTracker& hbm) {
    // Step 1: S = Q K^T
    Mat S = make_mat(N, N);
    hbm.read(N * d); hbm.read(N * d);   // 读 Q, K
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < d; k++) s += Q[i][k] * K[j][k];
            S[i][j] = s / sqrtf((float)d);
        }
    hbm.write(N * N);                    // 写 S

    // Step 2: P = softmax(S)
    Mat P = make_mat(N, N);
    hbm.read(N * N);                     // 读 S
    for (int i = 0; i < N; i++) {
        float mx = *std::max_element(S[i].begin(), S[i].end());
        float sm = 0;
        for (int j = 0; j < N; j++) { P[i][j] = expf(S[i][j] - mx); sm += P[i][j]; }
        for (int j = 0; j < N; j++) P[i][j] /= sm;
    }
    hbm.write(N * N);                    // 写 P

    // Step 3: O = P V
    Mat O = make_mat(N, d);
    hbm.read(N * N); hbm.read(N * d);   // 读 P, V
    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += P[i][k] * V[k][j];
            O[i][j] = s;
        }
    hbm.write(N * d);                    // 写 O
    return O;
}

// ========== Algorithm 1: FlashAttention ==========
Mat flash_attention(const Mat& Q, const Mat& K, const Mat& V,
                    int N, int d, int Br, int Bc, HBMTracker& hbm) {
    // Line 2: 初始化 O, ℓ, m
    Mat O = make_mat(N, d, 0.0f);
    Vec ell(N, 0.0f);                     // ℓ: softmax 分母累加
    Vec m(N, -1e30f);                     // m: 行最大值

    int Tc = (N + Bc - 1) / Bc;           // K/V 块数
    int Tr = (N + Br - 1) / Br;           // Q 块数

    // Line 5: 外层循环 — 遍历 K/V 块
    for (int j = 0; j < Tc; j++) {
        int kv_start = j * Bc;
        int kv_end   = std::min(kv_start + Bc, N);
        int kv_len   = kv_end - kv_start;

        // Line 6: 从 HBM 加载 Kⱼ, Vⱼ 到 SRAM
        hbm.read(kv_len * d);             // 读 Kⱼ
        hbm.read(kv_len * d);             // 读 Vⱼ

        // Line 7: 内层循环 — 遍历 Q 块
        for (int i = 0; i < Tr; i++) {
            int q_start = i * Br;
            int q_end   = std::min(q_start + Br, N);
            int q_len   = q_end - q_start;

            // Line 8: 从 HBM 加载 Qᵢ, Oᵢ, ℓᵢ, mᵢ
            hbm.read(q_len * d);           // 读 Qᵢ
            hbm.read(q_len * d);           // 读 Oᵢ
            hbm.read(q_len);               // 读 ℓᵢ
            hbm.read(q_len);               // 读 mᵢ

            // Line 9: Sᵢⱼ = Qᵢ Kⱼᵀ  (在 SRAM 中, 不写 HBM!)
            Mat Sij = make_mat(q_len, kv_len);
            for (int ii = 0; ii < q_len; ii++)
                for (int jj = 0; jj < kv_len; jj++) {
                    float s = 0;
                    for (int k = 0; k < d; k++)
                        s += Q[q_start + ii][k] * K[kv_start + jj][k];
                    Sij[ii][jj] = s / sqrtf((float)d);
                }

            // Line 10: 局部统计量 (在 SRAM 中)
            Vec m_tilde(q_len), ell_tilde(q_len, 0.0f);
            Mat P_tilde = make_mat(q_len, kv_len);
            for (int ii = 0; ii < q_len; ii++) {
                m_tilde[ii] = *std::max_element(
                    Sij[ii].begin(), Sij[ii].end());
                for (int jj = 0; jj < kv_len; jj++) {
                    P_tilde[ii][jj] = expf(Sij[ii][jj] - m_tilde[ii]);
                    ell_tilde[ii] += P_tilde[ii][jj];
                }
            }

            // Line 11-12: 更新 m, ℓ, O (在 SRAM 中计算, 写回 HBM)
            for (int ii = 0; ii < q_len; ii++) {
                int gi = q_start + ii;  // 全局行索引
                float m_new = std::max(m[gi], m_tilde[ii]);
                float ell_new = expf(m[gi] - m_new) * ell[gi]
                              + expf(m_tilde[ii] - m_new) * ell_tilde[ii];

                // 修正旧输出 + 加上新块贡献
                float scale_old = expf(m[gi] - m_new) * ell[gi] / ell_new;
                float scale_new = expf(m_tilde[ii] - m_new) / ell_new;

                for (int jj = 0; jj < d; jj++) {
                    float pv = 0;
                    for (int kk = 0; kk < kv_len; kk++)
                        pv += P_tilde[ii][kk] * V[kv_start + kk][jj];
                    O[gi][jj] = scale_old * O[gi][jj] + scale_new * pv;
                }

                m[gi]   = m_new;
                ell[gi] = ell_new;
            }

            // Line 12-13: 写回 Oᵢ, ℓᵢ, mᵢ 到 HBM
            hbm.write(q_len * d);          // 写 Oᵢ
            hbm.write(q_len);              // 写 ℓᵢ
            hbm.write(q_len);              // 写 mᵢ
        }
    }
    return O;
}

int main() {
    const int N  = 8;
    const int d  = 4;
    const int Br = 2;   // Q 块大小
    const int Bc = 2;   // K/V 块大小

    std::cout << "====== FlashAttention: Algorithm 0 vs Algorithm 1 ======\n";
    std::cout << "N=" << N << ", d=" << d
              << ", Br=" << Br << ", Bc=" << Bc << "\n\n";

    // 初始化 Q, K, V
    Mat Q = make_mat(N, d), K = make_mat(N, d), V = make_mat(N, d);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++) {
            Q[i][j] = sinf(i * 0.5f + j * 0.3f);
            K[i][j] = cosf(i * 0.4f + j * 0.2f);
            V[i][j] = sinf(i * 0.3f + j * 0.5f);
        }

    // --- 运行 Algorithm 0 ---
    HBMTracker hbm0;
    Mat O0 = standard_attention(Q, K, V, N, d, hbm0);

    // --- 运行 Algorithm 1 ---
    HBMTracker hbm1;
    Mat O1 = flash_attention(Q, K, V, N, d, Br, Bc, hbm1);

    // ========== 验证正确性 ==========
    std::cout << "--- 正确性验证: O₀ vs O₁ ---\n";
    float max_diff = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++)
            max_diff = std::max(max_diff, fabsf(O0[i][j] - O1[i][j]));
    std::cout << "  最大绝对误差: " << std::scientific
              << std::setprecision(2) << max_diff;
    std::cout << (max_diff < 1e-5f ? "  PASS 数值一致!\n" : "  FAIL 有差异\n");

    // 输出前 4 行
    std::cout << "\n--- 输出对比 (前 4 行) ---\n";
    for (int i = 0; i < std::min(N, 4); i++) {
        std::cout << "  Algo0 O[" << i << "] = [";
        for (int j = 0; j < d; j++)
            std::cout << (j?", ":"") << std::fixed
                      << std::setprecision(4) << O0[i][j];
        std::cout << "]\n  Algo1 O[" << i << "] = [";
        for (int j = 0; j < d; j++)
            std::cout << (j?", ":"") << std::fixed
                      << std::setprecision(4) << O1[i][j];
        std::cout << "]\n\n";
    }

    // ========== HBM 访问量对比 ==========
    std::cout << "========== HBM 访问量对比 ==========\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "  Algorithm 0 (标准):     "
              << "读=" << hbm0.reads << ", 写=" << hbm0.writes
              << ", 总=" << hbm0.total() << "\n";
    std::cout << "  Algorithm 1 (Flash):    "
              << "读=" << hbm1.reads << ", 写=" << hbm1.writes
              << ", 总=" << hbm1.total() << "\n";

    double ratio = (double)hbm1.total() / hbm0.total();
    std::cout << "\n  Flash / Standard = " << std::fixed
              << std::setprecision(2) << ratio << "x\n";

    // ========== 缩放分析 ==========
    std::cout << "\n========== N 增长时的 HBM 访问量 (理论值) ==========\n";
    std::cout << std::setw(8) << "N"
              << std::setw(16) << "Algo0 (4N²)"
              << std::setw(22) << "Algo1 (O(N²d/M))"
              << std::setw(10) << "比值" << "\n";

    int M_sram = 64;  // 模拟 SRAM 大小 (元素数)
    for (int n : {16, 64, 256, 1024}) {
        long long algo0 = 4LL * n * n + 4LL * n * d;
        // Flash: Tc * (2*Bc*d + Tr*(2*Br*d + 2*Br + 2*Br*d + 2*Br))
        // 简化为外层读 K/V + 内层读写 Q/O/m/l
        int bc = std::max(1, M_sram / (4 * d));
        int br = std::min(bc, d);
        int tc = (n + bc - 1) / bc;
        int tr = (n + br - 1) / br;
        long long algo1 = (long long)tc * (2*bc*d)
                        + (long long)tc * tr * (4*br*d + 4*br);
        double r = (double)algo1 / algo0;
        std::cout << std::setw(8) << n
                  << std::setw(16) << algo0
                  << std::setw(22) << algo1
                  << std::setw(9) << std::fixed << std::setprecision(2)
                  << r << "x\n";
    }

    std::cout << "\n[Key Insight]\n"
              << "  两个算法的输出完全一致 (精确算法, 非近似)!\n"
              << "  FlashAttention 多做了一些 FLOP (在线 Softmax 的修正计算),\n"
              << "  但 HBM 访问量大幅下降 —— 在 memory-bound 场景下, 这就是加速的来源.\n"
              << "  核心: S 和 P 始终留在 SRAM 中, 从未触碰 HBM!\n";

    return 0;
}
```

### 小结

FlashAttention 的 Algorithm 1 通过三个关键技术实现了 HBM 访问量的大幅下降：

| 技术 | 解决的问题 | 效果 |
|------|-----------|------|
| **Tiling（分块）** | \(S, P\) 太大放不进 SRAM | 分成 \(B_r \times B_c\) 的小块，每块在 SRAM 中完成全部计算 |
| **Online Softmax（在线 Softmax）** | Softmax 需要全局归一化 | 通过维护 \(m, \ell\) 统计量增量更新，结果精确一致 |
| **Recomputation（重计算）** | 反向传播需要 \(S, P\) | 只保存 \(O, m, \ell\)，反向时重新计算，减少 \(O(N^2) \to O(N)\) 额外内存 |

最终效果：**FLOPs 不变（甚至略多），但 HBM 访问从 \(O(N^2)\) 降至 \(O(N^2 d^2 M^{-1})\)**——在 memory-bound 场景下，这直接转化为 2-4 倍的墙钟加速。

---

## 分析：FlashAttention 的 IO 复杂度

上一节我们展示了 FlashAttention 算法的具体实现，但一个关键问题尚未严格回答：**FlashAttention 的 HBM 访问量到底是多少？这个量是最优的吗？**

论文在 Section 3.2 给出了两个核心理论结果：
1. **Theorem 2**（上界）：Algorithm 1 的 HBM 访问量为 \(O(N^2 d^2 M^{-1})\)
2. **Proposition 3**（下界）：任何精确注意力算法的 HBM 访问量不低于 \(\Omega(N^2 d^2 M^{-1})\)

两者匹配，说明 FlashAttention 在 HBM 访问次数上是 **渐近最优** 的。

### Theorem 2：HBM 访问量的上界

> **Theorem 2**: 设 \(N\) 为序列长度，\(d\) 为头维度，\(M\) 为 SRAM 大小（以元素计），且 \(d \leq M \leq Nd\)。Algorithm 1（FlashAttention）的 HBM 访问量为 \(O(N^2 d^2 M^{-1})\)。

证明方法很直接：**逐一统计 Algorithm 1 每一步的 HBM 读写量，然后求和**。

### 逐步推导

回顾 Algorithm 1 的块大小设定：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B_r = \min\!\left(\left\lceil \frac{M}{4d} \right\rceil,\; d\right)
$$

对应的块数为：

$$
T_c = \left\lceil \frac{N}{B_c} \right\rceil, \quad T_r = \left\lceil \frac{N}{B_r} \right\rceil
$$

现在统计每一层循环的 HBM 访问量：

**外层循环**（遍历 K/V 块，共 \(T_c\) 次）：
- 每次加载 \(K_j\)：\(B_c \times d\) 个元素
- 每次加载 \(V_j\)：\(B_c \times d\) 个元素
- 小计：\(2 B_c d\) 次读取

**内层循环**（遍历 Q 块，每个外层迭代执行 \(T_r\) 次）：
- 读取 \(Q_i, O_i\)：\(2 B_r d\) 次读取
- 读取 \(\ell_i, m_i\)：\(2 B_r\) 次读取
- 写回 \(O_i\)：\(B_r d\) 次写入
- 写回 \(\ell_i, m_i\)：\(2 B_r\) 次写入
- 小计：\(3 B_r d + 4 B_r\) 次访问

**总 HBM 访问量**：

$$
\text{Total} = T_c \cdot \big(2 B_c d + T_r \cdot (3 B_r d + 4 B_r)\big)
$$

主导项为内层循环部分：\(T_c \cdot T_r \cdot 3 B_r d\)。接下来分两种情况讨论：

#### 情况 1：\(M \leq 4d^2\)（SRAM 较小）

此时 \(B_r = B_c = \frac{M}{4d}\)，两个块大小相同。

$$
T_c = \frac{4Nd}{M}, \quad T_r = \frac{4Nd}{M}
$$

主导项：

$$
T_c \cdot T_r \cdot 3 B_r d = \frac{4Nd}{M} \cdot \frac{4Nd}{M} \cdot 3 \cdot \frac{M}{4d} \cdot d = \frac{16N^2d^2}{M^2} \cdot \frac{3M}{4} = \frac{12 N^2 d^2}{M}
$$

#### 情况 2：\(M > 4d^2\)（SRAM 较大）

此时 \(B_r = d\)，\(B_c = \frac{M}{4d}\)。

$$
T_c = \frac{4Nd}{M}, \quad T_r = \frac{N}{d}
$$

主导项：

$$
T_c \cdot T_r \cdot 3 B_r d = \frac{4Nd}{M} \cdot \frac{N}{d} \cdot 3 d^2 = \frac{12 N^2 d^2}{M}
$$

**两种情况均给出** \(\Theta\!\left(\frac{N^2 d^2}{M}\right)\)。

### 直觉理解：这个公式在说什么？

\(O(N^2 d^2 M^{-1})\) 这个表达式可以拆解理解：

| 因子 | 含义 |
|------|------|
| \(N^2\) | 注意力矩阵的"逻辑大小"——我们必须计算所有 \(N^2\) 个 token-pair 的交互 |
| \(d^2\) | 每对 token 的交互需要 \(d\) 维向量的内积；分块时每块还需要独立加载 \(d\) 维数据 |
| \(M^{-1}\) | SRAM 越大，每次能处理的数据块越大，需要的"轮次"越少 |

我们可以检查两个边界条件是否合理：

| 条件 | \(M\) 的值 | HBM 访问量 | 含义 |
|------|-----------|-----------|------|
| SRAM 极小 | \(M = d^2\) | \(N^2\) | 退化为标准注意力，和 Algorithm 0 一样 |
| SRAM 极大 | \(M = Nd\) | \(Nd\) | 只需读一遍输入，无需分块 |

这正好覆盖了从"完全放不下"到"完全放得下"的整个 SRAM 容量谱。

### Proposition 3：下界——FlashAttention 是最优的

> **Proposition 3**: 设 \(N \geq d\)，\(M\) 满足 \(d \leq M \leq Nd\)。则任何计算精确注意力的算法，其 HBM 访问量不低于 \(\Omega(N^2 d^2 M^{-1})\)。

这意味着 FlashAttention 的 HBM 访问量不仅仅是"足够好"——它已经达到了 **理论最优下界**，不可能再进一步减少（在渐近意义上）。

#### 证明思路

下界证明基于 **矩阵乘法的 IO 复杂度下界**（参考 Hong & Kung 1981 的"红蓝石子博弈"框架）：

1. **规约到矩阵乘法**：注意力计算必须（显式或隐式地）完成 \(S = QK^\top\) 这一步——这是一个 \((N \times d) \times (d \times N)\) 的矩阵乘法

2. **矩阵乘法的 IO 下界**：将 \(m \times k\) 矩阵与 \(k \times n\) 矩阵相乘，在 SRAM 大小为 \(M\) 的两级存储模型下，HBM 访问量的下界为：

$$
\Omega\!\left(\frac{m \cdot k \cdot n}{\sqrt{M}}\right)
$$

3. **应用到注意力**：对于 \(S = QK^\top\)，\(m = N, k = d, n = N\)：

$$
\Omega\!\left(\frac{N^2 d}{\sqrt{M}}\right)
$$

4. **注意力比纯矩阵乘法更难**：但注意力不仅要计算 \(S\)，还要对 \(S\) 做 softmax 再乘 \(V\)。softmax 是逐行操作，要求同一行的所有元素在同一时刻可达。论文通过更精细的分析证明，这个额外约束将下界提升至 \(\Omega(N^2 d^2 M^{-1})\)

::: tip 为什么下界比纯矩阵乘法更高？
纯矩阵乘法允许任意顺序计算输出元素，但 softmax 引入了 **行内全局依赖**：你必须看完 \(S\) 的整行才能做归一化。这迫使算法在 K/V 块的方向上"多扫几遍"，每多扫一遍就多一轮 HBM 读写。这个额外约束使 IO 下界从 \(\frac{N^2d}{\sqrt{M}}\) 提升到 \(\frac{N^2d^2}{M}\)。
:::

### 上下界匹配的意义

将 Theorem 2 和 Proposition 3 放在一起：

$$
\underbrace{\Omega\!\left(\frac{N^2 d^2}{M}\right)}_{\text{下界 (Prop. 3)}} \;\leq\; \text{FlashAttention 的 HBM 访问量} \;\leq\; \underbrace{O\!\left(\frac{N^2 d^2}{M}\right)}_{\text{上界 (Thm. 2)}}
$$

上下界完全匹配：**FlashAttention 是渐近最优的精确注意力算法**。

这是一个非常强的理论保证——它告诉我们：

1. **不要再找更好的**：在这个计算模型下，没有任何精确注意力算法能比 FlashAttention 做更少的 HBM 访问
2. **要想进一步加速，必须换赛道**：只能通过增加 SRAM（硬件改进）、使用近似注意力（允许误差）、或者改变注意力模式（如稀疏）来突破
3. **硬件设计的指导意义**：\(M^{-1}\) 的依赖关系表明，增加 SRAM 容量对注意力计算有直接的性能收益

### 动手试一试

下面的代码让你直观感受 SRAM 大小 \(M\) 如何影响 HBM 访问量，并验证理论公式 \(O(N^2 d^2 / M)\) 的准确性。

```cpp-run title="IO 复杂度分析: HBM 访问量 vs SRAM 大小 M"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>

// 模拟 Algorithm 1 的 HBM 访问量计数 (精确到每次读写)
long long simulate_flash_hbm(int N, int d, int M) {
    // 块大小设定 (Algorithm 1, Line 1)
    int Bc = std::max(1, (int)std::ceil((double)M / (4.0 * d)));
    int Br = std::min(Bc, d);

    // 块数
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;

    long long total = 0;

    // 外层循环: 遍历 K/V 块
    for (int j = 0; j < Tc; j++) {
        int kv_len = std::min(Bc, N - j * Bc);
        total += 2LL * kv_len * d;        // 读 Kj, Vj

        // 内层循环: 遍历 Q 块
        for (int i = 0; i < Tr; i++) {
            int q_len = std::min(Br, N - i * Br);
            total += 2LL * q_len * d;      // 读 Qi, Oi
            total += 2LL * q_len;           // 读 li, mi
            total += 1LL * q_len * d;       // 写 Oi
            total += 2LL * q_len;           // 写 li, mi
        }
    }
    return total;
}

// Algorithm 0 的 HBM 访问量 (固定公式)
long long standard_hbm(int N, int d) {
    return 4LL * N * d + 4LL * N * N;
}

int main() {
    const int d = 64;   // 典型头维度 (GPT-2)

    // ===== 实验 1: 固定 N, 改变 M =====
    std::cout << "===== 实验 1: 固定 N=" << 1024
              << ", d=" << d << ", 改变 SRAM 大小 M =====\n\n";

    int N = 1024;
    long long algo0 = standard_hbm(N, d);
    double theory_coeff = (double)N * N * d * d;  // N^2 * d^2

    std::cout << "  Algorithm 0 (标准): " << algo0 << " 次 HBM 访问\n\n";

    std::cout << std::setw(12) << "M (元素)"
              << std::setw(18) << "Flash HBM"
              << std::setw(18) << "N^2*d^2/M"
              << std::setw(12) << "实际/理论"
              << std::setw(14) << "Flash/Std"
              << "\n";
    std::cout << std::string(74, '-') << "\n";

    for (int M : {256, 1024, 4096, 16384, 32768, 65536}) {
        if (M < d || M > N * d) continue;

        long long flash = simulate_flash_hbm(N, d, M);
        double theory = theory_coeff / M;
        double ratio_theory = flash / theory;
        double ratio_std = (double)flash / algo0;

        std::cout << std::setw(12) << M
                  << std::setw(18) << flash
                  << std::setw(18) << (long long)theory
                  << std::setw(11) << std::fixed << std::setprecision(2)
                  << ratio_theory << "x"
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << ratio_std << "x"
                  << "\n";
    }

    // ===== 实验 2: 固定 M, 改变 N =====
    std::cout << "\n===== 实验 2: 固定 M=16384, d="
              << d << ", 改变序列长度 N =====\n\n";

    int M_fixed = 16384;

    std::cout << std::setw(8) << "N"
              << std::setw(16) << "Algo0 HBM"
              << std::setw(16) << "Flash HBM"
              << std::setw(14) << "Flash/Std"
              << std::setw(16) << "节省比例"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (int n : {128, 256, 512, 1024, 2048, 4096}) {
        long long a0 = standard_hbm(n, d);
        long long a1 = simulate_flash_hbm(n, d, M_fixed);
        double ratio = (double)a1 / a0;
        double saving = (1.0 - ratio) * 100;

        std::cout << std::setw(8) << n
                  << std::setw(16) << a0
                  << std::setw(16) << a1
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << ratio << "x"
                  << std::setw(13) << std::fixed << std::setprecision(1)
                  << saving << "%"
                  << "\n";
    }

    // ===== 实验 3: 验证两种情况的分界点 =====
    std::cout << "\n===== 实验 3: 情况 1 vs 情况 2 分界点 =====\n";
    std::cout << "  分界条件: M = 4d^2 = " << 4 * d * d << "\n\n";

    N = 512;
    std::cout << "  N=" << N << ", d=" << d << "\n";
    std::cout << std::setw(12) << "M"
              << std::setw(8) << "Br"
              << std::setw(8) << "Bc"
              << std::setw(8) << "Tc"
              << std::setw(8) << "Tr"
              << std::setw(14) << "情况"
              << std::setw(16) << "Flash HBM"
              << "\n";
    std::cout << std::string(74, '-') << "\n";

    for (int M : {1024, 4096, 8192, 16384, 32768}) {
        int Bc = std::max(1, (int)std::ceil((double)M / (4.0 * d)));
        int Br = std::min(Bc, d);
        int Tc = (N + Bc - 1) / Bc;
        int Tr = (N + Br - 1) / Br;
        long long flash = simulate_flash_hbm(N, d, M);

        const char* case_str = (M <= 4 * d * d) ? "Case 1" : "Case 2";
        std::cout << std::setw(12) << M
                  << std::setw(8) << Br
                  << std::setw(8) << Bc
                  << std::setw(8) << Tc
                  << std::setw(8) << Tr
                  << std::setw(14) << case_str
                  << std::setw(16) << flash
                  << "\n";
    }

    // ===== 边界条件验证 =====
    std::cout << "\n===== 边界条件验证 =====\n";
    N = 256;

    // M = d^2: 退化为标准注意力量级
    int M_small = d * d;
    long long flash_small = simulate_flash_hbm(N, d, M_small);
    long long std_small = standard_hbm(N, d);
    std::cout << "  M = d^2 = " << M_small
              << ": Flash=" << flash_small
              << ", Std=" << std_small
              << " (量级相当: "
              << std::fixed << std::setprecision(2)
              << (double)flash_small / std_small << "x)\n";

    // M = Nd: 只需读一遍输入
    int M_large = N * d;
    long long flash_large = simulate_flash_hbm(N, d, M_large);
    long long input_size = 3LL * N * d;
    std::cout << "  M = Nd = " << M_large
              << ": Flash=" << flash_large
              << ", 输入大小 3Nd=" << input_size
              << " (接近只读输入)\n";

    std::cout << "\n[Key Insight]\n"
              << "  FlashAttention 的 HBM 访问量 = Theta(N^2 * d^2 / M)\n"
              << "  这是所有精确注意力算法的渐近最优下界.\n"
              << "  SRAM 每增大一倍, HBM 访问量减半 -- 硬件升级直接转化为性能提升.\n";

    return 0;
}
```

### 与近似注意力方法的 IO 对比

论文还比较了 FlashAttention 与常见近似注意力方法的 IO 复杂度。虽然近似方法的 FLOPs 更少，但它们的 IO 模式未必更优：

| 方法 | FLOPs | HBM 访问量 | 精确？ |
|------|-------|-----------|--------|
| **标准注意力** (Algo 0) | \(O(N^2 d)\) | \(O(N^2 + Nd)\) | 是 |
| **FlashAttention** (Algo 1) | \(O(N^2 d)\) | \(O(N^2 d^2 M^{-1})\) | 是 |
| **稀疏注意力** (如 Longformer) | \(O(N \cdot S \cdot d)\) | \(O(N \cdot S + Nd)\) | 否 |
| **线性注意力** (如 Performer) | \(O(Nd^2)\) | \(O(Nd)\) | 否 |

其中 \(S\) 是稀疏注意力中每个 token 关注的邻域大小。

::: warning 一个关键观察
当 \(M\) 足够大（即 \(d^2 / M\) 足够小）时，FlashAttention 的 HBM 访问量 \(N^2 d^2 / M\) 可以远小于标准注意力的 \(N^2\)，甚至逼近线性注意力的 \(Nd\) 量级——**同时保持精确计算**。这就是 FlashAttention 在实践中能与近似方法竞争甚至胜出的理论基础。
:::

### 小结

| 理论结果 | 结论 |
|---------|------|
| **Theorem 2（上界）** | FlashAttention 的 HBM 访问量 \(\leq O(N^2 d^2 M^{-1})\) |
| **Proposition 3（下界）** | 任何精确注意力算法 \(\geq \Omega(N^2 d^2 M^{-1})\) |
| **合在一起** | FlashAttention 是渐近最优的 |

核心启示：

1. **算法已到极限**：在精确注意力的框架内，FlashAttention 的 IO 效率无法被本质性地超越
2. **SRAM 是关键资源**：\(M^{-1}\) 依赖意味着片上 SRAM 每扩大一倍，HBM 访问量减半。这为硬件设计（更大的 shared memory）和软件优化（更精细的 SRAM 管理）指明了方向
3. **FLOPs 不是全部**：FlashAttention 和标准注意力有相同的 FLOPs，但 IO 效率天差地别。**在 memory-bound 场景下，优化 IO 比优化计算更重要**

---

## 扩展：块稀疏 FlashAttention（Block-Sparse FlashAttention）

上两节我们证明了 FlashAttention 在 **精确注意力** 的框架下已达到 IO 最优。但在很多实际场景中，我们并不需要每个 token 都关注所有其他 token——**稀疏注意力（Sparse Attention）** 是一种广泛使用的提升效率的手段。

论文的一个重要贡献是：FlashAttention 的分块架构天然适合与块稀疏模式结合，只需 **极少的修改** 就能获得 IO 和 FLOPs 的双重收益。

### 什么是块稀疏注意力？

标准注意力计算完整的 \(N \times N\) 注意力矩阵。块稀疏注意力则引入一个 **块级掩码矩阵**：

$$
\tilde{M} \in \{0, 1\}^{T_r \times T_c}
$$

其中 \(T_r = \lceil N / B_r \rceil\)，\(T_c = \lceil N / B_c \rceil\)。当 \(\tilde{M}_{ij} = 0\) 时，表示 Q 的第 \(i\) 块和 K 的第 \(j\) 块之间的注意力被完全跳过——对应的 \(S_{ij}\) 块既不计算，也不参与 softmax。

```
块稀疏掩码示例 (N=16, Br=Bc=4, 所以 Tr=Tc=4):

        K₁   K₂   K₃   K₄
      ┌────┬────┬────┬────┐
  Q₁  │ 1  │ 1  │ 0  │ 0  │  → Q₁ 只关注 K₁, K₂
      ├────┼────┼────┼────┤
  Q₂  │ 1  │ 1  │ 1  │ 0  │  → Q₂ 关注 K₁, K₂, K₃
      ├────┼────┼────┼────┤
  Q₃  │ 0  │ 1  │ 1  │ 1  │  → Q₃ 关注 K₂, K₃, K₄
      ├────┼────┼────┼────┤
  Q₄  │ 0  │ 0  │ 1  │ 1  │  → Q₄ 只关注 K₃, K₄
      └────┴────┴────┴────┘

  1 = 计算该块     0 = 跳过该块
  非零块数: 10/16 = 62.5%  → 稀疏度 s = 0.625
```

### 从 Algorithm 1 到 Block-Sparse FlashAttention

由于 FlashAttention 已经将注意力按 \((B_r, B_c)\) 大小分块处理，加入块稀疏支持只需 **一行修改**：在内层循环中，检查掩码 \(\tilde{M}_{ij}\)，如果为 0 则直接跳过该块。

```
Algorithm 5: Block-Sparse FlashAttention (基于 Algorithm 1 的修改)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: Q, K, V ∈ R^{N×d}, 块稀疏掩码 M~ ∈ {0,1}^{Tr×Tc}

... (其余初始化与 Algorithm 1 相同) ...

5: for j = 1 to Tc do
6:     从 HBM 加载 Kⱼ, Vⱼ 到 SRAM
7:     for i = 1 to Tr do
7.5:       if M~[i][j] = 0 then continue    ← 唯一的新增行！
8:         从 HBM 加载 Qᵢ, Oᵢ, ℓᵢ, mᵢ 到 SRAM
9-13:     (与 Algorithm 1 完全相同)
14:    end for
15: end for

返回: O
```

这个"一行修改"看似简单，背后的深意是：**FlashAttention 的分块粒度恰好与块稀疏掩码的粒度对齐**。不需要任何索引重排、scatter/gather 操作或额外的数据结构——只需跳过零块即可。

### 常见的稀疏注意力模式

块稀疏框架可以统一表达多种流行的稀疏注意力模式。定义稀疏度 \(s\) 为非零块的比例：\(s = \frac{|\{(i,j) : \tilde{M}_{ij} = 1\}|}{T_r \times T_c}\)。

#### 1. 滑动窗口注意力（Local / Sliding Window）

每个 token 只关注前后 \(w\) 个位置范围内的 token：

$$
\tilde{M}_{ij} = \mathbb{1}\!\big[|i - j| \leq w / B_c\big]
$$

```
滑动窗口 (w=2 个块):

  ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │█│█│█│ │ │ │ │ │   稀疏度: s = (2w+1)/Tc
  │█│█│█│█│ │ │ │ │   当 w << N 时, s ≈ 2w/N
  │█│█│█│█│█│ │ │ │
  │ │█│█│█│█│█│ │ │   典型应用:
  │ │ │█│█│█│█│█│ │   - Longformer
  │ │ │ │█│█│█│█│█│   - Mistral
  │ │ │ │ │█│█│█│█│   - 局部上下文建模
  │ │ │ │ │ │█│█│█│
  └─┴─┴─┴─┴─┴─┴─┴─┘
```

#### 2. 因果注意力（Causal Attention）

自回归模型的标准模式——每个 token 只能看到自己及之前的 token：

$$
\tilde{M}_{ij} = \mathbb{1}\!\big[i \geq j\big]
$$

```
因果掩码 (下三角):

  ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │█│ │ │ │ │ │ │ │   稀疏度: s ≈ 0.5
  │█│█│ │ │ │ │ │ │
  │█│█│█│ │ │ │ │ │   典型应用:
  │█│█│█│█│ │ │ │ │   - GPT 系列
  │█│█│█│█│█│ │ │ │   - LLaMA
  │█│█│█│█│█│█│ │ │   - 所有自回归 LLM
  │█│█│█│█│█│█│█│ │
  │█│█│█│█│█│█│█│█│
  └─┴─┴─┴─┴─┴─┴─┴─┘
```

#### 3. 全局 + 滑动窗口（Global + Local）

少数特殊 token（如 [CLS]、句首）关注所有位置，其余 token 使用滑动窗口：

```
全局(前 2 行) + 滑动窗口:

  ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │█│█│█│█│█│█│█│█│   ← 全局 token
  │█│█│█│█│█│█│█│█│   ← 全局 token
  │█│█│█│█│█│ │ │ │
  │█│█│█│█│█│█│ │ │   典型应用:
  │█│█│ │█│█│█│█│ │   - BigBird
  │█│█│ │ │█│█│█│█│   - Longformer
  │█│█│ │ │ │█│█│█│   - ETC
  │█│█│ │ │ │ │█│█│
  └─┴─┴─┴─┴─┴─┴─┴─┘
```

#### 4. 步幅稀疏（Strided / Dilated）

以固定间隔采样关注位置，捕捉长距离依赖：

```
步幅稀疏 (stride=2):

  ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │█│ │█│ │█│ │█│ │   稀疏度: s = 1/stride
  │ │█│ │█│ │█│ │█│
  │█│ │█│ │█│ │█│ │   典型应用:
  │ │█│ │█│ │█│ │█│   - Sparse Transformer
  │█│ │█│ │█│ │█│ │   - 长序列音频建模
  │ │█│ │█│ │█│ │█│
  │█│ │█│ │█│ │█│ │
  │ │█│ │█│ │█│ │█│
  └─┴─┴─┴─┴─┴─┴─┴─┘
```

### IO 复杂度分析（Proposition 4）

> **Proposition 4**: 设块稀疏掩码的非零块比例为 \(s\)（\(0 < s \leq 1\)）。Block-Sparse FlashAttention 的 HBM 访问量为：
>
> $$O\!\left(\frac{N^2 d^2 s}{M}\right)$$

**证明直觉**：在 Algorithm 1 中，内层循环的迭代次数从 \(T_r \times T_c\) 减少为 \(s \cdot T_r \times T_c\)（只处理非零块）。每次迭代的 HBM 访问量不变，因此总量线性地乘以稀疏度 \(s\)。

与各方案的完整对比：

| 方法 | FLOPs | HBM 访问量 | 精确？ | 适用场景 |
|------|-------|-----------|--------|---------|
| 标准注意力 | \(O(N^2 d)\) | \(O(N^2)\) | 是 | 基线 |
| FlashAttention | \(O(N^2 d)\) | \(O(N^2 d^2 / M)\) | 是 | 通用加速 |
| **Block-Sparse Flash** | \(O(N^2 d \cdot s)\) | \(O(N^2 d^2 s / M)\) | 在稀疏模式内精确 | 已知稀疏模式 |
| 近似注意力 | 各异 | 各异 | 否 | 特定任务 |

::: tip Block-Sparse 的双重收益
注意 Block-Sparse FlashAttention 同时获得了 **FLOPs 和 IO 两方面的节省**——两者都乘以稀疏度 \(s\)。这与标准稀疏注意力不同：后者虽然减少了 FLOPs，但由于稀疏索引和 scatter/gather 操作，IO 反而可能更差。FlashAttention 的分块架构让稀疏模式的收益被"干净地"兑现。
:::

### 为什么 FlashAttention 特别适合块稀疏？

之所以 FlashAttention + 块稀疏如此自然，根本原因是 **粒度对齐**：

```
传统稀疏注意力:                    Block-Sparse FlashAttention:

  逐元素稀疏 → 需要稀疏索引          块级稀疏 → 只需 skip 整个块
  ┌─────────────────┐               ┌─────────────────┐
  │ 1 0 1 0 0 1 0 0 │               │ ██ │    │    │    │
  │ 0 1 0 0 1 0 0 1 │               │────┼────┼────┼────│
  │ 1 0 1 0 0 0 1 0 │   vs          │ ██ │ ██ │    │    │
  │ 0 0 0 1 0 1 0 0 │               │────┼────┼────┼────│
  │ 0 1 0 0 1 0 0 1 │               │    │ ██ │ ██ │    │
  │ 1 0 0 1 0 1 0 0 │               │────┼────┼────┼────│
  │ 0 0 1 0 0 0 1 0 │               │    │    │ ██ │ ██ │
  │ 0 1 0 0 1 0 0 1 │               └─────────────────┘
  └─────────────────┘
  - 需要 CSR/COO 等稀疏格式          - 掩码只有 Tr×Tc 个 bit
  - scatter/gather 内存不连续         - 整块 skip, 内存访问连续
  - GPU 利用率低 (warp divergence)    - 无 warp divergence
  - 额外索引开销抵消稀疏收益          - 稀疏收益被干净兑现
```

### 动手试一试

下面的代码对比标准 FlashAttention 和 Block-Sparse FlashAttention 在不同稀疏模式下的 HBM 访问量和 FLOPs。

```cpp-run title="Block-Sparse FlashAttention: 稀疏模式对比"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>

struct Stats {
    long long hbm = 0;
    long long flops = 0;
};

// 生成块稀疏掩码: 1 = 计算, 0 = 跳过
using Mask = std::vector<std::vector<int>>;

Mask make_dense_mask(int Tr, int Tc) {
    return Mask(Tr, std::vector<int>(Tc, 1));
}

Mask make_causal_mask(int Tr, int Tc) {
    Mask m(Tr, std::vector<int>(Tc, 0));
    for (int i = 0; i < Tr; i++)
        for (int j = 0; j <= i && j < Tc; j++)
            m[i][j] = 1;
    return m;
}

Mask make_sliding_window_mask(int Tr, int Tc, int w_blocks) {
    Mask m(Tr, std::vector<int>(Tc, 0));
    for (int i = 0; i < Tr; i++)
        for (int j = std::max(0, i - w_blocks);
             j <= std::min(Tc - 1, i + w_blocks); j++)
            m[i][j] = 1;
    return m;
}

Mask make_global_local_mask(int Tr, int Tc, int g_blocks, int w_blocks) {
    Mask m = make_sliding_window_mask(Tr, Tc, w_blocks);
    // 前 g_blocks 行关注所有列, 所有行也关注前 g_blocks 列
    for (int i = 0; i < std::min(g_blocks, Tr); i++)
        for (int j = 0; j < Tc; j++) m[i][j] = 1;
    for (int i = 0; i < Tr; i++)
        for (int j = 0; j < std::min(g_blocks, Tc); j++) m[i][j] = 1;
    return m;
}

Mask make_strided_mask(int Tr, int Tc, int stride) {
    Mask m(Tr, std::vector<int>(Tc, 0));
    for (int i = 0; i < Tr; i++)
        for (int j = 0; j < Tc; j++)
            if (j % stride == i % stride) m[i][j] = 1;
    return m;
}

// 统计掩码的非零块比例
double sparsity_ratio(const Mask& m) {
    int total = 0, nonzero = 0;
    for (auto& row : m)
        for (int v : row) { total++; nonzero += v; }
    return (double)nonzero / total;
}

// 模拟 Block-Sparse FlashAttention 的 HBM + FLOPs
Stats simulate_block_sparse(int N, int d, int Br, int Bc,
                            const Mask& mask) {
    Stats st;
    int Tc = (int)mask[0].size();
    int Tr = (int)mask.size();

    for (int j = 0; j < Tc; j++) {
        // 检查第 j 列是否有任何非零块
        bool col_used = false;
        for (int i = 0; i < Tr; i++)
            if (mask[i][j]) { col_used = true; break; }
        if (!col_used) continue;

        int kv_len = std::min(Bc, N - j * Bc);
        st.hbm += 2LL * kv_len * d;           // 读 Kj, Vj

        for (int i = 0; i < Tr; i++) {
            if (mask[i][j] == 0) continue;     // 跳过零块!

            int q_len = std::min(Br, N - i * Br);
            st.hbm += 2LL * q_len * d;         // 读 Qi, Oi
            st.hbm += 2LL * q_len;              // 读 li, mi
            st.hbm += 1LL * q_len * d;          // 写 Oi
            st.hbm += 2LL * q_len;              // 写 li, mi

            // FLOPs: Sij = Qi * Kj^T → 2*Br*Bc*d
            //        Pij * Vj       → 2*Br*Bc + 2*Br*d*Bc (approx)
            st.flops += 2LL * q_len * kv_len * d;  // QK^T
            st.flops += 2LL * q_len * kv_len * d;  // PV (approx)
        }
    }
    return st;
}

void print_mask_visual(const Mask& m, int max_size = 8) {
    int rows = std::min((int)m.size(), max_size);
    int cols = std::min((int)m[0].size(), max_size);
    std::cout << "    ";
    for (int j = 0; j < cols; j++)
        std::cout << "K" << j << " ";
    std::cout << "\n";
    for (int i = 0; i < rows; i++) {
        std::cout << "  Q" << i << " ";
        for (int j = 0; j < cols; j++)
            std::cout << (m[i][j] ? "## " : ".. ");
        std::cout << "\n";
    }
    if ((int)m.size() > max_size)
        std::cout << "    ... (共 " << m.size() << " 行)\n";
}

int main() {
    const int N  = 512;
    const int d  = 64;
    const int Br = 32;
    const int Bc = 32;
    const int Tr = (N + Br - 1) / Br;  // = 16
    const int Tc = (N + Bc - 1) / Bc;  // = 16

    std::cout << "====== Block-Sparse FlashAttention ======\n";
    std::cout << "N=" << N << ", d=" << d
              << ", Br=" << Br << ", Bc=" << Bc
              << ", Tr=" << Tr << ", Tc=" << Tc << "\n\n";

    // 定义各种稀疏模式
    struct Pattern {
        const char* name;
        Mask mask;
    };

    std::vector<Pattern> patterns = {
        {"Dense (全注意力)", make_dense_mask(Tr, Tc)},
        {"Causal (因果掩码)", make_causal_mask(Tr, Tc)},
        {"Sliding Window (w=2)", make_sliding_window_mask(Tr, Tc, 2)},
        {"Sliding Window (w=4)", make_sliding_window_mask(Tr, Tc, 4)},
        {"Global(1) + Local(2)", make_global_local_mask(Tr, Tc, 1, 2)},
        {"Strided (stride=2)", make_strided_mask(Tr, Tc, 2)},
        {"Strided (stride=4)", make_strided_mask(Tr, Tc, 4)},
    };

    // 打印前两个掩码的可视化
    for (int p = 2; p <= 3; p++) {
        std::cout << "--- " << patterns[p].name << " 掩码 ---\n";
        print_mask_visual(patterns[p].mask);
        std::cout << "\n";
    }

    // 统计对比
    Stats dense = simulate_block_sparse(N, d, Br, Bc, patterns[0].mask);

    std::cout << "===== HBM 访问量 & FLOPs 对比 =====\n\n";
    std::cout << std::setw(26) << "模式"
              << std::setw(10) << "稀疏度 s"
              << std::setw(14) << "HBM 访问"
              << std::setw(12) << "vs Dense"
              << std::setw(14) << "FLOPs"
              << std::setw(12) << "vs Dense"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (auto& p : patterns) {
        double s = sparsity_ratio(p.mask);
        Stats st = simulate_block_sparse(N, d, Br, Bc, p.mask);

        std::cout << std::setw(26) << p.name
                  << std::setw(9) << std::fixed << std::setprecision(3) << s
                  << std::setw(14) << st.hbm
                  << std::setw(10) << std::fixed << std::setprecision(3)
                  << (double)st.hbm / dense.hbm << "x"
                  << std::setw(14) << st.flops
                  << std::setw(10) << std::fixed << std::setprecision(3)
                  << (double)st.flops / dense.flops << "x"
                  << "\n";
    }

    // 验证: HBM 节省比例 ≈ 稀疏度 s
    std::cout << "\n===== 验证: HBM 节省比例 vs 稀疏度 s =====\n";
    std::cout << std::setw(26) << "模式"
              << std::setw(12) << "稀疏度 s"
              << std::setw(14) << "HBM 比例"
              << std::setw(12) << "差异"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (auto& p : patterns) {
        double s = sparsity_ratio(p.mask);
        Stats st = simulate_block_sparse(N, d, Br, Bc, p.mask);
        double hbm_ratio = (double)st.hbm / dense.hbm;
        double diff = std::abs(hbm_ratio - s);

        std::cout << std::setw(26) << p.name
                  << std::setw(11) << std::fixed << std::setprecision(4) << s
                  << std::setw(13) << std::fixed << std::setprecision(4)
                  << hbm_ratio
                  << std::setw(11) << std::fixed << std::setprecision(4)
                  << diff
                  << "\n";
    }

    // 缩放分析: N 增长时各模式的 HBM 访问量
    std::cout << "\n===== N 增长时的 HBM 访问量 (Sliding Window w=2 块) =====\n";
    std::cout << std::setw(8) << "N"
              << std::setw(14) << "Dense HBM"
              << std::setw(14) << "SW HBM"
              << std::setw(14) << "Causal HBM"
              << std::setw(12) << "SW/Dense"
              << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (int n : {128, 256, 512, 1024, 2048}) {
        int tr = (n + Br - 1) / Br;
        int tc = (n + Bc - 1) / Bc;
        Mask dm = make_dense_mask(tr, tc);
        Mask sw = make_sliding_window_mask(tr, tc, 2);
        Mask cm = make_causal_mask(tr, tc);

        Stats sd = simulate_block_sparse(n, d, Br, Bc, dm);
        Stats ss = simulate_block_sparse(n, d, Br, Bc, sw);
        Stats sc = simulate_block_sparse(n, d, Br, Bc, cm);

        std::cout << std::setw(8) << n
                  << std::setw(14) << sd.hbm
                  << std::setw(14) << ss.hbm
                  << std::setw(14) << sc.hbm
                  << std::setw(10) << std::fixed << std::setprecision(3)
                  << (double)ss.hbm / sd.hbm << "x"
                  << "\n";
    }

    std::cout << "\n[Key Insight]\n"
              << "  Block-Sparse FlashAttention 的 HBM 访问量 "
              << "= O(N^2 * d^2 * s / M)\n"
              << "  稀疏度 s 同时节省 FLOPs 和 IO, 两者近乎线性缩减.\n"
              << "  滑动窗口模式: N 越大, SW/Dense 比值越小 (s = O(w/N) → 0).\n"
              << "  因果掩码: 始终节省约 50%, 因为 s ≈ 0.5.\n";

    return 0;
}
```

### 实际影响

Block-Sparse FlashAttention 的实践意义体现在几个方面：

**1. 统一框架**

之前的稀疏注意力实现各自为政——Longformer 有自己的 CUDA kernel，BigBird 有自己的，Sparse Transformer 又是另一套。Block-Sparse FlashAttention 提供了一个统一的高效实现：只需指定不同的块掩码 \(\tilde{M}\)，就能支持任意块稀疏模式。

**2. IO 高效的稀疏注意力**

传统稀疏注意力的一个尴尬是：虽然 FLOPs 减少了，但由于稀疏索引和不连续内存访问，实际墙钟时间未必更快（论文实验中很多近似方法在 \(N < 2048\) 时反而比稠密注意力慢）。Block-Sparse FlashAttention 消除了这个问题——稀疏收益被干净地兑现为墙钟加速。

**3. 可组合性**

块稀疏掩码可以自由组合。例如："因果 + 滑动窗口"就是两个掩码的逐元素 AND。这使得用户可以根据任务需求灵活定制注意力模式，而无需为每种组合编写新的 kernel。

### 小结

| 维度 | 标准 FlashAttention | Block-Sparse FlashAttention |
|------|--------------------|-----------------------------|
| **FLOPs** | \(O(N^2 d)\) | \(O(N^2 d \cdot s)\) |
| **HBM 访问** | \(O(N^2 d^2 / M)\) | \(O(N^2 d^2 s / M)\) |
| **代码改动** | 基线 | 内层循环加一个 `if` |
| **支持的模式** | 全注意力 | 任意块稀疏模式 |
| **精确性** | 精确 | 在非零块内精确 |

核心观点：FlashAttention 的分块架构让块稀疏变成了"免费午餐"——几乎零代价地将稀疏模式的理论收益转化为实际加速。这也解释了为什么 FlashAttention 迅速成为几乎所有 Transformer 推理框架的底层注意力实现。

---

::: info 后续章节持续更新中
实验结果分析、FlashAttention-2 的改进等内容将在后续更新中补充。
:::

## 推荐阅读

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [Online Softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
- [vLLM: PagedAttention](./vllm-paper) — FlashAttention 的下游应用之一
