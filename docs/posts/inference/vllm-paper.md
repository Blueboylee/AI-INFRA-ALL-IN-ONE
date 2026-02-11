---
title: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"
date: 2026-02-12
---

# vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

<p style="color: var(--vp-c-text-2); font-size: 14px;">
📅 2026-02-12 &nbsp;·&nbsp; 🏷️ 推理引擎 &nbsp;·&nbsp; 📖 论文精读
</p>

> **论文信息**
> - **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng et al.
> - **机构**: UC Berkeley
> - **发表**: SOSP 2023
> - **链接**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

## 一句话总结

vLLM 提出了 **PagedAttention** 机制，借鉴操作系统虚拟内存分页思想管理 KV Cache，将 LLM 推理吞吐量提升 **2-4x**，同时几乎消除了内存浪费。

## 背景：LLM 推理的现状与挑战

### 1. 核心架构：自回归 Transformer

**数学本质**：LLM 本质上是在做"条件概率预测"，即根据已有的序列 \(x_1, \dots, x_n\) 预测下一个 Token \(x_{n+1}\)。

**计算特征**：
- **独立计算**：大部分层（如前馈网络、归一化层等）在每个位置上是独立运行的。
- **关联计算**：只有**自注意力层（Self-attention）** 需要将当前位置的 Query 与之前所有位置的 Key 和 Value 进行交互。

### 2. 生成过程的两阶段论

LLM 的服务过程被明确划分为两个性质截然不同的阶段：

**提示词阶段（Prompt Phase / Prefill）**：
- **特点**：一次性处理所有输入，计算并行度高。
- **性能**：属于**计算密集型**，能跑满 GPU 的算力。

**自回归生成阶段（Generation Phase / Decoding）**：
- **特点**：逐个生成 Token，每步都依赖上一步的结果。
- **性能**：由于每步只处理一个 Token，计算量小但需要频繁读取 KV Cache，导致 GPU 算力闲置。这是一个典型的**内存带宽受限（Memory-bound）** 过程。

### 3. KV 缓存（KV Cache）的重要性

- **定义**：为了避免重复计算，系统会将之前 Token 的 Key 和 Value 向量存入显存，这部分显存就是 KV Cache。
- **痛点**：由于序列长度动态增长，且每个 Token 的 KV 向量会随上下文位置变化，导致显存管理极其困难。

### 4. 批处理（Batching）的演进

- **传统静态批处理**：存在"木桶效应"，短请求必须等长请求跑完，且需要填充（Padding），浪费严重。
- **迭代级调度（Iteration-level Scheduling）**：这是目前的主流方案（如 Orca 提出）。它允许在每次 Token 生成的"缝隙"里踢出已完成的请求、塞入新请求。
- **结论**：虽然解决了调度灵活性，但并没有解决显存管理低效的问题。

---

## 研究背景与动机

在上述背景下，vLLM 聚焦于解决 LLM 推理服务面临的核心挑战：

1. **KV Cache 内存开销大**：每个请求需要维护 KV Cache，随序列长度线性增长
2. **内存碎片化严重**：传统方式预分配连续内存块，导致内部和外部碎片
3. **动态长度难管理**：请求的输出长度不可预知，无法精确预分配内存

## 核心方法：PagedAttention

### 关键思想

借鉴操作系统的 **虚拟内存 + 分页** 机制：

- 将 KV Cache 划分为固定大小的 **物理块 (Physical Block)**
- 每个序列的 KV Cache 通过 **页表 (Block Table)** 映射到非连续的物理块
- 支持按需分配，最后一个块可部分填充

### 架构示意

```
逻辑 KV Cache (连续视图)
┌──────────┬──────────┬──────────┐
│  Block 0 │  Block 1 │  Block 2 │
└──────────┴──────────┴──────────┘
      │           │          │
      ▼           ▼          ▼      (通过 Block Table 映射)
┌──────────┬──────────┬──────────┐
│ Phys  7  │ Phys  1  │ Phys  3  │  物理 GPU 内存 (非连续)
└──────────┴──────────┴──────────┘
```

### 优势

| 特性 | 传统方式 | PagedAttention |
|------|---------|----------------|
| 内存分配 | 预分配连续大块 | 按需分页分配 |
| 内部碎片 | 严重（预留未用空间） | 仅最后一个块 |
| 外部碎片 | 存在 | 完全消除 |
| 内存共享 | 不支持 | 支持 Copy-on-Write |

### 动手试一试

下面的 C++ 代码模拟了 PagedAttention 的核心机制：物理块池、页表映射、按需分配。点击 **运行** 观察两个并发序列的 KV Cache 如何在非连续的物理内存中被管理。你也可以修改代码来实验不同的参数。

```cpp-run title="PagedAttention KV Cache 模拟"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <iomanip>

// ===== 模拟 PagedAttention 的核心思想 =====
// 将 KV Cache 从连续分配改为按需分页管理

constexpr int BLOCK_SIZE   = 4;  // 每个物理块能存储的 Token 数
constexpr int TOTAL_BLOCKS = 8;  // GPU 显存中的总物理块数

struct PhysicalBlock {
    float data[BLOCK_SIZE] = {};
    int used = 0;
};

struct BlockTable {
    std::vector<int> mapping;  // 页表: 逻辑块号 -> 物理块号
};

class PagedKVCache {
    PhysicalBlock pool_[TOTAL_BLOCKS];
    bool free_[TOTAL_BLOCKS];
    std::unordered_map<int, BlockTable> tables_;

public:
    PagedKVCache() { for (auto& f : free_) f = true; }

    int alloc_block() {
        for (int i = 0; i < TOTAL_BLOCKS; i++)
            if (free_[i]) { free_[i] = false; return i; }
        return -1;  // OOM
    }

    void append_token(int seq_id, float value) {
        auto& table = tables_[seq_id];
        if (table.mapping.empty() ||
            pool_[table.mapping.back()].used >= BLOCK_SIZE) {
            int blk = alloc_block();
            if (blk < 0) {
                std::cout << "  [OOM] seq " << seq_id << "\n";
                return;
            }
            table.mapping.push_back(blk);
        }
        int pid = table.mapping.back();
        pool_[pid].data[pool_[pid].used++] = value;
    }

    void print_status() const {
        std::cout << "\n=== 物理内存布局 ===\n";
        for (int i = 0; i < TOTAL_BLOCKS; i++) {
            std::cout << "  Block " << i
                      << (free_[i] ? " [free]  " : " [used]  ");
            if (!free_[i])
                for (int j = 0; j < pool_[i].used; j++)
                    std::cout << pool_[i].data[j] << " ";
            std::cout << "\n";
        }
        std::cout << "\n=== 页表映射 (logical -> physical) ===\n";
        for (auto& [sid, t] : tables_) {
            std::cout << "  Seq " << sid << ": ";
            for (size_t i = 0; i < t.mapping.size(); i++)
                std::cout << (i ? ", " : "") << i << "->" << t.mapping[i];
            std::cout << "\n";
        }
    }

    int free_count() const {
        int n = 0; for (auto f : free_) n += f; return n;
    }
};

int main() {
    std::cout << "====== PagedAttention KV Cache 模拟 ======\n";
    std::cout << "Block size : " << BLOCK_SIZE << " tokens\n";
    std::cout << "Total blocks: " << TOTAL_BLOCKS << "\n";

    PagedKVCache cache;

    // 模拟两个并发请求的 KV Cache 动态增长
    std::cout << "\n-- Seq 0: append 6 tokens --\n";
    for (int i = 0; i < 6; i++) cache.append_token(0, i * 1.0f);

    std::cout << "-- Seq 1: append 3 tokens --\n";
    for (int i = 0; i < 3; i++) cache.append_token(1, (i + 10) * 1.0f);

    cache.print_status();

    double util = (1.0 - (double)cache.free_count() / TOTAL_BLOCKS) * 100;
    std::cout << "\nFree blocks: " << cache.free_count()
              << "/" << TOTAL_BLOCKS
              << "  utilization: " << std::fixed << std::setprecision(1)
              << util << "%\n";

    std::cout << "\n[Key insight]\n"
              << "  两个序列的 KV Cache 在物理内存中是非连续、交错的,\n"
              << "  但通过页表, 每个序列看到的是一段连续的逻辑地址空间.\n"
              << "  按需分配 + 分页映射 = PagedAttention 的核心思想!\n";
    return 0;
}
```

## 关键优化

### 1. Copy-on-Write (CoW)

当多个序列共享前缀（如 system prompt）时，可以共享物理块，仅在写入时复制，节省大量内存。

### 2. 调度策略

vLLM 实现了一套高效的请求调度策略：
- **First-Come-First-Served (FCFS)** 基础调度
- **Preemption** 机制：当内存不足时，抢占低优先级请求
- 支持 **Swap** 和 **Recomputation** 两种抢占恢复方式

## 实验结果

- 在 OPT-13B / OPT-175B 上，相比 HuggingFace Transformers 吞吐提升 **14-24x**
- 相比 NVIDIA FasterTransformer 吞吐提升 **2.2-4.3x**
- KV Cache 内存浪费从 60-80% 降至 **< 4%**

## 个人思考

1. **分页思想的迁移**：操作系统的经典设计模式在 AI 系统中焕发新生，说明底层系统设计思维的通用性
2. **工程与研究的结合**：vLLM 不仅是论文，更是生产级系统，值得深入阅读源码
3. **后续影响**：PagedAttention 已成为 LLM Serving 的标配，SGLang、TensorRT-LLM 等均有类似设计

## 推荐阅读

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention 原理动画](https://blog.vllm.ai/2023/06/20/vllm.html)
