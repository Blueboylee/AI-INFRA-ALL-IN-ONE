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

## 核心方法一：PagedAttention 算法（核心计算引擎）

这是 vLLM 的底层算法创新，解决了 KV 缓存必须连续存储的限制。

### 关键思想

借鉴操作系统的 **虚拟内存 + 分页** 机制：

- **分块化（Partitioning）**：将每个序列的 KV 缓存划分为固定大小的 **KV 块（KV Blocks）**，每个块包含 \(B\) 个 token 的 Key 和 Value 数据
- **非连续存储**：允许 KV 块散落在物理显存的不同位置，不再要求物理上的连续性
- **分块计算**：在注意力计算时，内核（Kernel）会根据地址索引，分别抓取不同的块进行分块计算，最后合并结果。这种方式对模型精度 **完全无损**

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

---

## 核心方法二：KV 缓存管理器（类虚拟内存系统）

vLLM 模仿操作系统，构建了一套 **逻辑与物理分离** 的完整内存管理体系。

### 逻辑块 vs 物理块

| 概念 | 类比 | 说明 |
|------|------|------|
| **逻辑块** | 虚拟页 | 模型视角下的连续序列（第一块、第二块……） |
| **物理块** | 物理页帧 | GPU 显存中实际的存储格子 |
| **块表 (Block Table)** | 页表 | 存储逻辑块到物理块的映射关系 |

### 按需分配（Dynamic Allocation）

系统 **不会** 预先为最大长度分配空间，只有当现有的逻辑块写满时，才会去物理内存池申请下一个块。这彻底消除了"预留空间浪费"和"内部碎片"。

```
传统方式 (预分配 max_seq_len):
┌─────────────────────────────────────────────────┐
│ Token 1 │ Token 2 │ Token 3 │ ░░ 浪费 ░░░░░░░░ │  ← 预留了 2048 但只用了 3
└─────────────────────────────────────────────────┘
  内部碎片 ≈ 99.85%

PagedAttention (按需分配):
┌──────────────────┐
│ Token 1 │ Token 2 │ Token 3 │ (空槽) │  ← 只分配了 1 个块 (B=4)
└──────────────────┘
  内部碎片 ≈ 最后一个块的空槽 (≤ B-1 个 Token)
```

::: info 与操作系统的精确类比
vLLM 的整套设计可以与 OS 虚拟内存一一对应：逻辑块 = 虚拟页，物理块 = 物理页帧，块表 = 页表，按需分配 = 请求调页（Demand Paging），Swap = 页面交换到磁盘。唯一的区别是：OS 管理的是 CPU 内存 ↔ 磁盘，vLLM 管理的是 GPU 显存 ↔ CPU 内存。
:::

---

## 核心方法三：多样化解码场景的内存共享

这是 vLLM 相比传统系统吞吐量翻倍的关键：**物理块共享**。通过引用计数 + 写时复制机制，vLLM 在多种解码策略下实现了极致的显存利用率。

### 1. 并行采样（Parallel Sampling）

当用户对同一个 Prompt 请求多个输出结果时（如 `n=4`）：

- **共享前缀**：多个输出序列共享同一个 Prompt 的物理块，通过 **引用计数（Reference Count）** 追踪共享关系
- **写时复制（Copy-on-Write）**：只有当某个输出分支产生新 Token 且当前物理块已满时，才会复制出一份新的物理块供其独立写入

```
Prompt: "写一首关于春天的诗"

共享阶段 (Prefill):
  Output 1 ─┐
  Output 2 ─┤── 共享 Prompt 的物理块 [P0, P1, P2]  ref_count = 4
  Output 3 ─┤
  Output 4 ─┘

分叉阶段 (Decoding):
  Output 1 ── [P0, P1, P2] → [P3]  (独立分配新块)
  Output 2 ── [P0, P1, P2] → [P4]  (独立分配新块)
  Output 3 ── [P0, P1, P2] → [P5]
  Output 4 ── [P0, P1, P2] → [P6]

  显存节省 = 3 × sizeof(Prompt_KV)  (传统方式需要 4 份独立拷贝)
```

### 2. 束搜索（Beam Search）

束搜索是一种更复杂的解码策略，不同候选束之间的共享关系呈 **树状结构**：

- **动态分叉**：每个束在每一步可能被保留或淘汰，形成不断变化的树
- **自动回收**：当某个束被淘汰，其对应物理块的引用计数减 1，减至 0 时立即释放
- **类进程树管理**：类似于 OS 中父进程 fork 子进程后的内存共享模式

```
Beam Search (beam_width = 3):

  Step 0: [root] ─┬─ beam_0: "The cat"     ref(P0)=3
                   ├─ beam_1: "The dog"
                   └─ beam_2: "A bird"

  Step 1: beam_0 ──┬─ beam_0': "The cat sat"    ref(P0)=2, ref(P1_cat)=2
                   └─ beam_1': "The cat ran"
          beam_2 ──── beam_2': "A bird flew"     ref(P2_bird)=1

  Step 2: beam_1' 被淘汰 → ref(P1_cat) -= 1
          beam_2' 被淘汰 → ref(P2_bird) = 0 → 立即释放!
```

### 3. 共享前缀（Shared Prefix）

像操作系统共享 **动态链接库（DLL）** 一样，多个请求可以共用同一段预热好的物理块：

- **典型场景**：System Prompt（系统指令）在所有请求间共享
- **效果**：极大节省显存并跳过冗余的 Prefill 计算

```
System Prompt: "You are a helpful assistant..."

  请求 A ── [Sys_P0, Sys_P1] → [A_P0, A_P1, ...]
  请求 B ── [Sys_P0, Sys_P1] → [B_P0, ...]        共享!
  请求 C ── [Sys_P0, Sys_P1] → [C_P0, C_P1, ...]
                ↑
           ref_count = 3, 物理上只存一份
```

---

## 调度策略与抢占机制

当请求过多导致显存"爆仓"时，vLLM 采取了一套精心设计的保护措施。

### 调度原则

- **先来先服务（FCFS）**：确保公平，防止先来的长任务被"饿死"
- **全有或全无（All-or-nothing）**：抢占时以整个 **序列组（Sequence Group）** 为单位，要么全留，要么全踢，防止共享块逻辑混乱

### 抢占恢复方案

当显存不足时，vLLM 提供两种恢复策略：

| 方案 | 机制 | 适用场景 | 代价 |
|------|------|---------|------|
| **交换（Swapping）** | 将物理块搬到 CPU 内存暂存，腾出 GPU 空间 | CPU 内存充裕时 | PCIe 传输延迟 |
| **重计算（Recomputation）** | 直接丢弃 KV Cache，需要时利用 Prefill 的高并行特性重新算 | CPU 内存也紧张时 | 重复计算开销 |

```
显存爆仓时的抢占流程:

  ┌──────────────────────────────────────────────┐
  │              GPU 显存 (已满!)                  │
  │  [Seq A] [Seq B] [Seq C] [Seq D] [Seq E]    │
  └──────────────────────────────────────────────┘
                     │
            新请求 F 到达, 需要分配块
                     │
                     ▼
        选择最后到达的 Seq E 进行抢占
                   ╱     ╲
                  ╱       ╲
    方案 1: Swap           方案 2: Recompute
    [Seq E] ──PCIe──► CPU  [Seq E] 的 KV Cache 丢弃
    (后续可搬回 GPU)        (后续重新 Prefill)
```

::: tip Swap vs Recompute 的权衡
Swap 保留了已计算的 KV Cache，恢复时只需要搬回来，但要占用 CPU 内存和 PCIe 带宽。Recomputation 不占额外内存，但恢复时需要重跑 Prefill。在实践中，短序列倾向于 Recompute（Prefill 很快），长序列倾向于 Swap（避免大量重复计算）。
:::

---

## 分布式执行（Multi-GPU）

vLLM 支持 **Megatron-LM 风格的张量并行（Tensor Parallelism）**，实现了优雅的多 GPU 协同。

### 设计思路：单脑多肢

```
                    ┌──────────────────┐
                    │   中央调度器      │
                    │  (维护统一块表)   │
                    └────────┬─────────┘
                             │ 广播 Block Table IDs
                ┌────────────┼────────────┐
                ▼            ▼            ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │  GPU 0   │ │  GPU 1   │ │  GPU 2   │
          │ Heads 0-7│ │Heads 8-15│ │Heads16-23│
          └──────────┘ └──────────┘ └──────────┘
          各自管理自己负责的 Attention Heads 数据
```

- **中央调度器** 维护一套统一的块表（逻辑到物理的 ID 映射）
- **控制广播**：调度器只需将块表 ID 广播给所有 GPU Worker，各 GPU 按照相同的 ID 去读写各自负责的那部分数据（Attention Heads）
- **无感知共享**：GPU Worker 无需在执行时同步内存状态，只需专注于模型计算。整个过程中 **不需要额外的内存管理通信**

---

## 实验结果

### 吞吐量对比

- 在 OPT-13B / OPT-175B 上，相比 HuggingFace Transformers 吞吐提升 **14-24x**
- 相比 NVIDIA FasterTransformer 吞吐提升 **2.2-4.3x**
- KV Cache 内存浪费从 60-80% 降至 **< 4%**

### 不同解码场景的收益

| 解码策略 | 显存节省 | 核心机制 |
|---------|---------|---------|
| 基础采样 (n=1) | 按需分配，消除碎片 | 分页 + 动态分配 |
| 并行采样 (n=4) | Prompt KV Cache 只存一份 | 引用计数 + CoW |
| 束搜索 (beam=4) | 中间状态树形共享 | 引用计数 + 自动回收 |
| 共享前缀 | System Prompt 全局共享 | 共享物理块映射 |

---

## 个人思考

1. **分页思想的迁移**：操作系统的经典设计模式在 AI 系统中焕发新生，说明底层系统设计思维的通用性。vLLM 几乎将 OS 虚拟内存的全套工具箱（分页、页表、按需分配、CoW、Swap）搬到了 GPU 显存管理中
2. **"内存共享"才是真正的杀手锏**：PagedAttention 的分页机制消除碎片只是基础，真正让吞吐量翻倍的是物理块共享——并行采样、束搜索、共享前缀这三种场景下的引用计数 + CoW 机制，使得 vLLM 能用同等显存服务远超传统系统的并发量
3. **工程与研究的结合**：vLLM 不仅是论文，更是生产级系统，值得深入阅读源码
4. **后续影响**：PagedAttention 已成为 LLM Serving 的标配，SGLang、TensorRT-LLM 等均有类似设计

## 推荐阅读

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention 原理动画](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Clipper: 在线推理服务系统](./clipper) — vLLM 自适应批处理思想的前身
- [SGLang: 结构化生成语言](./sglang) — 在 PagedAttention 基础上进一步实现跨请求 KV Cache 复用
