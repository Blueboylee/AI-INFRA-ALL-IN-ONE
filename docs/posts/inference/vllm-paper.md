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

## 研究背景与动机

LLM 推理服务面临的核心挑战：

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
