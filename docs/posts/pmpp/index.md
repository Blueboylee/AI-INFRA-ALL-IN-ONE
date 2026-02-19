# ⚡ PMPP：大规模并行处理器编程

::: info 📖 关于本专栏
本专栏是对 **"Programming Massively Parallel Processors: A Hands-on Approach"（第四版）** 一书的系统性精读笔记。作者 David B. Kirk（前 NVIDIA 首席科学家）和 Wen-mei W. Hwu（NVIDIA 研究高级副总裁、UIUC 教授）将 GPU 并行编程从硬件架构到算法设计讲透，是 CUDA 编程领域的"圣经"。
:::

---

## 为什么读这本书？

如果你正在做 AI Infra 相关的工作——无论是推理引擎优化、训练框架开发还是高性能算子编写——CUDA 编程都是绑定的底层能力。市面上关于 CUDA 的资料多而杂，而 PMPP 这本书的独特价值在于：

- **从原理出发**：不是 API 手册式罗列，而是从 GPU 硬件架构（SM、Warp、内存层次）出发推导编程范式
- **渐进式深入**：从最简单的向量加法到卷积、归约、前缀和、稀疏矩阵等经典并行模式，每一章都建立在前一章的基础上
- **性能分析驱动**：每个优化技巧都有定量分析——为什么 Shared Memory 能提速、Coalescing 到底省了多少带宽、Occupancy 如何影响延迟隐藏
- **紧跟前沿**：第四版加入了 Tensor Core、协作组（Cooperative Groups）、统一内存（Unified Memory）等现代特性

读完这本书，你会对以下问题有清晰的回答：

> GPU 有几千个核心，为什么我的 Kernel 还是很慢？\
> Shared Memory 到底该怎么用，Bank Conflict 怎么避免？\
> 为什么同样的算法，换一种内存访问模式就能快 10 倍？\
> Warp Divergence 对性能的影响有多大？\
> 如何系统性地分析和优化一个 CUDA Kernel？

---

## 专栏章节规划

本专栏按照原书章节结构组织，每一篇都包含：核心概念梳理、关键代码实现、性能分析与优化思考。

<div class="chapter-grid">

<a class="chapter-card" href="./ch01">
  <div class="chapter-num">01</div>
  <div class="chapter-body">
    <h3>Introduction</h3>
    <p class="chapter-desc">异构并行计算的背景与动机：为什么需要 GPU，CPU vs GPU 的架构差异，CUDA 编程模型概述。</p>
    <div class="chapter-tags">
      <span class="tag">背景</span>
      <span class="tag">异构计算</span>
      <span class="tag">CUDA 概述</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch02">
  <div class="chapter-num">02</div>
  <div class="chapter-body">
    <h3>Heterogeneous Data Parallel Computing</h3>
    <p class="chapter-desc">第一个 CUDA 程序：向量加法。理解 Kernel、Thread、Block、Grid 的基本概念和 Host-Device 交互流程。</p>
    <div class="chapter-tags">
      <span class="tag">Kernel</span>
      <span class="tag">Thread/Block/Grid</span>
      <span class="tag">CUDA API</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch03">
  <div class="chapter-num">03</div>
  <div class="chapter-body">
    <h3>Multidimensional Grids and Data</h3>
    <p class="chapter-desc">多维网格与数据映射：图像处理与矩阵运算中如何将多维数据映射到线程层次结构。</p>
    <div class="chapter-tags">
      <span class="tag">多维索引</span>
      <span class="tag">行优先布局</span>
      <span class="tag">图像处理</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch04">
  <div class="chapter-num">04</div>
  <div class="chapter-body">
    <h3>Compute Architecture and Scheduling</h3>
    <p class="chapter-desc">GPU 计算架构深入：SM 结构、Warp 执行模型、Warp 调度、Occupancy 与延迟隐藏。</p>
    <div class="chapter-tags">
      <span class="tag">SM</span>
      <span class="tag">Warp</span>
      <span class="tag">Occupancy</span>
      <span class="tag">延迟隐藏</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch05">
  <div class="chapter-num">05</div>
  <div class="chapter-body">
    <h3>Memory Architecture and Data Locality</h3>
    <p class="chapter-desc">GPU 内存层次结构：全局内存、共享内存、常量/纹理内存。Tiling 技术与数据局部性优化。</p>
    <div class="chapter-tags">
      <span class="tag">Shared Memory</span>
      <span class="tag">Tiling</span>
      <span class="tag">内存层次</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch06">
  <div class="chapter-num">06</div>
  <div class="chapter-body">
    <h3>Performance Considerations</h3>
    <p class="chapter-desc">性能优化核心：内存合并访问（Coalescing）、Bank Conflict、Warp Divergence、资源分配与性能权衡。</p>
    <div class="chapter-tags">
      <span class="tag">Coalescing</span>
      <span class="tag">Bank Conflict</span>
      <span class="tag">Warp Divergence</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch07">
  <div class="chapter-num">07</div>
  <div class="chapter-body">
    <h3>Convolution</h3>
    <p class="chapter-desc">并行卷积：基础实现、常量内存优化、Tiled 卷积与 Halo 元素处理，从 1D 到 2D 的推广。</p>
    <div class="chapter-tags">
      <span class="tag">卷积</span>
      <span class="tag">常量内存</span>
      <span class="tag">Halo Cells</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch08">
  <div class="chapter-num">08</div>
  <div class="chapter-body">
    <h3>Stencil</h3>
    <p class="chapter-desc">Stencil 并行模式：3D Stencil 计算、寄存器 Tiling、线程粗化（Thread Coarsening）策略。</p>
    <div class="chapter-tags">
      <span class="tag">Stencil</span>
      <span class="tag">Register Tiling</span>
      <span class="tag">Thread Coarsening</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch09">
  <div class="chapter-num">09</div>
  <div class="chapter-body">
    <h3>Parallel Histogram</h3>
    <p class="chapter-desc">并行直方图：原子操作、私有化（Privatization）、粗化与聚合策略，处理输出冲突的经典范式。</p>
    <div class="chapter-tags">
      <span class="tag">直方图</span>
      <span class="tag">原子操作</span>
      <span class="tag">Privatization</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch10">
  <div class="chapter-num">10</div>
  <div class="chapter-body">
    <h3>Reduction and Minimizing Divergence</h3>
    <p class="chapter-desc">并行归约：朴素归约的问题、减少 Warp Divergence 的优化、分段归约与 Kernel 级联策略。</p>
    <div class="chapter-tags">
      <span class="tag">归约</span>
      <span class="tag">Warp Divergence</span>
      <span class="tag">分段归约</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch11">
  <div class="chapter-num">11</div>
  <div class="chapter-body">
    <h3>Prefix Sum (Scan)</h3>
    <p class="chapter-desc">并行前缀和：Kogge-Stone、Brent-Kung 算法，工作效率分析，分层 Scan 处理任意长度输入。</p>
    <div class="chapter-tags">
      <span class="tag">Scan</span>
      <span class="tag">Kogge-Stone</span>
      <span class="tag">Brent-Kung</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch12">
  <div class="chapter-num">12</div>
  <div class="chapter-body">
    <h3>Merge</h3>
    <p class="chapter-desc">并行归并：Co-Rank 函数、分块并行归并、循环归并排序与高效负载均衡。</p>
    <div class="chapter-tags">
      <span class="tag">归并</span>
      <span class="tag">Co-Rank</span>
      <span class="tag">负载均衡</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch13">
  <div class="chapter-num">13</div>
  <div class="chapter-body">
    <h3>Sorting</h3>
    <p class="chapter-desc">并行排序：基数排序的并行化、基于 Scan 的分配策略，大规模数据的多遍排序。</p>
    <div class="chapter-tags">
      <span class="tag">基数排序</span>
      <span class="tag">并行排序</span>
      <span class="tag">Scan 应用</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch14">
  <div class="chapter-num">14</div>
  <div class="chapter-body">
    <h3>Sparse Matrix Computation</h3>
    <p class="chapter-desc">稀疏矩阵运算：COO、CSR、ELL、JDS 等存储格式，SpMV 的并行实现与性能对比。</p>
    <div class="chapter-tags">
      <span class="tag">SpMV</span>
      <span class="tag">CSR/ELL/JDS</span>
      <span class="tag">稀疏矩阵</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch15">
  <div class="chapter-num">15</div>
  <div class="chapter-body">
    <h3>Graph Traversal</h3>
    <p class="chapter-desc">并行图遍历：BFS 的并行化、层级同步与 Frontier-based 方法、非规则数据结构的挑战。</p>
    <div class="chapter-tags">
      <span class="tag">BFS</span>
      <span class="tag">图遍历</span>
      <span class="tag">Frontier</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch16">
  <div class="chapter-num">16</div>
  <div class="chapter-body">
    <h3>Deep Learning</h3>
    <p class="chapter-desc">深度学习中的并行计算：卷积层、全连接层的 GPU 实现，反向传播的并行化与 Tensor Core。</p>
    <div class="chapter-tags">
      <span class="tag">深度学习</span>
      <span class="tag">反向传播</span>
      <span class="tag">Tensor Core</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch17">
  <div class="chapter-num">17</div>
  <div class="chapter-body">
    <h3>Iterative MRI Reconstruction</h3>
    <p class="chapter-desc">迭代式 MRI 重建：非均匀 FFT、正则化重建的 GPU 加速，真实科学计算应用案例。</p>
    <div class="chapter-tags">
      <span class="tag">MRI 重建</span>
      <span class="tag">NUFFT</span>
      <span class="tag">科学计算</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch18">
  <div class="chapter-num">18</div>
  <div class="chapter-body">
    <h3>Electrostatic Potential Map</h3>
    <p class="chapter-desc">静电势图计算：直接库仑求和、截断方法、多级粒度优化与散射/聚集模式。</p>
    <div class="chapter-tags">
      <span class="tag">分子动力学</span>
      <span class="tag">库仑求和</span>
      <span class="tag">Scatter/Gather</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch19">
  <div class="chapter-num">19</div>
  <div class="chapter-body">
    <h3>Parallel Programming and Computational Thinking</h3>
    <p class="chapter-desc">并行编程方法论：问题分解、算法选择、计算思维，从串行思维到并行思维的转变。</p>
    <div class="chapter-tags">
      <span class="tag">计算思维</span>
      <span class="tag">问题分解</span>
      <span class="tag">方法论</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch20">
  <div class="chapter-num">20</div>
  <div class="chapter-body">
    <h3>Programming a Heterogeneous Computing Cluster</h3>
    <p class="chapter-desc">异构集群编程：MPI + CUDA 混合编程、多 GPU 通信、大规模并行系统的编程模型。</p>
    <div class="chapter-tags">
      <span class="tag">MPI + CUDA</span>
      <span class="tag">多 GPU</span>
      <span class="tag">集群编程</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch21">
  <div class="chapter-num">21</div>
  <div class="chapter-body">
    <h3>CUDA Dynamic Parallelism</h3>
    <p class="chapter-desc">CUDA 动态并行：Kernel 内启动 Kernel、递归并行模式、自适应网格与动态负载均衡。</p>
    <div class="chapter-tags">
      <span class="tag">动态并行</span>
      <span class="tag">嵌套 Kernel</span>
      <span class="tag">递归并行</span>
    </div>
  </div>
</a>

<a class="chapter-card" href="./ch22">
  <div class="chapter-num">22</div>
  <div class="chapter-body">
    <h3>Advanced Practices and Future Evolution</h3>
    <p class="chapter-desc">高级实践与未来展望：Unified Memory、协作组、CUDA Graphs、GPU 架构演进趋势。</p>
    <div class="chapter-tags">
      <span class="tag">Unified Memory</span>
      <span class="tag">Cooperative Groups</span>
      <span class="tag">CUDA Graphs</span>
    </div>
  </div>
</a>

</div>

---

## 阅读建议

| 读者背景 | 推荐路径 |
|---------|---------|
| **零基础入门** | Ch 1–6 打基础 → Ch 7–9 练手 → Ch 10–11 掌握核心算法模式 |
| **有 CUDA 基础** | 快速过 Ch 1–3 → 重点读 Ch 4–6 → 按兴趣选读应用章节 |
| **做 AI Infra** | Ch 4–6（性能优化）→ Ch 16（DL 相关）→ Ch 5 + Ch 7（理解 FlashAttention 的 Tiling 思想） |
| **做科学计算** | Ch 1–11（核心基础）→ Ch 14–15（稀疏/图）→ Ch 17–18（应用案例） |

::: tip 💡 持续更新中
本专栏将按章节逐步更新，每篇笔记都附带完整的代码示例和性能分析。敬请关注！
:::

<style>
.chapter-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
  margin: 24px 0;
}

@media (min-width: 768px) {
  .chapter-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.chapter-card {
  display: flex;
  gap: 16px;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  text-decoration: none !important;
  color: inherit !important;
  transition: all 0.3s ease;
  background: var(--vp-c-bg-soft);
}

.chapter-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.chapter-num {
  font-size: 28px;
  font-weight: 800;
  flex-shrink: 0;
  margin-top: 2px;
  color: var(--vp-c-brand-1);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  opacity: 0.8;
  min-width: 40px;
  text-align: center;
}

.chapter-body h3 {
  margin: 0 0 6px 0;
  font-size: 16px;
  font-weight: 600;
  line-height: 1.4;
  color: var(--vp-c-text-1);
}

.chapter-desc {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.chapter-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tag {
  padding: 2px 10px;
  font-size: 12px;
  border-radius: 999px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 500;
}
</style>
