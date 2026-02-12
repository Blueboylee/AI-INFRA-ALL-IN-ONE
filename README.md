# AI Infra 学习笔记

> 从底层到上层，系统学习 AI 基础设施 —— 涵盖 CUDA、vLLM、NVIDIA Triton、OpenAI Triton、TensorRT 等 AI Infra 全栈技术。

📖 **在线阅读**：[https://blueboylee.github.io/AI-INFRA-ALL-IN-ONE/](https://blueboylee.github.io/AI-INFRA-ALL-IN-ONE/)

## 内容板块

| 板块 | 主题 | 说明 |
|------|------|------|
| ⚡ GPU 编程与优化 | CUDA 并行编程 | GPU 架构、内存优化与高性能计算核心技巧（参考 PMPP 教材） |
| 🚀 推理引擎与服务化 | vLLM / TensorRT / Triton Inference Server | 主流推理框架原理分析与部署方案 |
| 🔧 编译器与算子优化 | OpenAI Triton | 算子融合、Kernel 自动生成等前沿技术 |
| 🧠 AI Infra 全链路 | 训练 / 推理 / 调度 / 部署 | 从硬件到软件栈的基础设施全貌 |

## 已收录文章

### 推理引擎

- **vLLM** — PagedAttention 论文精读与解析
- **SGLang** — 结构化生成语言系统分析
- **Clipper** — 低延迟在线预测服务系统

> 持续更新中，后续将涵盖 TensorRT-LLM、Triton Inference Server 等更多主题。

## 特色功能

- **交互式 C++ Playground** — 基于 [Godbolt Compiler Explorer](https://godbolt.org/) 的在线 C++ 代码编辑与运行，支持 C++20 标准
- **VitePress 驱动** — 快速、轻量的静态文档站点
- **自动化部署** — 推送 `main` 分支即自动部署至 GitHub Pages

## 本地开发

**环境要求**：Node.js >= 20

```bash
# 克隆仓库
git clone https://github.com/Blueboylee/PMPP_cpp_AI_infra.git
cd PMPP_cpp_AI_infra

# 安装依赖
npm install

# 启动开发服务器
npm run docs:dev

# 构建生产版本
npm run docs:build

# 预览构建结果
npm run docs:preview
```

## 项目结构

```
.
├── docs/                          # 文档站点源码
│   ├── .vitepress/
│   │   ├── config.mts             # VitePress 站点配置
│   │   ├── markdown-it-cpp-playground.ts  # C++ Playground 插件
│   │   └── theme/                 # 自定义主题与组件
│   ├── posts/                     # 文章内容
│   │   └── inference/             # 推理引擎专题
│   ├── img/                       # 图片资源
│   ├── index.md                   # 首页
│   └── about.md                   # 关于页面
├── src/                           # 代码示例
├── .github/workflows/deploy.yml   # GitHub Pages 自动部署
└── package.json
```

## 许可证

[MIT](./LICENSE) © Blueboylee
