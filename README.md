# Legal Agentic RAG

基于 Agentic RAG 与轻量级微调的垂直领域（法律）大模型分析系统

## 功能特性

- **AST长文本解析**: 按"编-章-节"层级切分法典并强绑定 Metadata
- **HyDE检索增强**: 抹平口语与法言法语语义鸿沟
- **双阶段检索**: FAISS向量粗排 + BGE-Reranker交叉精排
- **数据蒸馏**: Teacher LLM + ReAct轨迹 + IRAC法律推演
- **QLoRA微调**: 4-bit量化，<1B参数模型适配8GB GPU
- **对抗训练**: 15%负样本内化安全拒答边界
- **Self-Reflection**: 智能体推理闭环，动态触发二次/三次定向追查

## 模型路径

- 基座模型: `/home/y/project/hgmodels/Qwen3-0.6B` (<1B参数)
- Embedding: `/home/y/project/hgmodels/bge-small-zh-v1.5`

## 项目结构

```
legal_agentic_rag/
├── configs/          # 配置文件
├── data/             # 数据目录
├── models/           # 模型输出
├── scripts/          # 执行脚本
└── src/
    ├── data/         # 数据处理
    ├── retrieval     # 检索模块
    ├── training      # 训练模块
    └── inference     # 推理模块
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成数据

```bash
python scripts/generate_data.py
```

### 3. 构建索引

```bash
python scripts/build_index.py
```

### 4. 训练模型

```bash
python scripts/train.py
```

### 5. 推理

```bash
# 单次查询
python scripts/inference.py --query "什么是合同法"

# 交互模式
python scripts/inference.py --interactive
```

## 数据说明

- **RAG数据**: 从民法典解析的1260条法律条文
- **训练数据**: 3450条 (3000条蒸馏数据 + 450条对抗负样本)
- **is_best=1**: 从lawzhidao_filter.csv筛选优质问答对

## 训练配置

- 基座模型: Qwen3-0.6B (约600M参数)
- LoRA rank: 16
- 量化: 4-bit NF4
- 批次大小: 2
- 学习率: 2e-4
- 序列长度: 1024
