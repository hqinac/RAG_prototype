# RAG 智能问答系统

一个基于 LangGraph 和 Gradio 构建的检索增强生成（RAG）智能问答系统，支持 Markdown 文档上传、智能路由、多种文本分割策略和答案质量评估。

## 🚀 功能特性

- **智能路由系统**：自动判断用户查询类型，选择最佳处理策略
- **多种文档分割策略**：支持固定长度、递归、Markdown 和语义分割
- **混合检索**：结合 FAISS 向量检索和 BM25 关键词检索
- **答案质量评估**：实时评估检索质量和生成答案的准确性
- **缓存管理**：智能缓存系统，提升响应速度
- **Web 界面**：基于 Gradio 的友好用户界面
- **心跳监控**：自动检测浏览器连接状态

## 📁 项目架构

```
RAG_prototype/
├── main.py              # 主程序入口，定义 LangGraph 工作流
├── gradio_app.py        # Gradio Web 界面
├── saver.py             # 文档保存和向量化处理
├── retriever.py         # 文档检索逻辑
├── evaluator.py         # 答案质量评估
├── cache_manager.py     # 缓存管理系统
├── pyproject.toml       # 项目配置和依赖
├── requirements.txt     # Python 依赖列表
└── saved_files/         # 数据存储目录
    ├── faiss_index/     # FAISS 向量索引
    ├── bm25.pkl         # BM25 检索器
    ├── split_docs.pkl   # 分割后的文档
    └── doc_info.json    # 文档信息
```

## 🛠️ 安装说明

### 环境要求

- Python 3.10+
- UV 包管理器（推荐）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd RAG_prototype
```

2. **使用 UV 安装依赖**
```bash
uv sync
```

或使用 pip：
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
创建 `.env` 文件并配置以下变量：
```env
# 通义千问 API 配置
DASHSCOPE_API_KEY=your_api_key_here

# 数据存储路径（可选）
URI=./saved_files

# 模型配置（可选）
EMBEDDING_MODEL=text-embedding-v1
LLM_MODEL=qwen-plus
```

## 🚀 使用方法

### 启动应用

```bash
# 使用 UV
uv run python gradio_app.py

# 或使用 Python
python gradio_app.py
```

应用启动后，访问 `http://localhost:7860` 即可使用。

### 基本使用流程

1. **上传文档**：支持 Markdown (.md) 格式文档
2. **选择分割策略**：
   - `fixed`：固定长度分割
   - `recursive`：递归字符分割
   - `md`：Markdown 结构化分割
   - `semantic`：语义分割
3. **开始对话**：输入问题，系统自动检索相关内容并生成答案
4. **查看评估**：系统会显示检索质量和答案质量评分

## 🏗️ 核心组件

### 1. 主工作流 (main.py)

基于 LangGraph 构建的状态图，包含三个核心节点：

- **路由节点 (router_node)**：智能判断查询类型
- **保存节点 (save_node)**：处理文档上传和向量化
- **检索节点 (retrieve_node)**：执行文档检索和答案生成

### 2. 缓存管理 (cache_manager.py)

提供统一的缓存管理接口：

- LLM 模型缓存
- 嵌入模型缓存
- FAISS 向量存储缓存
- BM25 检索器缓存
- 重排序模型缓存

### 3. 文档处理 (saver.py)

支持多种文本分割策略：

- **固定长度分割**：按字符数固定分割
- **递归分割**：智能递归分割
- **Markdown 分割**：基于 Markdown 结构
- **语义分割**：基于语义相似度

### 4. 检索系统 (retriever.py)

混合检索策略：

- **向量检索**：使用 FAISS 进行语义相似度检索
- **关键词检索**：使用 BM25 进行关键词匹配
- **结果融合**：智能合并两种检索结果

### 5. 质量评估 (evaluator.py)

双重评估机制：

- **检索评估**：评估检索结果的相关性和多样性
- **生成评估**：评估答案的忠实性、相关性和完整性

## 🔧 配置说明

### 模型配置

系统支持通义千问系列模型：

- **LLM 模型**：`qwen-plus`、`qwen-turbo`、`qwen-max`
- **嵌入模型**：`text-embedding-v1`、`text-embedding-v2`

### 分割参数

可在代码中调整分割参数：

```python
# 固定长度分割
chunk_size = 1000
chunk_overlap = 200

# 递归分割
separators = ["\n\n", "\n", " ", ""]

# 语义分割
breakpoint_threshold_type = "percentile"
```

## 📊 性能优化

- **缓存机制**：避免重复加载模型和数据
- **异步处理**：支持异步操作提升响应速度
- **内存管理**：智能垃圾回收和资源清理
- **心跳监控**：自动检测连接状态，及时释放资源

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的 LLM 应用开发框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - 状态图工作流引擎
- [Gradio](https://github.com/gradio-app/gradio) - 快速构建机器学习 Web 界面
- [FAISS](https://github.com/facebookresearch/faiss) - 高效向量相似度搜索
- [通义千问](https://dashscope.aliyun.com/) - 阿里云大语言模型服务
- [Reranker](# https://github.com/AnswerDotAI/rerankers) - 提供方便的重排序操作

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发起 Discussion
- 邮件联系：[your-email@example.com]

---

**注意**：使用前请确保已正确配置 API 密钥和环境变量。