# 🤖 RAG智能问答系统

基于LangGraph和Gradio构建的智能文档问答系统，支持上传Markdown文档并进行智能对话。

## ✨ 功能特性

- 📁 **文档上传**: 支持Markdown(.md)文件上传
- 🔍 **智能检索**: 基于向量相似度的文档检索
- 💬 **对话交互**: 友好的Web聊天界面
- 🧠 **智能路由**: 自动识别保存和检索操作
- 📊 **状态监控**: 实时显示系统状态

## 🏗️ 系统架构

```
用户界面 (Gradio)
    ↓
文档处理模块
    ↓
LangGraph路由器
    ↓
┌─────────────┬─────────────┬─────────────┐
│  保存节点    │  检索节点    │  未知节点    │
└─────────────┴─────────────┴─────────────┘
    ↓
向量数据库 (FAISS)
```

## 🚀 快速开始

### 1. 环境准备

确保您有Python 3.8+环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 环境变量配置

设置必要的环境变量：

**Windows (PowerShell):**
```powershell
$env:DASHSCOPE_API_KEY='your_dashscope_api_key'
$env:DASH_MODEL_NAME='qwen-plus'
$env:DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/api/v1'
```

**Linux/Mac:**
```bash
export DASHSCOPE_API_KEY='your_dashscope_api_key'
export DASH_MODEL_NAME='qwen-plus'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/api/v1'
```

### 3. 启动应用

使用启动脚本：
```bash
python run.py
```

或直接启动：
```bash
python gradio_app.py
```

### 4. 使用系统

1. 打开浏览器访问 `http://localhost:7860`
2. 上传Markdown文档
3. 在聊天框中提问
4. 享受智能问答体验！

## 📁 项目结构

```
RAG_prototype/
├── gradio_app.py          # Gradio界面主文件
├── main.py               # 原始LangGraph定义
├── main_fixed.py         # 修复版LangGraph定义
├── saver.py             # 向量存储模块
├── run.py               # 启动脚本
├── requirements.txt     # 依赖列表
└── README.md           # 项目说明
```

## 🔧 核心组件

### LangGraph路由器
- **router_node**: 智能分类用户意图
- **save_node**: 处理文档保存操作
- **retrieve_node**: 执行文档检索和问答
- **unknown_node**: 处理未知请求

### Gradio界面
- **文档上传区**: 支持拖拽上传MD文件
- **聊天界面**: 实时对话交互
- **状态监控**: 显示系统运行状态

## 🎯 使用示例

### 上传文档
1. 点击"上传Markdown文档"按钮
2. 选择.md文件
3. 系统自动处理并显示预览

### 智能问答
- **保存操作**: "请保存这个文档"
- **检索操作**: "文档中提到了什么？"
- **一般问答**: "请总结文档的主要内容"

## ⚙️ 配置说明

### 环境变量
- `DASHSCOPE_API_KEY`: 阿里云DashScope API密钥
- `DASH_MODEL_NAME`: 使用的模型名称（如qwen-plus）
- `DASHSCOPE_BASE_URL`: API基础URL

### 可调参数
- 文档分块大小: 默认1000字符
- 检索文档数量: 默认3个
- 服务器端口: 默认7860

## 🛠️ 开发指南

### 扩展功能
1. **添加新的文档格式**: 修改`file_processor.py`
2. **自定义路由逻辑**: 编辑`router_node`函数
3. **优化检索算法**: 调整相似度阈值和检索数量

### 调试模式
启动时添加调试参数：
```python
demo.launch(debug=True, show_error=True)
```

## 🔒 安全注意事项

- 不要在代码中硬编码API密钥
- 限制上传文件的大小和类型
- 定期更新依赖包版本

## 🐛 常见问题

### Q: 启动时提示缺少环境变量
A: 请按照上述说明正确设置环境变量

### Q: 文档上传失败
A: 检查文件格式是否为.md，文件大小是否合理

### Q: 检索结果不准确
A: 尝试调整文档分块大小或检索参数

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

💡 **提示**: 如果遇到问题，请检查环境变量配置和依赖包安装情况。