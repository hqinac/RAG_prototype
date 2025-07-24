import os
import gradio as gr
import signal
import sys
import atexit
import gc
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# 导入现有的模块
from main import graph, RouterState
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGChatInterface:
    """RAG聊天界面处理类"""
    
    def __init__(self):
        """初始化RAG聊天接口"""
        self.uploaded_documents = []  # 存储上传的文档信息
        self.chat_history = []        # 存储聊天历史
        self.doc_info = []
        
        # 注册程序退出时的清理函数
        atexit.register(self.cleanup_on_exit)
    
    def cleanup_on_exit(self):
        """程序退出时的清理函数"""
        try:
            print("🧹 正在清理数据...")
            
            # 清空上传文档列表
            self.uploaded_documents.clear()
            
            # 清空聊天历史
            self.chat_history.clear()

            
            # 强制垃圾回收
            gc.collect()
            
            print("✅ 数据清理完成")
            
        except Exception as e:
            print(f"⚠️ 清理过程中出现错误: {e}")
    
    def process_uploaded_file(self, file_path: str) -> Tuple[str, str]:
        """
        处理上传的Markdown文件
        """
        try:
            # 检查文件类型
            if file_path and not file_path.lower().endswith('.md'):
                return "文件类型错误，请上传Markdown(.md)文件", ""
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return "❌ 文件不存在", ""
            
            # 获取文件信息
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # 检查文件大小（限制为10MB）
            if file_size > 10 * 1024 * 1024:
                return "❌ 文件大小超过10MB限制", ""
            
            # 读取文件内容
            loader = UnstructuredMarkdownLoader(file_path, encoding='utf-8')
            documents = loader.load()
            file_name = Path(file_path).name
            documents[0].metadata['source'] = file_name
            # 添加到文档列表
            self.uploaded_documents.extend(documents)
            self.doc_info.append(file_name)
            
            return f"成功上传文件: {file_name}, {documents[0].page_content[:100]} ,..."
            
        except Exception as e:
            return f"文件处理失败: {str(e)}", ""
    
    def chat_with_rag(self, message: str, history: List[dict]) -> Tuple[List[dict], str]:
        """与RAG系统聊天"""
        if not message.strip():
            return history, ""
        
        try:
            # 调用图进行问答
            result = graph.invoke({
                "input": message,
                "documents": self.uploaded_documents,
                "doc_info": self.doc_info,
                "route": "",
                "output": ""
            })
            
            # 获取回答
            answer = result.get("output", "抱歉，我无法回答这个问题。")
            
            # 更新聊天历史 - 使用新的messages格式
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            self.chat_history = history
            self.clear_documents()
            
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ 处理消息时出错: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def clear_chat(self) -> Tuple[List[Dict], str]:
        """清空聊天历史"""
        self.chat_history = []
        gc.collect()  # 强制垃圾回收
        return [], "✅ 聊天记录已清空"
    
    def clear_documents(self) -> Tuple[str, str]:
        """清空已上传的文档"""
        try:
            # 清空文档列表
            self.uploaded_documents.clear()
            self.doc_info.clear()
            # 强制垃圾回收
            gc.collect()
            
            return "文档库已清空", ""
            
        except Exception as e:
            return f"清空文档库时出错: {str(e)}", ""
    
    def get_system_status(self) -> str:
        """获取系统状态"""
        doc_count = len(self.uploaded_documents)
        status = f"已上传文档块数: {doc_count}\n"
        status += f"对话轮次: {len(self.chat_history)}\n"
        status += f"系统状态: 正常运行"
        return status


def create_interface():
    """创建Gradio界面"""
    global rag_instance
    
    # 创建RAG聊天接口实例
    rag_instance = RAGChatInterface()
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-container {
        height: 500px !important;
        overflow-y: auto !important;
    }
    .upload-area {
        border: 2px dashed #ccc !important;
        border-radius: 10px !important;
        padding: 20px !important;
        text-align: center !important;
    }
    .status-box {
        background-color: #f0f0f0 !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin: 10px 0 !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="RAG智能问答系统", theme=gr.themes.Soft()) as demo:
        
        # 标题和描述
        gr.Markdown(
            """
            RAG智能问答系统！您可以上传Markdown文档搭建知识库，与AI助手进行对话对知识库进行检索。
            
            ## 使用说明：
            1. **上传文档**: 点击下方上传按钮，选择Markdown(.md)文件
            2. **开始对话**: 在聊天框中输入您的问题
            3. **智能检索**: 系统会自动在您的文档中搜索相关信息并回答
            """
        )
        
        with gr.Row():
            # 左侧：文档上传区域
            with gr.Column(scale=1):
                gr.Markdown("### 文档管理")
                
                file_upload = gr.File(
                    label="上传Markdown文档",
                    file_types=[".md"],
                    elem_classes=["upload-area"]
                )
                
                upload_status = gr.Textbox(
                    label="上传状态",
                    interactive=False,
                    lines=2
                )
                
                doc_preview = gr.Textbox(
                    label="文档预览",
                    interactive=False,
                    lines=8,
                    placeholder="上传文档后将显示预览..."
                )
                
                with gr.Row():
                    clear_docs_btn = gr.Button("🗑️ 清空文档库", variant="secondary")
                
                # 系统状态
                gr.Markdown("### 系统状态")
                system_status = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=4,
                    value=rag_instance.get_system_status()
                )
            
            # 右侧：聊天区域
            with gr.Column(scale=2):
                gr.Markdown("###  智能对话")
                
                chatbot = gr.Chatbot(
                    label="",
                    type="messages",
                    elem_classes=["chat-container"],
                    height=500,
                    placeholder=" 您好！我是RAG智能助手，您可以向我提问。"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="",
                        placeholder="请输入您的问题...",
                        scale=4,
                        lines=1
                    )
                    send_btn = gr.Button(" 发送", variant="primary", scale=1)
                
                with gr.Row():
                    clear_chat_btn = gr.Button(" 清空对话", variant="secondary")
                    refresh_status_btn = gr.Button(" 刷新状态", variant="secondary")
        
        # 事件绑定
        
        # 文件上传事件
        file_upload.upload(
            fn=rag_instance.process_uploaded_file,
            inputs=[file_upload],
            outputs=[upload_status, doc_preview]
        ).then(
            fn=rag_instance.get_system_status,
            outputs=[system_status]
        )
        
        # 发送消息事件
        send_btn.click(
            fn=rag_instance.chat_with_rag,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        ).then(
            fn=rag_instance.get_system_status,
            outputs=[system_status]
        )
        
        # 回车发送
        msg_input.submit(
            fn=rag_instance.chat_with_rag,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        ).then(
            fn=rag_instance.get_system_status,
            outputs=[system_status]
        )
        
        # 清空对话
        clear_chat_btn.click(
            fn=rag_instance.clear_chat,
            outputs=[chatbot, upload_status]
        )
        
        # 清空文档
        clear_docs_btn.click(
            fn=rag_instance.clear_documents,
            outputs=[upload_status, doc_preview]
        ).then(
            fn=rag_instance.get_system_status,
            outputs=[system_status]
        )
        
        # 刷新状态
        refresh_status_btn.click(
            fn=rag_instance.get_system_status,
            outputs=[system_status]
        )
        
        # 页面底部信息
        gr.Markdown(
            """
            ---
            **提示**: 
            - 支持上传多个Markdown文档
            - 系统会自动对文档进行分块处理
            - 可以询问文档中的任何内容
            - 支持多轮对话，保持上下文连贯性
            """
        )
    
    return demo


# 全局变量存储RAG实例
rag_instance = None

def signal_handler(signum, frame):
    """信号处理函数"""
    print(f"\n接收到信号 {signum}，准备关闭程序...")
    
    if rag_instance:
        rag_instance.cleanup_on_exit()
    
    print("程序已安全关闭")
    sys.exit(0)

def main():
    """主函数"""
    global rag_instance
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    # 检查环境变量
    required_env_vars = ["DASHSCOPE_API_KEY", "DASH_MODEL_NAME"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请设置以下环境变量后重新运行:")
        for var in missing_vars:
            print(f"  export {var}=your_value")
        return
    
    try:
        print("界面将在浏览器中自动打开")
        print("按 Ctrl+C 可安全关闭程序")
        
        # 创建并启动界面
        demo = create_interface()
        
        # 启动应用
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7860,       # 端口号
            share=False,            # 是否创建公共链接
            debug=True,             # 调试模式
            show_error=True,        # 显示错误信息
            inbrowser=True          # 自动在浏览器中打开
        )
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
        if rag_instance:
            rag_instance.cleanup_on_exit()
    except Exception as e:
        print(f"程序运行出错: {e}")
        if rag_instance:
            rag_instance.cleanup_on_exit()
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()