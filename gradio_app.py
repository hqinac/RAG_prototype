import os
import gradio as gr
import signal
import sys
import atexit
import gc
import threading
import time
import json
import  asyncio
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# 导入现有的模块
from main import graph, RouterState
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class HeartbeatMonitor:
    """心跳监控器，用于检测浏览器是否关闭"""
    
    def __init__(self, timeout=60, initial_delay=10):
        self.last_heartbeat = time.time()
        self.timeout = timeout  # 60秒超时，与页面检测保持一致
        self.initial_delay = initial_delay  # 10秒初始延迟
        self.is_running = True
        self.monitor_thread = None
        self.started = False
        
    def start_monitoring(self):
        """开始监控心跳"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print(f"🔄 心跳监控已启动，初始延迟{self.initial_delay}秒，超时时间{self.timeout}秒")
    
    def update_heartbeat(self):
        """更新心跳时间"""
        self.last_heartbeat = time.time()
        if not self.started:
            self.started = True
            print("首次心跳接收成功，监控正式开始")
        return "heartbeat_ok"
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        
    def _monitor_loop(self):
        """监控循环"""
        # 初始延迟，等待浏览器页面完全加载
        print(f"心跳监控等待{self.initial_delay}秒后开始...")
        time.sleep(self.initial_delay)
        
        while self.is_running:
            time.sleep(3)  # 每3秒检查一次
            
            # 如果还没有收到第一次心跳，继续等待
            if not self.started:
                continue
                
            # 检查心跳超时
            if time.time() - self.last_heartbeat > self.timeout:
                print("检测到浏览器连接丢失，程序即将退出...")
                self._force_exit()
                break
                
    def _force_exit(self):
        """强制退出程序"""
        try:
            # 在强制退出前，确保调用清理函数保存数据
            global rag_instance
            if rag_instance:
                print(" 浏览器连接丢失，正在保存数据...")
                rag_instance.cleanup_on_exit()
            
            print("RAG智能问答系统已关闭，感谢使用！")
            os._exit(0)
        except Exception as e:
            print(f"⚠️ 退出时出现错误: {e}")
            # 即使出错也要尝试保存数据
            try:
                if rag_instance:
                    rag_instance.cleanup_on_exit()
            except:
                pass
            os._exit(1)


# 全局心跳监控器
heartbeat_monitor = HeartbeatMonitor()


class RAGChatInterface:
    """RAG聊天界面处理类"""
    
    def __init__(self):
        """初始化RAG聊天接口"""
        URI = os.getenv("URI", "./saved_files")
        self.uploaded_documents = []  # 存储上传的文档信息
        self.chat_history = []        # 存储聊天历史
        self.doc_info = []
        self.temp_files = []
        self.doc_count=0
        self.doc_info_file = f"{URI}/doc_info.json"  # 本地存储文件路径
        
        # 从本地文件加载 doc_info
        self.load_doc_info()
        
        # 注册程序退出时的清理函数
        atexit.register(self.cleanup_on_exit)
    
    def cleanup_on_exit(self):
        """程序退出时的清理函数"""
        try:
            print("🧹 正在清理数据...")
            print(f"数据库中的文档为: {self.doc_info}")
            # 保存 doc_info 到本地文件
            self.save_doc_info()
            
            # 清空临时文件和上传文档
            self.clear_temp()
            
            # 清空聊天历史
            self.chat_history.clear()
            
            # 强制垃圾回收
            gc.collect()
            
            print("数据清理完成")
            
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
            self.temp_files.append(file_name)
            self.doc_count+=1
            
            return f"成功上传文件: {file_name}", f"{documents[0].page_content[:100]}..."
            
        except Exception as e:
            return f"文件处理失败: {str(e)}", ""
    
    def chat_with_rag(self, message: str, history: List[dict]) -> Tuple[List[dict], str]:
        """与RAG系统聊天"""
        if not message.strip():
            return history, ""
        
        try:
            # 调用图进行问答
            result = asyncio.run(graph.ainvoke({
                "input": message,
                "documents": self.uploaded_documents,
                "doc_info": self.temp_files,
                "doc_list": self.doc_info,
                "route": "",
                "knowledgebase": "",
                "RetrievalEvaluator":None,
                "query": "",
                "answer": "",
                "output": ""
            }))
            # 获取回答
            answer = result.get("output", "抱歉，我无法回答这个问题。")
            
            # 更新聊天历史 - 使用新的messages格式
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            self.chat_history = history
            self.doc_info = result.get("doc_list", [])
            print(self.doc_info)
            self.clear_temp()
            
            return history, ""
            
        except Exception as e:
            error_msg = f"处理消息时出错: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def clear_chat(self) -> Tuple[List[Dict], str]:
        """清空聊天历史"""
        self.chat_history = []
        gc.collect()  # 强制垃圾回收
        return [], "聊天记录已清空"
    
    def clear_documents(self) -> Tuple[str, str]:
        """清空已上传的文档"""
        try:
            # 使用 clear_temp() 清空文档列表
            self.clear_temp()
            # 注意：保留 doc_info 不被清空
            # 强制垃圾回收
            gc.collect()
            
            return "文档库已清空（文档信息已保留）", ""
            
        except Exception as e:
            return f"清空文档库时出错: {str(e)}", ""
    
    def clear_temp(self):
        """清空临时文件"""
        try:
            # 清空临时文件列表
            self.temp_files.clear()
            self.uploaded_documents.clear()
            return "临时文件已清空", ""
            
        except Exception as e:
            return f"清空临时文件时出错: {str(e)}", ""
    
    def save_doc_info(self):
        """保存 doc_info 到本地文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.doc_info_file), exist_ok=True)
            
            with open(self.doc_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.doc_info, f, ensure_ascii=False, indent=2)
            print(f" doc_info 已保存到 {self.doc_info_file}")
            print(f" 保存的文档信息: {self.doc_info}")
        except Exception as e:
            print(f" 保存 doc_info 失败: {e}")
            print(f" 尝试保存的文档信息: {self.doc_info}")
            print(f" 目标文件路径: {self.doc_info_file}")
    
    def load_doc_info(self):
        """从本地文件加载 doc_info"""
        try:
            if os.path.exists(self.doc_info_file):
                with open(self.doc_info_file, 'r', encoding='utf-8') as f:
                    self.doc_info = json.load(f)
                print(f" 从 {self.doc_info_file} 加载了 {len(self.doc_info)} 个文档信息")
            else:
                print(f" {self.doc_info_file} 不存在，使用空的 doc_info")
        except Exception as e:
            print(f" 加载 doc_info 失败: {e}，使用空的 doc_info")
            self.doc_info = []
    
    def get_system_status(self) -> str:
        """获取系统状态"""
        status = f"已上传文档块数: {self.doc_count}\n"
        status += f"知识库中文档列表：{self.doc_info}\n"
        status += f"对话轮次: {len(self.chat_history)}\n"
        status += f"系统状态: 正常运行"
        return status
    
    def heartbeat(self) -> str:
        """处理心跳请求"""
        return heartbeat_monitor.update_heartbeat()
    
    def shutdown_application(self) -> str:
        """关闭应用程序"""
        try:
            print(" 接收到浏览器关闭信号，正在关闭应用...")
            self.cleanup_on_exit()
            
            # 使用更强制的方式关闭程序
            import threading
            import time
            
            def force_shutdown():
                time.sleep(0.5)  # 给一点时间让响应返回
                print(" RAG智能问答系统已关闭，感谢使用！")
                os._exit(0)  # 强制退出
            
            # 在后台线程中执行关闭
            shutdown_thread = threading.Thread(target=force_shutdown, daemon=True)
            shutdown_thread.start()
            
            return "应用程序即将关闭..."
                
        except Exception as e:
            print(f"关闭应用时出错: {e}")
            # 即使出错也要尝试关闭
            import threading
            import time
            
            def emergency_shutdown():
                time.sleep(0.5)
                os._exit(1)
            
            threading.Thread(target=emergency_shutdown, daemon=True).start()
            return f"关闭失败，但程序将强制退出: {str(e)}"


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
    
    # JavaScript代码来监听浏览器关闭事件
    custom_js = """
    function() {
        console.log('🚀 RAG应用已启动，监听浏览器关闭事件...');
        
        let isClosing = false;
        let shutdownTriggered = false;
        let heartbeatInterval = null;
        
        // 启动心跳机制
        function startHeartbeat() {
            // 每5秒发送一次心跳
            heartbeatInterval = setInterval(() => {
                if (!isClosing) {
                    console.log('💓 发送心跳信号...');
                    // 查找心跳按钮并点击
                    const heartbeatBtn = document.querySelector('#heartbeat_trigger');
                    if (heartbeatBtn) {
                        heartbeatBtn.click();
                        console.log('✅ 心跳信号已发送');
                    } else {
                        console.log('❌ 未找到心跳按钮');
                    }
                }
            }, 5000);
            console.log('💓 心跳监控已启动，每5秒发送一次心跳');
        }
        
        // 停止心跳
        function stopHeartbeat() {
            if (heartbeatInterval) {
                clearInterval(heartbeatInterval);
                heartbeatInterval = null;
                console.log('💔 心跳监控已停止');
            }
        }
        
        // 触发关闭的函数
        function triggerShutdown() {
            if (shutdownTriggered) return;
            shutdownTriggered = true;
            
            console.log('🔄 触发应用关闭...');
            stopHeartbeat();
            
            // 查找隐藏的关闭按钮并点击
            const shutdownBtn = document.querySelector('#shutdown_trigger');
            
            if (shutdownBtn) {
                console.log('✅ 找到关闭按钮，触发点击事件');
                shutdownBtn.click();
            } else {
                console.log('❌ 未找到关闭按钮');
            }
        }
        
        // 监听页面卸载事件
        window.addEventListener('beforeunload', function(e) {
            if (!isClosing) {
                isClosing = true;
                console.log('🔄 检测到页面即将关闭，发送关闭信号...');
                triggerShutdown();
            }
        });
        
        // 监听页面可见性变化
        let visibilityTimer = null;
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'hidden' && !isClosing) {
                console.log('📱 页面被隐藏，可能是切换标签页或最小化窗口');
                
                // 清除之前的定时器
                if (visibilityTimer) {
                    clearTimeout(visibilityTimer);
                }
                
                // 延迟检查，如果页面持续隐藏较长时间则认为是关闭
                visibilityTimer = setTimeout(() => {
                    if (document.visibilityState === 'hidden' && !isClosing) {
                        isClosing = true;
                        console.log('✅ 页面长时间隐藏，确认为关闭，发送关闭信号');
                        triggerShutdown();
                    }
                }, 30000); // 增加到30秒检查，避免误判
            } else if (document.visibilityState === 'visible') {
                console.log('👁️ 页面重新可见，取消关闭检测');
                // 页面重新可见，取消关闭检测
                if (visibilityTimer) {
                    clearTimeout(visibilityTimer);
                    visibilityTimer = null;
                }
            }
        });
        
        // 监听窗口关闭事件
        window.addEventListener('unload', function() {
            if (!isClosing) {
                isClosing = true;
                console.log('🔄 检测到窗口关闭事件');
                triggerShutdown();
            }
        });
        
        // 启动心跳监控
        setTimeout(() => {
            console.log('🚀 准备启动心跳监控...');
            startHeartbeat();
        }, 1000); // 减少到1秒启动，确保尽快开始心跳
        
        // 添加手动关闭按钮
        function addCloseButton() {
            if (document.getElementById('manual-close-btn')) return;
            
            const closeBtn = document.createElement('button');
            closeBtn.id = 'manual-close-btn';
            closeBtn.innerHTML = '🔴 关闭应用';
            closeBtn.style.cssText = `
                position: fixed;
                top: 15px;
                right: 15px;
                z-index: 10000;
                background: linear-gradient(45deg, #ff4444, #cc0000);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            `;
            
            closeBtn.onmouseover = function() {
                closeBtn.style.transform = 'scale(1.1)';
                closeBtn.style.boxShadow = '0 6px 12px rgba(0,0,0,0.4)';
            };
            
            closeBtn.onmouseout = function() {
                closeBtn.style.transform = 'scale(1)';
                closeBtn.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';
            };
            
            closeBtn.onclick = function() {
                if (confirm('确定要关闭RAG智能问答系统吗？')) {
                    isClosing = true;
                    closeBtn.innerHTML = '正在关闭...';
                    closeBtn.disabled = true;
                    
                    console.log(' 用户手动触发关闭');
                    triggerShutdown();
                    
                    // 显示关闭消息
                    setTimeout(function() {
                        alert('应用程序即将关闭，感谢使用！');
                        window.close();
                    }, 500);
                }
            };
            
            document.body.appendChild(closeBtn);
            console.log('关闭按钮已添加到页面右上角');
        }
        
        // 等待页面完全加载后添加关闭按钮
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(addCloseButton, 1000);
            });
        } else {
            setTimeout(addCloseButton, 1000);
        }
    }
    """
    
    with gr.Blocks(css=custom_css, title="RAG智能问答系统", theme=gr.themes.Soft(), js=custom_js) as demo:
        
        # 隐藏的关闭按钮和状态显示
        shutdown_trigger = gr.Button("关闭应用", visible=False, elem_id="shutdown_trigger")
        shutdown_status = gr.Textbox(visible=False)
        
        # 隐藏的心跳按钮
        heartbeat_trigger = gr.Button("心跳", visible=False, elem_id="heartbeat_trigger")
        heartbeat_status = gr.Textbox(visible=False)
        
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
            fn=lambda: None,  # 重置文件上传组件
            outputs=[file_upload]
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
        
        # 关闭应用事件 - 接收JavaScript发送的关闭信号
        shutdown_trigger.click(
            fn=rag_instance.shutdown_application,
            outputs=[shutdown_status]
        )
        
        # 绑定心跳事件
        heartbeat_trigger.click(
            fn=rag_instance.heartbeat,
            outputs=heartbeat_status
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
        print("正在启动RAG智能问答系统...")
        print("界面将在浏览器中自动打开")
        print("可以点击页面右上角的红色按钮关闭应用")
        print("也可以按 Ctrl+C 安全关闭程序")
        print("关闭浏览器页面时程序会自动退出")
        print("心跳监控: 已启用，每5秒发送一次心跳，60秒超时")
        print("页面检测: 页面隐藏30秒后才认为是关闭，避免误判切换/最小化")
        print("初始延迟: 10秒后开始监控，确保页面完全加载")
        
        # 启动心跳监控
        heartbeat_monitor.start_monitoring()
        
        # 创建并启动界面
        demo = create_interface()
        
        # 启动应用
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7860,       # 端口号
            share=False,            # 是否创建公共链接
            debug=False,            # 关闭调试模式以减少输出
            show_error=True,        # 显示错误信息
            inbrowser=True,         # 自动在浏览器中打开
            prevent_thread_lock=False,  # 防止线程锁定
            quiet=True,             # 减少启动信息
            favicon_path=None,      # 不使用自定义图标
            auth=None               # 不使用身份验证
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