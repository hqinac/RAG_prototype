#!/usr/bin/env python3
"""
RAG智能问答系统启动脚本
"""

import os
import sys
from pathlib import Path

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    required_env_vars = [
        "DASHSCOPE_API_KEY",
        "DASH_MODEL_NAME"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 缺少必要的环境变量:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n请设置环境变量后重新运行:")
        print("Windows (PowerShell):")
        for var in missing_vars:
            print(f"   $env:{var}='your_value'")
        print("\nLinux/Mac:")
        for var in missing_vars:
            print(f"   export {var}='your_value'")
        return False
    
    print("✅ 环境变量配置正确")
    return True

def check_dependencies():
    """检查依赖包"""
    print("📦 检查依赖包...")
    
    required_packages = [
        "gradio",
        "langchain",
        "langchain-community",
        "langchain-experimental",
        "langgraph",
        "faiss-cpu",
        "dashscope"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少必要的依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请安装缺少的包:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("依赖包检查通过")
    return True

def main():
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    print("\n启动应用...")
    
    # 导入并启动应用
    try:
        from gradio_app import main as app_main
        app_main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保 gradio_app.py 文件存在且正确")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()