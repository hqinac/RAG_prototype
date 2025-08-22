#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件上传功能，重现zip文件错误
"""

import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader

def test_markdown_loader():
    """测试UnstructuredMarkdownLoader处理不同文件的情况"""
    
    # 创建一个临时的markdown文件
    test_content = "# 测试文档\n\n这是一个测试markdown文档。\n\n## 章节1\n\n内容1\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        print(f"测试文件: {temp_file}")
        print(f"文件存在: {os.path.exists(temp_file)}")
        print(f"文件大小: {os.path.getsize(temp_file)} bytes")
        
        # 测试UnstructuredMarkdownLoader
        print("\n开始测试UnstructuredMarkdownLoader...")
        loader = UnstructuredMarkdownLoader(temp_file, encoding='utf-8')
        documents = loader.load()
        
        print(f"成功加载文档数量: {len(documents)}")
        if documents:
            print(f"文档内容预览: {documents[0].page_content[:100]}...")
            print(f"文档元数据: {documents[0].metadata}")
        
    except Exception as e:
        print(f"错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_non_markdown_file():
    """测试处理非markdown文件的情况"""
    
    # 创建一个临时的文本文件（非.md扩展名）
    test_content = "这是一个普通文本文件，不是markdown格式。"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        print(f"\n测试非markdown文件: {temp_file}")
        print(f"文件存在: {os.path.exists(temp_file)}")
        
        # 测试UnstructuredMarkdownLoader处理非markdown文件
        print("\n开始测试UnstructuredMarkdownLoader处理非markdown文件...")
        loader = UnstructuredMarkdownLoader(temp_file, encoding='utf-8')
        documents = loader.load()
        
        print(f"成功加载文档数量: {len(documents)}")
        
    except Exception as e:
        print(f"错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    print("=== 测试UnstructuredMarkdownLoader ===\n")
    
    test_markdown_loader()
    test_non_markdown_file()
    
    print("\n=== 测试完成 ===")