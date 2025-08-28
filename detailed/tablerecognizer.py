from dataclasses import dataclass
from nt import TMP_MAX
import os
from pickle import TRUE
import re 
import copy
import asyncio
#import tiktoken
from symtable import Class
from utils import outputtest_file, struct, fuzzy_match,check_unique,delete_addition_splits,merge_2chunk

from dotenv import load_dotenv
from bs4 import BeautifulSoup

import logging
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

#from cache_manager import get_embeddings, get_faiss, get_bm25, get_doc_cache, update_faiss_cache, update_bm25_cache, update_doc_cache

#用于测试chunks切分效果
from pathlib import Path
from langchain_community.document_loaders import TextLoader

#工具函数用的库
import Levenshtein

load_dotenv()

# 设置默认的向量数据库存储路径
URI = os.getenv("URI", "./saved_files")
HEAD_PATTERN = (
    ("cover", r"#.*公告.*"),
    ("notice", r"# 前言"),
    ("forehead", r"# 目次"),
    ("outlines_cn", r"(?i)# CONTENTS"),
    ("outlines_en", r"# \d+[\s]+[\u4e00-\u9fa5a-zA-Z]+")
)#存储正文前各部分的名字与各部分结尾的正则表达式（即下一部分的开头）
OUTLINE_PATTERN = ((r"\d+ [\u4e00-\u9fa5a-zA-Z0-9\s]+",r"(附|付|符)录[A-Za-z0-9] [\u4e00-\u9fa5a-zA-Z0-9\s]+", r"[\u4e00-\u9fa5a-zA-Z]", ),
                   (r"\d+\.\d+ [\u4e00-\u9fa5a-zA-Z0-9\s]+",),
                   (r"\d+\.\d+\.\d+ [\u4e00-\u9fa5a-zA-Z0-9\s]+",))


NOTICE_PATTERN = r"#.*\n公告\n"
FOREWORD_PATTERN = r"# 前言\n"
OUTLINES_CN_PATTERN = r"# 目次\n"
OUTLINES_EN_PATTERN = r"# CONTENTS\n"
CONTENT_PATTERN = r"[a-zA-Z]+\n+#? ?(?:\d+ )?[\u4e00-\u9fa5]+"


class Article:
    #读取正文前的内容，包括封面、公告、前言、目录、英文目录，识别目录，正文，分割附件。
    cover: Document
    notice: Document
    forehead: Document
    outlines_cn: Document
    outlines_en: Document
    outline: list[list[str]]
    content: Document
    additions: list['Article']
    base_splits: list[Document]


    def __init__(self, document,use_en = True):
        Start = 0
        lost_header = []
        for i,pattern in enumerate(HEAD_PATTERN):
            name,p = pattern
            match = re.search(p, document.page_content[Start:])
            if match is None:
                End = -1
            else:
                End = Start + match.start()
        
            if End == -1:
                #print(f"文件{name}部分结尾不存在。")
                empty_doc = Document(page_content="", metadata=document.metadata)
                empty_doc.metadata['type'] = name
                setattr(self, name, empty_doc)
                continue
            else:
                doc = Document(page_content=document.page_content[Start:End],metadata=document.metadata)
                doc.metadata['type'] = name
                setattr(self, name, doc)
                Start = End
        
        #处理有部分不存在的情况
        tmp_name = ""
        for i,pattern in enumerate(HEAD_PATTERN):
            if getattr(self, pattern[0]).page_content == "":
                if pattern[0] in lost_header:
                    continue
                if i ==len(HEAD_PATTERN)-1 or tmp_name != "":
                    lost_header.append(pattern[0])
                    continue
                if tmp_name == "":
                    tmp_name = HEAD_PATTERN[i][0]
                    continue
            elif tmp_name != "":
                # 直接使用属性访问而不是eval
                tmp_attr = getattr(self, tmp_name)
                pattern_attr = getattr(self, pattern[0])
                tmp_attr.page_content = pattern_attr.page_content
                #print(f"处理{tmp_name}完毕，目前{tmp_name}内容是{getattr(self, tmp_name).page_content}")
                pattern_attr.page_content = ''
                tmp_name = ""
                lost_header.append(pattern[0])

        if 'header' in document.metadata:
            file_name = document.metadata["source"]+'：'+document.metadata['header'][0]
        else:
            file_name = document.metadata['source']
        print(f"文件{file_name}处理完成,缺失{lost_header}部分")
        
        if self.outlines_en.page_content == "":
            use_en = False
        self.content = Document(page_content=document.page_content[Start:],metadata=document.metadata)
        self.additions = []  # 初始化additions属性
        self.outline = self.outline_recognize(use_en)

    def outline_recognize(self,use_en = True):
        outline = []
        cn = []
        en = []
        lines = self.outlines_cn.page_content.splitlines()
        enlines = self.outlines_en.page_content.splitlines()
        if self.outlines_cn.page_content == "":
            print("文件不存在目录")
            return outline
        if self.outlines_en.page_content == "":
            print("文件不存在英文目录")
            use_en = False

        for line in lines[1:]:
            line = line.strip()
            if line == "":
                continue
            match = re.search(r"…+",line)
            if match is None:
                match = re.search(r" \d+(?=$)",line)
                if match is None:
                    match = re.search(r"\.+",line)

            if match is not None:
                tail = match.start()
            else:
                tail = len(line)
            processed_line = line[:tail].strip()
            line = re.sub(r"^# +","",processed_line)
            # 确保处理后的行不为空且有实际内容
            if line and len(line) > 1:
                cn.append(line)

        i = 1
        while i<len(enlines): #因为英文标题会转行，需要处理转行问题
            line = enlines[i].strip()
            if line == "":
                i+=1
                continue
            match = re.search(r" \d+(?=$)",line)
            if match is not None:
                tail = match.start()
            else:
                #print("转行了")
                enlines[i] += enlines.pop(i+1)
                continue
            line = line[:tail].strip()
            line = re.sub(r"^# +","",line)
            # 检查是否与最后一个条目重复
            if len(en) > 0 and line == en[-1]:
                print(f"重复识别(最后一个): '{line}' == '{en[-1]}'")
                i+=1
                continue
            en.append(line)
            i+=1

        # 添加调试信息
        print(f"中文目录条目数量: {len(cn)}")
        print(f"英文目录条目数量: {len(en)}")
        print(f"中文目录前5条: {cn[:5]}")
        print(f"英文目录前5条: {en[:5]}")
        
        if not use_en or len(cn) != len(en):
            outline = [[line,"",[]] for line in cn]
        else:
            if len(cn) != len(en):
                print("中英文目录对照存在问题，请检查文件目录内容。")
                print(f"中文目录完整内容: {cn}")
                print(f"英文目录完整内容: {en}")
            for i in range(len(cn)):
                outline.append([cn[i],en[i],[]])

        depth = -1
        f =[] #记录父标题
        for i, t in enumerate(outline):
            #print(f"开始识别目录{t[0]}的结构")
            matched = False
            for j ,p in enumerate(OUTLINE_PATTERN):
                for k, OUTLINE in enumerate(p):
                    if re.match(OUTLINE, t[0]):
                        if j > depth:
                            #print(f"目录层级比现有层级高，为{j}")
                            t[2] = copy.deepcopy(f)
                            #print(f"父目录为{t[2]}")
                            f.append(t[:2])
                        else:
                            #print(f"切换到同级或低级目录，层级为{j}")
                            f = (f[:j] if j>0 else [])
                            t[2] = copy.deepcopy(f)
                            #print(f"父目录为{t[2]}")
                            f.append(t[:2])
                        depth = j
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                print(f"目录{t[0]}没有匹配到任何模式")
        return outline
    
    #删除误识别的页码
    def clear_splits(self):
        for split in self.base_splits:
            if bool(re.fullmatch(r"^-?\d+(\.\d+)?$", split.page_content)):
                self.base_splits.remove(split) 
            split.page_content=re.sub(r"^#+","",split.page_content)
            split.page_content=split.page_content.strip()
            if split.page_content == "":
                self.base_splits.remove(split)
        return self.base_splits




def mdfile_recognizer(document:Document, use_en = True, has_chunk_size = False, chunk_size = -1):

    # 抑制CharacterTextSplitter的警告输出
    logging.getLogger('langchain_text_splitters.base').setLevel(logging.ERROR)
    
    document.metadata["type"] = "text"
    base_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1,
        chunk_overlap=0
    )
    article = Article(document,use_en)
    
    #识别目录结构
    print("目录结构搭建完成，开始识别附件")
    for outline in article.outline[::-1]:
        if re.match(r"(附|付|符)：[\u4e00-\u9fa5a-zA-Z0-9]",outline[0]) : #附件
            match = re.search(outline[0][2:],article.content.page_content)
            if match is not None:
                print("读取到附件"+outline[0][2:]+"位置在"+article.content.page_content[match.start():match.start()+100])
                tail = match.start()
            else:
                print("未能读取附件"+outline[0])
                continue
            doc = Document(page_content=article.content.page_content[tail:],metadata=article.content.metadata)
            article.content.page_content = article.content.page_content[:tail]
            doc.metadata["header"] = outline
            article.additions.append(Article(doc,use_en))
            continue
        break
    article.additions = article.additions[::-1]
    
    #后续对totalsplits进行修改时会同时修改article跟additions的base_splits（直接引用）
    article.base_splits = base_splitter.split_documents([article.content])
    total_splits = article.clear_splits()
    for addition in article.additions:
        addition.base_splits = base_splitter.split_documents([addition.content])
        total_splits.extend(addition.clear_splits())

    output_content = []
    for chunk in total_splits:       
        # page_content
        output_content.append(chunk.page_content)
        output_content.append("")  # 空行
        # metadata信息
        metadata_lines = []
        for key, value in chunk.metadata.items():
            metadata_lines.append(f"{key}: {value}")
        output_content.extend(metadata_lines)
        output_content.append("")  # 空行
    with open("basesplittest.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))
    
    figures = check_figures(total_splits, article)
    outputtest_file(figures,"figures.md")
    equations = check_equations(total_splits, article)
    outputtest_file(equations,"equations.md")
    tables = check_table(total_splits, article)
    outputtest_file(tables,"tables.md")

    outputtest_file(total_splits,"total.md")
    for addition in article.additions:
        outputtest_file(addition.base_splits,"addition.md")

    if has_chunk_size:
        chunks = merge_size_chunk(article, chunk_size)
    else:
        chunks = merge_chunk(article, total_splits)
    return chunks



def check_figures(base_splits, article=None):
    figures = []
    i = 0
    while i < len(base_splits):
        if base_splits[i].page_content.startswith("!["):
            #print("开始处理图片"+base_splits[i+1].page_content)
            base_splits[i].metadata["type"] = "figure"
            image_link = re.search(r"images/([\s\S]*?)\.jpg",base_splits[i].page_content).group()
            image_name = re.search(r"^图[\s　]+([^\n\r]*)",base_splits[i+1].page_content)
            if image_name:
                base_splits[i].metadata["image_name"] = image_name.group()
            base_splits[i].metadata["image_link"] = image_link
            base_splits[i].page_content = base_splits[i].page_content+'\n'+image_name.group()
            
            # 获取要删除的图片标题内容
            title_to_remove = base_splits[i+1].page_content
            base_splits.pop(i+1)
            # 如果传入了article对象，同时从各个附件的base_splits中删除对应的图片标题
            if article:
                for addition in article.additions:
                    deleted = delete_addition_splits(title_to_remove, addition)
                    if deleted:
                        break
                print(f"未在附件中找到图片标题：{title_to_remove}")
            figures.append(copy.deepcopy(base_splits[i]))
        i+=1
    return figures


def check_equations(base_splits, article=None):
    equations = []
    i = 0
    while i<len(base_splits):
        if base_splits[i].page_content.startswith("$$"):
            #print("开始处理公式"+base_splits[i-1].page_content)
            base_splits[i].metadata["type"] = "equation"
            if base_splits[i-1].page_content.endswith("：") or base_splits[i-1].page_content.endswith(":"):
                base_splits[i].metadata["equation_name"] = base_splits[i-1].page_content
                base_splits[i].page_content = base_splits[i-1].page_content +"\n\n"+ base_splits[i].page_content
                if article:
                    for addition in article.additions:
                        deleted = delete_addition_splits(base_splits[i-1].page_content, addition)
                        if deleted:
                            break
                    print(f"未在附件中找到图片标题：{base_splits[i-1].page_content}")
                base_splits.pop(i-1)
                i-=1
            struct_equation(base_splits,i)
            while True:
                if i+1 >= len(base_splits):
                    break
                if re.match(r"(\$?)([\s\S]*?)(\$?)\s*——\s*(.+)",base_splits[i+1].page_content) \
                    or base_splits[i+1].page_content.startswith("$$")\
                    or re.match(r"^(注|主)：",base_splits[i+1].page_content):
                    if base_splits[i+1].page_content.startswith("$$"):
                        struct_equation(base_splits,i+1)
                        base_splits[i].page_content = base_splits[i].page_content +"\n"+ base_splits[i+1].page_content
                    else:
                        base_splits[i].page_content = base_splits[i].page_content +"\n"+ base_splits[i+1].page_content
                    if article:
                        for addition in article.additions:
                            deleted = delete_addition_splits(base_splits[i+1].page_content, addition)
                            if deleted:
                                break
                        print(f"未在附件中找到图片标题：{base_splits[i+1].page_content}")
                    base_splits.pop(i+1)
                else:
                    break
            equations.append(copy.deepcopy(base_splits[i]))
        i += 1
    return equations

def struct_equation(base_splits,i):
    while True:
        tmp = base_splits.pop(i+1)
        base_splits[i].page_content = base_splits[i].page_content +"\n"+ tmp.page_content
        if tmp.page_content == "$$":
            base_splits[i].page_content = base_splits[i].page_content +"\n"
            break

def check_table(base_splits, article=None):
    '''
    识别表的标题、备注、续表，进行合成
    '''
    tables = []
    i=0
    hasheader = False
    hastail=False
    while i < len(base_splits):
        if base_splits[i].page_content.startswith("<table>"):
            base_splits[i].metadata["type"] = "table"
            uptitle = min (i-5,-1)
            for j in range(i-1, uptitle, -1):
                if re.match(r"^表 ",base_splits[j].page_content):
                    uptitle = j
                    hasheader = True    
                    break
            if hasheader:
                table_name = base_splits[uptitle].page_content
                base_splits[i].metadata["table_name"] = table_name
                while i > uptitle:
                    base_splits[i].page_content = base_splits[i-1].page_content + "\n\n" + base_splits[i].page_content
                    if article:
                        for addition in article.additions:
                            deleted = delete_addition_splits(base_splits[i-1].page_content, addition)
                            if deleted:
                                break
                        print(f"未在附件中找到图片标题：{base_splits[i-1].page_content}")
                    base_splits.pop(i-1)
                    i -= 1
                hasheader = False
            while True:
                if i+1 >= len(base_splits):
                    break
                if re.match(r"^续表 ",base_splits[i+1].page_content):
                    if re.match(r"^<table>",base_splits[i+2].page_content):
                        base_splits[i].page_content = merge_table(base_splits[i].page_content,base_splits[i+2].page_content)
                        for j in range(i+2,i,-1):
                            if article:
                                for addition in article.additions:
                                    deleted = delete_addition_splits(base_splits[j].page_content, addition)
                                    if deleted:
                                        break
                                print(f"未在附件中找到图片标题：{base_splits[j].page_content}")
                            base_splits.pop(j)
                    else: 
                        print(table_name + "的续表格式错误或不存在,请检查原文档")
                    continue
                if re.match(r"^(注|主)：",base_splits[i+1].page_content):
                    base_splits[i].page_content = base_splits[i].page_content + "\n\n" + base_splits[i+1].page_content
                    if article:
                        for addition in article.additions:
                            deleted = delete_addition_splits(base_splits[i+1].page_content, addition)
                            if deleted:
                                break
                        print(f"未在附件中找到图片标题：{base_splits[i+1].page_content}")
                    base_splits.pop(i+1)
                    #hastail=True
                    continue
                '''
                原本用于处理注的1234点被分开的情况，但似乎ocr能够将注的分点识别到同一行，且会识别到紧接的正文数字分点。
                if hastail and re.match(r"^\d ",base_splits[i+1].page_content):
                    base_splits[i].page_content = base_splits[i].page_content + "\n\n" + base_splits[i+1].page_content
                    base_splits.pop(i+1)
                    continue
                '''
                break
            tables.append(copy.deepcopy(base_splits[i]))
        i += 1
    return tables


def merge_table(table1:str,table2:str):
    '''
    将续表合并，去掉重合的表头
    '''
    s = re.search("<table>", table1).start()
    e = re.search("</table>", table1).end()
    header = table1[:s] #表格标题
    tail = table1[e:]
    puretable1 = table1[s:e]
    soup1 = BeautifulSoup(puretable1, 'html.parser')
    soup2 = BeautifulSoup(table2, 'html.parser')
    table1 = soup1.find('table')
    table2 = soup2.find('table')
    rows1 = table1.find_all('tr')
    rows2 = table2.find_all('tr')
    def get_row_content(row):
        cells = row.find_all(['td', 'th'])  # 同时查找td和th
        return [cell.get_text(separator="").strip() for cell in cells]
    l = min(len(rows1),len(rows2))
    for i in range(l):
        if get_row_content(rows1[i]) != get_row_content(rows2[i]):
            # 正确地将rows2中从第i行开始的行添加到table1中
            for row in rows2[i:]:
                table1.append(row)
            break
    else:
        if len(rows2) > l:
            for row in rows2[l:]:
                table1.append(row)
    ntable = header + str(table1) + tail
    return ntable

def merge_chunk_through_outlines(article:Article,total_splits):
    '''
    仅根据标题结构合并单一目录结构的文档（不含附件）
    '''
    chunks = []
    threshold = 0.6
    i=0
    j = len(total_splits)
    for num, outline in enumerate(article.outline[:len(article.outline)-len(article.additions)]):
        #处理有数字编号的标题（子标题可能含有x.x.x)
        print(f"开始处理标题{outline[0]}")
        parent_match = re.match(r"^(\d+\.\d+) ", outline[0])
        if not parent_match:
            parent_match = re.match(r"^(\d+) [\u4e00-\u9fa5a-zA-Z0-9]+", outline[0])

        while i < j:
            if num < len(article.outline)-1:
                #print(f"DEBUG fuzzy_match调用: outline[0]={repr(article.outline[num+1][0])}, total_splits[i].page_content前50字符={repr(total_splits[i].page_content[:50])}")
                isoutline, outlineend = fuzzy_match(article.outline[num+1][0],total_splits[i].page_content,threshold)
                #print(f"DEBUG fuzzy_match结果: isoutline={isoutline}, outlineend={outlineend}")
                if isoutline:#切到目录中的下一条
                    print(f"切到下一条{article.outline[num+1][0]}，当前位置为{total_splits[i].page_content}")
                    newoutline = struct([article.outline[num+1][:2]])
                    if outline[:2] in article.outline[num+1][2]: #父标题切到子标题的情况
                        if chunks[-1].metadata["start"] == i-1: #处理父标题子标题相邻的情况
                            if chunks[-1].metadata["outlineend"] == len(chunks[-1].page_content):#父标题紧接子标题
                                chunks[-1].page_content = chunks[-1].page_content + newoutline + total_splits[i].page_content[outlineend:]
                                chunks[-1].metadata["outlineend"] += len(newoutline)
                            else: 
                                chunks[-1].page_content = chunks[-1].page_content + "\n" + newoutline + total_splits[i].page_content[outlineend:]
                                chunks[-1].metadata["outlineend"] += 1
                            chunks[-1].metadata["start"] = i
                            chunks[-1].metadata["outline"] = article.outline[num+1]
                            check_unique(chunks[-1],total_splits[i])
                            i+=1
                            break
                
                    fatheroutline = struct(chunks[-1].metadata["outline"][2])
                    chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                    chunks[-1].metadata["outlineend"] += len(fatheroutline)
                    chunks.append(copy.deepcopy(total_splits[i]))
                    init_chunk(chunks[-1])
                    chunks[-1].page_content = newoutline + total_splits[i].page_content[outlineend:]
                    chunks[-1].metadata["outline"] = article.outline[num+1]
                    chunks[-1].metadata["start"] = i
                    chunks[-1].metadata["outlineend"] = len(newoutline)
                    check_unique(chunks[-1], total_splits[i])#检查特殊格式，并将相关参数合并到chunks中
                    i+=1
                    break

            isoutline, outlineend = fuzzy_match(outline[0],total_splits[i].page_content,threshold)
            if isoutline: #处理文件开头
                if chunks!=[]: print("warning: 重复识别到同一标题"+outline[0]+"\n当前位置在："+total_splits[i-1].page_content+"\n\n"+total_splits[i].page_content+"\n\n"+total_splits[i+1].page_content)
                else:
                    chunks.append(copy.deepcopy(total_splits[i]))
                    newoutline = struct([outline[:2]])
                    remaining_content = total_splits[i].page_content[outlineend:].strip()
                    chunks[-1].page_content = newoutline + remaining_content
                    init_chunk(chunks[-1])
                    chunks[-1].metadata["outline"] = outline
                    chunks[-1].metadata["start"] = i
                    chunks[-1].metadata["outlineend"] = len(newoutline)
                    check_unique(chunks[-1], total_splits[i])
                    i += 1
                    continue

            
            '''
            切到目录条目的子标题 （目录条目中的父子条目已经在上面处理完了）
            结构为2.1->2.1.1或者2.1->(I)->2.1.1 或 1->1.0.1
            '''
            if parent_match:
                parent_num = parent_match.group(1)
                #print(f"parent_numw为{parent_num}")
                numeric_child = rf"^({re.escape(parent_num)}\.(?!0)\d+)\s*"
                tmp = re.match(numeric_child, total_splits[i].page_content)
                if not tmp:
                    #print(f"尝试匹配: '{numeric_child}' with '{total_splits[i].page_content[:20]}'")
                    tmp = re.match(rf"^({re.escape(parent_num)}\.0\.\d+)\s*", total_splits[i].page_content)
                if tmp: 
                    tmp = tmp.group(1)
                    #print(f"识别到子标题{tmp}")
                    '''
                    处理 2.1->(I)->正文->2.1.1的情况可能会出现问题
                    '''
                    if (chunks[-1].metadata["outline"][:2] == outline[:2] or re.match(r"^\([IVX\u2160-\u217F]+\)",chunks[-1].metadata["outline"][0])) and chunks[-1].metadata["start"] == i-1: #父子标题相邻
                        #print(f"{tmp}父子标题相邻")
                        if chunks[-1].metadata["outlineend"] == len(chunks[-1].page_content):#父标题紧接子标题\
                            #print("父标题紧接子标题")
                            #chunks[-1].metadata["outlineend"] += len(chunks[-1].metadata["outline"][0])
                            chunks[-1].page_content = chunks[-1].page_content + total_splits[i].page_content
                            chunks[-1].metadata["start"] = i
                            #print(f"目前outlineend为{chunks[-1].metadata['outlineend']}，切片长度为{len(chunks[-1].page_content)}")
                        else:
                            chunks[-1].page_content = chunks[-1].page_content+ "\n"+ total_splits[i].page_content
                            chunks[-1].metadata["outlineend"] += 1
                        chunks[-1].metadata["outline"] = [tmp,"",chunks[-1].metadata["outline"][2] + [chunks[-1].metadata["outline"][:2]] ]
                        #chunks[-1].metadata["outlineend"] += len(tmp)
                        check_unique(chunks[-1], total_splits[i])
                    else:
                        #print("创建新块"+tmp)
                        #print(chunks[-1].metadata["outline"][2])
                        #print([chunks[-1].metadata["outline"][:2]])if "hasIVX" in chunks[-2].metadata:
                        fatheroutline = struct(chunks[-1].metadata["outline"][2])
                        chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                        chunks[-1].metadata["outlineend"] += len(fatheroutline)
                        chunks.append(copy.deepcopy(total_splits[i]))
                        init_chunk(chunks[-1])
                        new_outline = [tmp,"",chunks[-2].metadata["outline"][2]]
                        chunks[-1].metadata["outline"] = new_outline
                        chunks[-1].metadata["start"] = i
                        chunks[-1].metadata["outlineend"] = 0
                        check_unique(chunks[-1], total_splits[i])
                    i += 1
                    continue

            tmp = re.match(r"^\([IVX\u2160-\u217F]+\) .*?(?=\n|$)",total_splits[i].page_content)
            if tmp:
                print(f"识别到{tmp},原文为{total_splits[i].page_content}")
                tmp = tmp.group()
                print(chunks[-1].metadata["outline"])
                if chunks[-1].metadata["outline"][0] == outline[0] and chunks[-1].metadata["start"] == i-1: #父子标题相邻
                    #print(f"父子标题相邻，用于比较的长度分别为{chunks[-1].metadata["outlineend"]}， {len(chunks[-1].page_content)}")
                    #print(f"DEBUG: page_content repr={repr(chunks[-1].page_content)}")
                    if chunks[-1].metadata["outlineend"] == len(chunks[-1].page_content):#父标题紧接子标题
                        print("父标题紧接子标题")
                        chunks[-1].page_content = chunks[-1].page_content+ tmp + ":" + total_splits[i].page_content[len(tmp):]
                        chunks[-1].metadata["outlineend"] += len(tmp)
                        #print(f"存储被两层标题夹住的{tmp}到metadata中")
                        #chunks[-1].metadata["hasIVX"] = True #处理 x.x:(I):x.x.x... x.x.x的情况。
                    else:
                        chunks[-1].page_content = chunks[-1].page_content+ "\n"+ tmp + ":" + total_splits[i].page_content[len(tmp):]
                        chunks[-1].metadata["outlineend"] += 1
                    chunks[-1].metadata["outline"] = [tmp,"",chunks[-1].metadata["outline"][2] + [chunks[-1].metadata["outline"][:2]] ]
                    #chunks[-1].metadata["outlineend"] += len(tmp)
                    chunks[-1].metadata["start"] = i
                    check_unique(chunks[-1], total_splits[i])
                else:
                    fatheroutline = struct(chunks[-1].metadata["outline"][2])
                    chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                    chunks[-1].metadata["outlineend"] += len(fatheroutline)
                    chunks.append(copy.deepcopy(total_splits[i]))
                    init_chunk(chunks[-1])
                    if re.match(r"^\([IVX\u2160-\u217F]+\)",chunks[-2].metadata["outline"][0]):
                        new_outline = [tmp,"",chunks[-2].metadata["outline"][2]]
                    else:
                        new_outline = [tmp,"",chunks[-2].metadata["outline"][2][:-1]]
                    chunks[-1].page_content = tmp + ":" + total_splits[i].page_content[len(tmp):]
                    chunks[-1].metadata["outline"] = new_outline
                    chunks[-1].metadata["start"] = i
                    chunks[-1].metadata["outlineend"] = len(tmp)+1
                    check_unique(chunks[-1], total_splits[i])
                i += 1
                continue
            if chunks==[]: #开头必然是第一个标题，与之强行匹配
                chunks.append(copy.deepcopy(total_splits[i]))
                newoutline = struct([outline[2]])
                outlineend = 0
                chunks[-1].page_content = newoutline + total_splits[i].page_content[outlineend:]
                init_chunk(chunks[-1])
                chunks[-1].metadata["outline"] = outline
                chunks[-1].metadata["start"] = i
                chunks[-1].metadata["outlineend"] = outlineend
                check_unique(chunks[-1], total_splits[i])
                i += 1
                continue
            #不属于任何标题的内容，归入当前chunk
            chunks[-1].page_content = chunks[-1].page_content+"\n"+total_splits[i].page_content
            check_unique(chunks[-1], total_splits[i])
            i += 1
            continue
    return chunks


#存储特殊格式
def init_chunk(chunk):
    chunk.metadata["type"] = "chunk"
    chunk.metadata["has_table"] = False
    chunk.metadata["has_equation"] = False
    chunk.metadata["has_figure"] = False
    chunk.metadata["table_names"] = []
    chunk.metadata["equation_names"] = []
    chunk.metadata["figure_links"] = []
    chunk.metadata["figure_names"] = []
    chunk.metadata["outline"] = ["", "", []]

def add_description(add_chunks, base_chunks):
    j=0
    for i, chunk in enumerate(add_chunks):
        pure_chunk = chunk.page_content[chunk.metadata["outlineend"]:].strip()
        space_index = pure_chunk.find(' ')
        chunk_head = chunk.metadata["outline"][0].strip()
        if space_index != -1:
            pure_content = pure_chunk[space_index+1:]
            head = pure_chunk[:space_index]
            if chunk_head == "":
                chunk_head = re.match(r"^\d+\.\d+\.\d+*", head)
                if chunk_head:
                    chunk_head = chunk_head.group()
                else:
                    print(f"warning:无法识别附件说明切片的表头，目前切片为{chunk.page_content}")
                    continue
        else:
            head = re.match(r'^\d+\.\d+\.\d+(?:\s*[~\\~、]\s*\d+\.\d+\.\d+)*', pure_chunk)
            if not head and chunk_head == "":
                print(f"warning:无法识别附件说明切片的表头，目前切片为{chunk.page_content}")
                continue
            pure_content = pure_chunk[head.end():]
            head = head.group()
            if chunk_head == "":
                chunk_head = re.match(r"^\d+\.\d+\.\d+*", head)
                if chunk_head:
                    chunk_head = chunk_head.group()
                else:
                    print(f"warning:无法识别附件说明切片的表头，目前切片为{chunk.page_content}")
                    continue
        print(f"head为{head},原切片为{pure_chunk}")
        tmp = head.find('~')
        if tmp != -1:
            chunk_tail = head[tmp+1:].strip()
            insideHead = False
            while j<len(base_chunks):
                #print(base_chunks[j].metadata["outline"][0])
                if chunk_head in base_chunks[j].metadata["outline"][0] or chunk_head == base_chunks[j].metadata["outline"][0]:
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_content
                    merge_2chunk(base_chunks[j], chunk)
                    insideHead = True
                    j+=1
                    continue
                if chunk_tail in base_chunks[j].metadata["outline"][0] or chunk_head == base_chunks[j].metadata["outline"][0]:
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_content
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    insideHead = False
                    break
                if insideHead:
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_content
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    continue
                j+=1
            continue
        tmp = head.find('、')
        if tmp != -1:
            chunk_tail = head[tmp+1:].strip()
            while j<len(base_chunks):
                #print(base_chunks[j].metadata["outline"][0])
                if chunk_head in base_chunks[j].metadata["outline"][0] or chunk_head == base_chunks[j].metadata["outline"][0]:
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_content
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    continue
                if chunk_tail in base_chunks[j].metadata["outline"][0] or chunk_head == base_chunks[j].metadata["outline"][0]:
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_content
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    break
                j+=1
            continue
        else:
            while j<len(base_chunks):
                print(base_chunks[j].metadata["outline"][0])
                if chunk_head in base_chunks[j].metadata["outline"][0] or chunk_head == base_chunks[j].metadata["outline"][0]:
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_content
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    break
                j+=1
            
        if j == len(base_chunks):
            print(f"warning:无法找到条文说明{chunk_head}对应的正文，请检查切片")


        
            



def merge_chunk(article:Article,total_splits):
    '''
    处理包含附件的文档
    '''
    threshold = 0.8
    i=0
    b = sum(len(addition.base_splits)for addition in article.additions)
    base_chunks = merge_chunk_through_outlines(article, article.base_splits[:len(article.base_splits)-b])
    for i,chunk in enumerate(base_chunks):
        chunk.metadata["chunk_id"] = i
    for addition in article.additions:
        add_chunks = merge_chunk_through_outlines(addition, addition.base_splits)
        if fuzzy_match("附：条文说明",addition.content.metadata["header"][0],0.8):
            add_description(add_chunks, base_chunks)
        else:
            for i, chunk in enumerate(add_chunks):
                header = struct(chunk.metadata["header"][2])+struct([chunk.metadata["header"][:2]])
                chunk.page_content = header+chunk.page_content
                chunk.metadata["outlineend"] += len(header)
                chunk.metadata["chunk_id"] = i+len(base_chunks)
            base_chunks.extend(add_chunks)
    return base_chunks


'''
#chunk_size暂定为字符串长度
def merge_size_chunk(article:Article, total_splits, chunk_size):
    basic = merge_chunk(article, total_splits)
    sized_chunk = []
    for chunk in basic:
        if sized_chunk == [] or sized_chunk[-1].metadata["size"]>chunk_size:
            sized_chunk.append(copy.deepcopy(chunk))
            sized_chunk[-1].metadata["size"] = len(sized_chunk[-1])
            if "header" in chunk.metadata:
                sized_chunk.metadata["header"] = [chunk.metadata["header"]]
            else:
                sized_chunk.metadata["header"] = []
            continue
        outline = sized_chunk[-1].metadata["outline"]
        noutline = chunk.metadata["outline"]
        tmp_head = ""
        for f in range(0,min(len(outline[2]),len(noutline[2]))):
            if outline[2][f] not in noutline[2]:
                if f > 0:
                    tmphead += struct(outline[2][:f])
                if outline[:2] in noutline[2]:
                    tmphead += struct([outline[:2]])
                if "header" in chunk.metadata:
                    header = struct(chunk.metadata["header"][2])+struct([chunk.metadata["header"][:2]])
                    if chunk.metadata["header"] in sized_chunk.metadata["header"]:
                        npagecontent = chunk.page_content[len(header)+len(tmphead):]
                    else:
                        npagecontent = header + chunk.page_content[len(header)+len(tmphead):]
                    if sized_chunk.metadata["size"] < chunk_size/10:
'''                        



if __name__ == "__main__":
    file_path = r"GB50023-2009：建筑抗震鉴定标准.md"
    # 使用TextLoader保持原始markdown格式
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    file_name = Path(file_path).name               
    documents[0].metadata['source'] = file_name
    chunks = mdfile_recognizer(documents[0])
    output_content = []
    for chunk in chunks:       
        # page_content
        output_content.append(chunk.page_content)
        output_content.append("")  # 空行
        # metadata信息
        metadata_lines = []
        for key, value in chunk.metadata.items():
            metadata_lines.append(f"{key}: {value}")
        output_content.extend(metadata_lines)
        output_content.append("")  # 空行
    with open("chunktest.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))