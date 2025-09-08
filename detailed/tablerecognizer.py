from dataclasses import dataclass
from nt import TMP_MAX
from pickle import TRUE
import re 
import copy
import asyncio
#import tiktoken
from symtable import Class
from turtle import title
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from .utils import outputtest_file, struct, fuzzy_match,check_unique,delete_addition_splits,merge_2chunk,extract_matching_parts
except ImportError:
    from utils import outputtest_file, struct, fuzzy_match,check_unique,delete_addition_splits,merge_2chunk,extract_matching_parts

from outline_recognizer import infer_hierarchy, extract_features, checkFirst
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
    ("forehead", r"#+ 目\s*(次|录)"),
    ("outlines_cn", r"(?i)#+\s*(CONTENTS|TABLE\s*OF\s*CONTENTS)"),
    ("outlines_en", r"#+[ \t]*(?!.*(?:\.{3,}|……))(?:[^ \t\n0-9]+|[^ \t\n]+.*[^ \t\n0-9]+)(?:[ \t]+(?!\d+(?:[ \t]*$))[^ \t\n]+)?[ \t]*\n")
    #("outlines_en", r"#+ \d+[\s]+[\u4e00-\u9fa5a-zA-Z]+")
)#存储正文前各部分的名字与各部分结尾的正则表达式（即下一部分的开头）
OUTLINE_PATTERN = ((r"\d+ [\u4e00-\u9fa5a-zA-Z0-9\s]+",r"(附|付|符)录[A-Za-z0-9] [\u4e00-\u9fa5a-zA-Z0-9\s]+", r"[\u4e00-\u9fa5a-zA-Z]", ),
                   (r"\d+\.\d+ [\u4e00-\u9fa5a-zA-Z0-9\s]+",),
                   (r"^\([IVX\u2160-\u217F]+\) .*?(?=\n|$)",),
                   (r"^(\d+\.\d+\.\d+).+",))
#PARENT_DEPTH = 2 #目录在OUTLINE_PATTERN中的深度为[:2]
logging.basicConfig(level=logging.INFO,
                    filename='./detailed/recognizer.log',
                    filemode='w',
                    format='%(levelname)s - %(message)s',
                    encoding='utf-8')  

class Article:
    #读取正文前的内容，包括封面、公告、前言、目录、英文目录，识别目录，正文，分割附件。
    '''
    注意：变量中的outlines_cn其实是用于匹配正文中相同内容的目录，outlines_en是对主目录的翻译(有的文件会附上对中文目录的翻译)，use_en实际上是在生成目录时附上翻译
    目前的逻辑是：
    先正常读取中文目录outlines_cn与英文目录outlines_en(中文目录在英文目录前面),
    如果中英文目录都存在，就不改变位置，outlines_cn作为主目录，outlines_en作为翻译目录,并且use_en设置为True;
    如果只存在一种，就将其作为主目录，outline_en 设置为空的，use_en设置为False（在只有英文目录的情况下会将英文目录赋值给outlines_cn然后将outlines_en设置为空;
    如果都不存在，从正文提取标题结构作为目录。
    显然：该逻辑暂时无法处理二者都存在但需要英文目录作为主目录的情况或英文目录在中文目录前的情况。如果英文目录在中文目录前，只会读取到中文目录，并将英文目录作为forhead的一部分。
    '''
    cover: Document
    notice: Document
    forehead: Document
    outlines_cn: Document
    outlines_en: Document
    outline: list[list[str]]
    content: Document
    additions: list['Article']
    base_splits: list[Document]
    lost_header: list[str]
    pattern_tree: list
    dir_features: list


    def __init__(self, document,use_en = True):
        Start = 0
        lost_header = []
        for i,pattern in enumerate(HEAD_PATTERN):
            name,p = pattern
            logging.info(f"文件{name}部分开头为{document.page_content[Start:Start+100]}")
            match = re.search(p, document.page_content[Start:])
            if match is None:
                End = -1
            else:
                End = Start + match.start()
                logging.info(f"文件{name}部分结尾存在，位置为{match.group()}")
        
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
                if name == 'outlines_en':
                    Start = End
                else:
                    Start = Start + match.end()
        
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
        self.lost_header = lost_header
        logging.info(f"文件{file_name}处理完成,缺失{lost_header}部分")

        if 'outlines_cn' in lost_header and 'outlines_en' in lost_header:
            logging.info("文件不存在目录，直接从正文中提取目录结构")
            self.content = document
            self.forehead.page_content = ""
            self.lost_header.append('forehead')
            self.additions = []
            self.outline, self.pattern_tree,self.dir_features = self.default_outline_recognize() # 确保outline被初始化
        else:
            if self.outlines_en.page_content == "":
                use_en = False
            elif self.outlines_cn.page_content == "":
                use_en = False
                self.outlines_cn = self.outlines_en
                self.outlines_en = Document(page_content="",metadata=self.outlines_en.metadata)
            self.content = Document(page_content=document.page_content[Start:],metadata=document.metadata)
            self.additions = []  # 初始化additions属性
            self.outline, self.pattern_tree,self.dir_features = self.outline_recognize(use_en)
            
    
    def default_outline_recognize(self):
        outlineline = []
        outlines = []
        contentlines = self.content.page_content.splitlines()
        for line in contentlines:
            line = line.strip()
            if line == "":
                continue
            match = re.search(r"^#+",line)
            if match is None:
                continue
            else:
                line = re.sub(r"^#+ +","",line)
                outlineline.append(line)
        outline, pattern_tree,dir_features = infer_hierarchy(outlineline, useen=False)
        return outline, pattern_tree,dir_features



    def outline_recognize(self,use_en = True):
        outline = []
        cn = []
        en = []
        lines = self.outlines_cn.page_content.splitlines()
        enlines = self.outlines_en.page_content.splitlines()
        if self.outlines_cn.page_content == "":
            logging.info("文件不存在中文目录,目前只支持有中文目录的文件")
            return outline
        if self.outlines_en.page_content == "":
            logging.info("文件不存在英文目录")
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
            line = re.sub(r"^#+ +","",processed_line)
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
        #print(f"中文目录条目数量: {len(cn)}")
        #print(f"英文目录条目数量: {len(en)}")
        #print(f"中文目录前5条: {cn[:5]}")
        #print(f"英文目录前5条: {en[:5]}")
        outline, pattern_tree,dir_features = infer_hierarchy(cn, en, use_en)
        return outline, pattern_tree,dir_features
        '''
        if not use_en or len(cn) != len(en):
            outline = [[line,"",[]] for line in cn]
            if use_en and len(cn) != len(en):
                logging.warning("中英文目录对照存在问题，请检查文件目录内容。")
                logging.warning(f"中文目录完整内容: {cn}")
                logging.warning(f"英文目录完整内容: {en}")
        else:
            #if len(cn) != len(en):
                #print("中英文目录对照存在问题，请检查文件目录内容。")
                #print(f"中文目录完整内容: {cn}")
                #print(f"英文目录完整内容: {en}")
            for i in range(len(cn)):
                outline.append([cn[i],en[i],[]])

        maxdepth = 0
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
                        if depth > maxdepth:
                            maxdepth = depth
                        matched = True
                        break
                if matched:
                    break
            #if not matched:
                #print(f"目录{t[0]}没有匹配到任何模式")
        
        #PARENT_DEPTH = maxdepth+1
        return outline
        '''
    
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

    def addtree(self,depth, pattern):
        if depth >= len(self.pattern_tree):
            self.pattern_tree.extend([[] for _ in range(depth + 1 - len(self.pattern_tree))])
        self.pattern_tree[depth].append(pattern)


def mdfile_recognizer(document:Document, use_en = True, chunk_size = -1):

    # 抑制CharacterTextSplitter的警告输出
    logging.getLogger('langchain_text_splitters.base').setLevel(logging.ERROR)
    
    document.metadata["type"] = "text"
    base_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1,
        chunk_overlap=0
    )
    article = Article(document,use_en)
    print(f"目录完整内容: {article.outline}")
    #识别目录结构
    logging.info("正文目录结构搭建完成，开始识别附件")
    for outline in article.outline[::-1]:
        if re.match(r"(附|付|符)：[\u4e00-\u9fa5a-zA-Z0-9]",outline[0]) : #附件
            match = re.search(outline[0][2:],article.content.page_content)
            if match is not None:
                logging.info("读取到附件"+outline[0][2:]+"位置在"+article.content.page_content[match.start():match.start()+100])
                tail = match.start()
            else:
                logging.warning("未能读取附件"+outline[0])
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

    '''
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
    '''
    
    figures, equations, tables = [], [], []
    head_splits = []
    head_docs = []
    for head in HEAD_PATTERN[:1]:#不处理已经识别完的目录
        if head[0] not in article.lost_header:
            tmp_attr = getattr(article, head[0])
            head_docs.extend(tmp_attr)
            tmp_split = base_splitter.split_documents([tmp_attr])
            figures.extend(check_figures(tmp_split))
            equations.extend(check_equations(tmp_split))
            tables.extend(check_table(tmp_split))
            head_splits.extend(tmp_split)
            continue
    figures.extend(check_figures(total_splits, article))
    #outputtest_file(figures,"figures.md")
    equations.extend(check_equations(total_splits, article))
    #outputtest_file(equations,"equations.md")
    tables.extend(check_table(total_splits, article))
    outputtest_file(tables,"tables.md")

    #outputtest_file(total_splits,"total.md")
    #for addition in article.additions:
        #outputtest_file(addition.base_splits,"addition.md")

    if chunk_size != -1:
        head_chunks = simple_size_chunk(head_splits, chunk_size)
        #outputtest_file(head_chunks,"head.md")
        basic_chunks = merge_size_chunk(article, total_splits, chunk_size)
        #outputtest_file(basic_chunks,"basic.md")
        head_chunks.extend(basic_chunks)
        head_chunks.extend(figures)
        head_chunks.extend(equations)
        head_chunks.extend(tables)
        chunks = head_chunks
    else:
        head_docs.extend(merge_chunk(article, total_splits))
        head_docs.extend(figures)
        head_docs.extend(equations)
        head_docs.extend(tables)
        chunks = head_docs
    return chunks



def check_figures(base_splits, article=None):
    figures = []
    i = 0
    while i < len(base_splits):
        if base_splits[i].page_content.startswith("!["):
            #print("开始处理图片"+base_splits[i+1].page_content)
            base_splits[i].metadata["type"] = "figure"
            image_link = re.search(r"images/([\s\S]*?)\.jpg",base_splits[i].page_content).group()
            image_name = re.search(r"(?i)^(图|figure|image)\s*[^\s\n\r]*(?:\s+[^\n\r]*)?",base_splits[i+1].page_content)
            if image_name:
                base_splits[i].metadata["image_name"] = image_name.group()
            base_splits[i].metadata["image_link"] = image_link
            base_splits[i].page_content = base_splits[i].page_content+'\n'+image_name.group()
            
            # 获取要删除的图片标题内容
            title_to_remove = base_splits[i+1].page_content
            base_splits.pop(i+1)
            # 如果传入了article对象，同时从各个附件的base_splits中删除对应的图片标题
            deleted = False
            if article:
                for addition in article.additions:
                    deleted = delete_addition_splits(title_to_remove, addition)
                    if deleted:
                        break
                #if not deleted:
                    #print(f"未在附件中找到图片标题：{title_to_remove}")
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
                title_to_remove = base_splits[i-1].page_content
                base_splits.pop(i-1)
                if article and i+1>=len(base_splits)-sum(len(addition.base_splits) for addition in article.additions):
                    for addition in article.additions:
                        deleted = delete_addition_splits(title_to_remove, addition)
                        if deleted:
                            break
                    #print(f"未在附件中找到图片标题：{base_splits[i-1].page_content}")
                
                i-=1
            struct_equation(base_splits,i,article)
            while True:
                if i+1 >= len(base_splits):
                    break
                if re.match(r"(\$?)([\s\S]*?)(\$?)\s*——\s*(.+)",base_splits[i+1].page_content) \
                    or base_splits[i+1].page_content.startswith("$$")\
                    or re.match(r"^(注|主)：",base_splits[i+1].page_content):
                    if base_splits[i+1].page_content.startswith("$$"):
                        struct_equation(base_splits,i+1,article)
                        base_splits[i].page_content = base_splits[i].page_content +"\n"+ base_splits[i+1].page_content
                    else:
                        base_splits[i].page_content = base_splits[i].page_content +"\n"+ base_splits[i+1].page_content
                    
                        #print(f"未在附件中找到图片标题：{base_splits[i+1].page_content}")
                    title_to_remove = base_splits[i+1].page_content
                    base_splits.pop(i+1)
                    if article and i+1>=len(base_splits)-sum(len(addition.base_splits) for addition in article.additions):
                        for addition in article.additions:
                            deleted = delete_addition_splits(title_to_remove, addition)
                            if deleted:
                                break
                else:
                    break
            equations.append(copy.deepcopy(base_splits[i]))
        i += 1
    return equations

def struct_equation(base_splits,i,article=None):
    while True:
        title_to_remove = base_splits[i+1].page_content
        tmp = base_splits.pop(i+1)
        base_splits[i].page_content = base_splits[i].page_content +"\n"+ tmp.page_content
        if article and i+1>=len(base_splits)-sum(len(addition.base_splits) for addition in article.additions):
            for addition in article.additions:
                deleted = delete_addition_splits(title_to_remove, addition)
                if deleted:
                    break
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
    firstnote = None
    while i < len(base_splits):
        if base_splits[i].page_content.startswith("<table>"):
            base_splits[i].metadata["type"] = "table"
            uptitle = min (i-5,-1)
            for j in range(i-1, uptitle, -1):
                if re.match(r"(?i)^(表|table)\s*[^\s\n\r]*(?:\s+[^\n\r]*)?",base_splits[j].page_content):
                    uptitle = j
                    hasheader = True    
                    break
            if hasheader:
                table_name = base_splits[uptitle].page_content
                base_splits[i].metadata["table_name"] = table_name
                while i > uptitle:
                    base_splits[i].page_content = base_splits[i-1].page_content + "\n\n" + base_splits[i].page_content
                    title_to_remove = base_splits[i-1].page_content
                    base_splits.pop(i-1)
                    if article:
                        for addition in article.additions:
                            deleted = delete_addition_splits(title_to_remove, addition)
                            if deleted:
                                break
                    i -= 1
                hasheader = False
            while True:
                if i+1 >= len(base_splits):
                    break
                cont = re.match(r"^续表 ",base_splits[i+1].page_content)
                if not cont:
                    cont = base_splits[i+1].page_content.startswith(table_name)
                #if re.match(r"^续表 ",base_splits[i+1].page_content):
                if cont:
                    if re.match(r"^<table>",base_splits[i+2].page_content):
                        base_splits[i].page_content = merge_table(base_splits[i].page_content,base_splits[i+2].page_content)
                        for j in range(i+2,i,-1):
                            title_to_remove = base_splits[j].page_content
                                #print(f"未在附件中找到图片标题：{base_splits[j].page_content}")
                            base_splits.pop(j)
                            if article:
                                for addition in article.additions:
                                    deleted = delete_addition_splits(title_to_remove, addition)
                                    if deleted:
                                        break
                    else: 
                        logging.warning("表格\n"+ {base_splits[i+1].page_content} + "\n的续表格式错误或不存在,请检查原文档")
                    continue
                if re.match(r"^(注|主)：",base_splits[i+1].page_content):
                    base_splits[i].page_content = base_splits[i].page_content + "\n\n" + base_splits[i+1].page_content
                    title_to_remove = base_splits[i+1].page_content
                    firstnote = base_splits.pop(i+1)
                    if article:
                        for addition in article.additions:
                            deleted = delete_addition_splits(title_to_remove, addition)
                            if deleted:
                                break
                    #hastail=True
                    continue
                elif re.match(r"(?i)^Notes?",base_splits[i+1].page_content):
                    base_splits[i].page_content = base_splits[i].page_content + "\n\n" + base_splits[i+1].page_content
                    title_to_remove = base_splits[i+1].page_content
                    firstnote = base_splits.pop(i+1)
                    if article:
                        for addition in article.additions:
                            deleted = delete_addition_splits(title_to_remove, addition)
                            if deleted:
                                break
                    continue

                if firstnote and len(firstnote.page_content) <= 6:#处理注下面分点的情况
                    notetype = extract_features(base_splits[i+1].page_content)
                    if notetype['type'] == 'content':
                        logging.warning(f"表格备注识别错误，备注下的第一条是{base_splits[i+1].page_content}，如有需要，请检查文档，为备注加上序号或去掉错误备注再进行切片。")
                        break
                    if "parentthese" in features["type"]:
                        check = checkFirst(features['inside'])
                    else:
                        check = checkFirst(features['prefix'])
                    if not check:
                        logging.warning(f"表格备注识别错误，备注下的第一条是{base_splits[i+1].page_content}，如有需要，请检查文档，为备注加上序号或更改错误序号或去掉错误备注再进行切片。")
                        break
                    else:
                        base_splits[i].page_content = base_splits[i].page_content + "\n" + base_splits[i+1].page_content
                        title_to_remove = base_splits[i+1].page_content
                        if article and i+1>=len(base_splits)-sum(len(addition.base_splits) for addition in article.additions):
                            for addition in article.additions:
                                deleted = delete_addition_splits(title_to_remove, addition)
                                if deleted:
                                    break
                        lastcode = sum(ord(c) for c in notetype["prefix"])
                        while True:
                            tmp = extract_features(base_splits[i+1].page_content)
                            if tmp['type'] == notetype['type'] and tmp['dot_count'] == notetype['dot_count']:
                                tmpcode = sum(ord(c) for c in tmp["prefix"])
                                if tmpcode == lastcode + 1:
                                    base_splits[i].page_content = base_splits[i].page_content + "\n" + base_splits[i+1].page_content
                                    title_to_remove = base_splits[i+1].page_content
                                    base_splits.pop(i+1)
                                    if article:
                                        for addition in article.additions:
                                            deleted = delete_addition_splits(title_to_remove, addition)
                                            if deleted:
                                                break
                                    lastcode = tmpcode
                                    continue
                            firstnote = None
                            break
                '''
                原本用于处理注的1234点被分开的情况，但似乎ocr能够将注的分点识别到同一行，且会识别到紧接的正文数字分点。
                if hastail and re.match(r"^\\d ",base_splits[i+1].page_content):
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
    print(f"merge_chunk_through_outlines: article.outline 类型: {type(article.outline)}, 内容: {article.outline}")
    print(f"merge_chunk_through_outlines: article.additions 类型: {type(article.additions)}, 内容: {article.additions}")

    PARENT_DEPTH = len(article.pattern_tree)
    chunks = []
    threshold = 0.6
    i=0
    j = len(total_splits)
    ftmp = {}
    for num, outline in enumerate(article.outline[:len(article.outline)-len(article.additions)]):
        #处理有数字编号的标题（子标题可能含有x.x.x)
        logging.info(f"开始处理标题{outline[0]}")
        '''
        parent_match = re.match(r"^(\d+\.\d+) ", outline[0])
        if not parent_match:
            parent_match = re.match(r"^(\d+) [\u4e00-\u9fa5a-zA-Z0-9]+", outline[0])
        #print(f"标题数字部分为{parent_match.group(1)}")
        '''
        #处理可能被修正的标题结构
        outlinefeature = article.dir_features[num]
        t = outlinefeature['depth']
        for patterns in article.pattern_tree[outlinefeature['depth']:]:
            changed = False
            if [outlinefeature['type'],outlinefeature['dot_count']] not in patterns:
                t +=1
                changed = True
                continue
            else:
                break
        if changed:
            logging.info(f"标题{outline[0]}的目录结构与文档结构不一致，进行修正")
            n=0
            while n<t:
                if n >= len(outline[2]):
                    outline[2].append([])
                    n+=1
                    continue
                tmpfeature = extract_features(outline[2][n][0])
                if [tmpfeature['type'], tmpfeature['dot_count']] in pattern_tree[n]:
                    n+=1
                else:
                    while [tmpfeature['type'], tmpfeature['dot_count']] not in pattern_tree[n]:
                        outline[2].insert(n, [])
                        n+=1
            

        while i < j:
            tmp= extract_features(total_splits[i].page_content)
            if chunks!=[]:
                logging.info(f"目前ftmp为{ftmp},目录为{chunks[-1].metadata['outline']}")
            for k, patterns in enumerate(article.pattern_tree):
                if [tmp['type'],tmp['dot_count']] in patterns:
                    tmp['depth'] = k
                    break
            logging.info(f"当前处理的行为{tmp}")
            if num < len(article.outline)-1:
                #print(f"DEBUG fuzzy_match调用: outline[0]={repr(article.outline[num+1][0])}, total_splits[i].page_content前50字符={repr(total_splits[i].page_content[:50])}")
                isoutline, outlineend = fuzzy_match(article.outline[num+1][0],total_splits[i].page_content,threshold)
                #print(f"DEBUG fuzzy_match结果: isoutline={isoutline}, outlineend={outlineend}")
                if isoutline:#切到目录中的下一条
                    #print(f"切到下一条{article.outline[num+1][0]}，当前位置为{total_splits[i].page_content}")
                    tmp = article.dir_features[num+1]
                    newoutline = struct([article.outline[num+1][:2]])
                    if outline[:2] in article.outline[num+1][2]: #父标题切到子标题的情况
                        if chunks[-1].metadata["start"] == i-1: #处理父标题子标题相邻的情况
                            if chunks[-1].metadata["outlineend"] == len(chunks[-1].page_content):#父标题紧接子标题
                                chunks[-1].page_content = chunks[-1].page_content + newoutline + total_splits[i].page_content[outlineend:]
                                chunks[-1].metadata["outlineend"] += len(newoutline)
                                #如果不完全相邻直接创建新块
                            #else: 
                                #chunks[-1].page_content = chunks[-1].page_content + "\n" + newoutline + total_splits[i].page_content[outlineend:]
                                #chunks[-1].metadata["outlineend"] += 1
                                chunks[-1].metadata["start"] = i
                                chunks[-1].metadata["outline"] = article.outline[num+1]
                                check_unique(chunks[-1],total_splits[i])
                                i+=1
                                break
                    if chunks[-1].metadata["outline"][0] == outline[0]:
                        fatheroutline = struct(chunks[-1].metadata["outline"][2]) + struct([chunks[-1].metadata["outline"][:2]])
                        chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                        chunks[-1].metadata["outlineend"] = len(fatheroutline)
                    else:
                        fatheroutline = struct(chunks[-1].metadata["outline"][2])
                        chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                        chunks[-1].metadata["outlineend"] = len(fatheroutline)
                    chunks.append(copy.deepcopy(total_splits[i]))
                    init_chunk(chunks[-1])
                    chunks[-1].page_content = newoutline + total_splits[i].page_content[outlineend:]
                    chunks[-1].metadata["outline"] = article.outline[num+1]
                    chunks[-1].metadata["start"] = i
                    chunks[-1].metadata["outlineend"] = len(newoutline)
                    check_unique(chunks[-1], total_splits[i])#检查特殊格式，并将相关参数合并到chunks中
                    i+=1
                    ftmp = tmp
                    break

            isoutline, outlineend = fuzzy_match(outline[0],total_splits[i].page_content,threshold)
            if isoutline: #处理文件开头
                logging.debug(f"识别到可能的文件开头")
                if chunks!=[]: print("warning: 重复识别到同一标题"+outline[0]+"\n当前位置在："+total_splits[i-1].page_content+"\n\n"+total_splits[i].page_content+"\n\n"+total_splits[i+1].page_content)
                else:
                    tmp = article.dir_features[0]
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
                    ftmp = tmp
                    continue

            
            '''
            切到目录条目的子标题 （目录条目中的父子条目已经在上面处理完了）
            结构为2.1->2.1.1或者2.1->(I)->2.1.1 或 1->1.0.1
            '''
            '''
            if parent_match:
                parent_num = parent_match.group(1)
                numeric_child = rf"^({re.escape(parent_num)}\.(?!0)\d+)\s*"
                #print(f"parent_numw为{parent_num},尝试匹配: '{numeric_child}' with '{total_splits[i].page_content[:20]}'")
            '''
            #如果分点标题只有序号，没有题目，序号将被保留在outlineend之后，否则完整标题将被保留在outlineend之前
            
            #if not tmp:
                #tmp = re.match(rf"^({re.escape(parent_num)}\.0\.\d+)\s*", total_splits[i].page_content)
            if tmp['type'] != 'content': 
                #tmp = tmp.group(1)
                logging.debug(f"识别到子标题{tmp}")
                logging.debug(f"当前最后一个切片为{chunks[-1]}")
                if tmp['type'] in ['appendix','annex']: #用于处理default_outline_recognizer中没读取到的附件格式
                    if ftmp["head"] == ftmp["content"]:
                        fatheroutline = struct(chunks[-1].metadata["outline"][2])
                    else:
                        fatheroutline = struct(chunks[-1].metadata["outline"][2]) + struct([chunks[-1].metadata["outline"][:2]])
                    chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                    chunks[-1].metadata["outlineend"] = len(fatheroutline)
                    chunks.append(copy.deepcopy(total_splits[i]))
                    init_chunk(chunks[-1])
                    new_outline = [tmp['content'],"",[]]
                    chunks[-1].metadata["outline"] = new_outline
                    chunks[-1].metadata["start"] = i
                    chunks[-1].metadata["outlineend"] = len(tmp['content'])
                    check_unique(chunks[-1], total_splits[i])
                    if not tmp['depth']:
                        article.pattern_tree[0].append([tmp['type'],tmp['dot_count']])
                        tmp['depth'] = 0
                    i += 1
                    ftmp = tmp
                    continue


                def addChildOutline():
                    #用于处理父标题切到子标题情况的内部函数。如果父子标题完全相邻，将子标题加入父标题的块，如果不完全相邻，创建新块
                    nonlocal i, ftmp
                    if chunks[-1].metadata["outlineend"] == len(chunks[-1].page_content):#父标题紧接子标题
                        chunks[-1].page_content = chunks[-1].page_content + total_splits[i].page_content
                        chunks[-1].metadata["start"] = i
                        #print(f"目前outlineend为{chunks[-1].metadata['outlineend']}，切片长度为{len(chunks[-1].page_content)}")
                        chunks[-1].metadata["outline"] = [tmp["content"],"",chunks[-1].metadata["outline"][2] + [chunks[-1].metadata["outline"][:2]] + [[]]*(tmp['depth']-ftmp['depth']-1)]
                        check_unique(chunks[-1], total_splits[i])
                        i += 1
                        ftmp = tmp
                    else:
                        nfather = chunks[-1].metadata["outline"][2]+[chunks[-1].metadata["outline"][:2]]+ [[]]*(tmp['depth']-ftmp['depth']-1)
                        if ftmp["head"] == ftmp["content"]:
                            fatheroutline = struct(chunks[-1].metadata["outline"][2])
                        else:
                            fatheroutline = struct(chunks[-1].metadata["outline"][2]) + struct([chunks[-1].metadata["outline"][:2]])
                        chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                        chunks[-1].metadata["outlineend"] = len(fatheroutline)
                        chunks.append(copy.deepcopy(total_splits[i]))
                        init_chunk(chunks[-1])
                        new_outline = [tmp['content'],"",nfather]
                        chunks[-1].metadata["outline"] = new_outline
                        chunks[-1].metadata["start"] = i
                        if tmp['head'] == tmp['content']:
                            chunks[-1].metadata["outlineend"] = 0
                            if tmp['prefix']!=tmp['head']:
                                chunks[-1].metadata['outline'][0] = tmp['prefix']
                            #处理条文说明中1.0.1~1.0.7之类的情况，将目录序号存储为第一个。
                        else:
                            chunks[-1].metadata["outlineend"] = len(tmp['content'])
                            check_unique(chunks[-1], total_splits[i])
                        i += 1
                        ftmp = tmp
                
                def addParentOutline():
                    #用于处理子标题切到父标题情况的内部函数。创建新块
                    nonlocal i, ftmp
                    new_outline = [tmp['content'],"",chunks[-1].metadata["outline"][2][:tmp['depth']+1]]
                    if ftmp["head"] == ftmp["content"]:
                        fatheroutline = struct(chunks[-1].metadata["outline"][2])
                    else:
                        fatheroutline = struct(chunks[-1].metadata["outline"][2]) + struct([chunks[-1].metadata["outline"][:2]])
                    chunks[-1].page_content = fatheroutline + chunks[-1].page_content[chunks[-1].metadata["outlineend"]:]
                    chunks[-1].metadata["outlineend"] = len(fatheroutline)
                    chunks.append(copy.deepcopy(total_splits[i]))
                    init_chunk(chunks[-1])
                    chunks[-1].metadata["outline"] = new_outline
                    chunks[-1].metadata["start"] = i
                    if tmp['head'] == tmp['content']:
                        chunks[-1].metadata["outlineend"] = 0
                        if tmp['prefix']!=tmp['head']:
                            chunks[-1].metadata['outline'][0] = tmp['prefix']
                            #处理条文说明中1.0.1~1.0.7之类的情况，将目录序号存储为第一个。
                    else:
                        chunks[-1].metadata["outlineend"] = len(tmp['content'])
                    check_unique(chunks[-1], total_splits[i])
                    i += 1
                    ftmp = tmp



                #处理相邻目录标题类型相同的情况（例如A.1->A.1.1）
                if tmp['type'] == ftmp['type']:
                    if tmp['dot_count'] <= ftmp['dot_count']:
                        if  tmp['depth'] == None:
                            tmp['depth'] = ftmp['depth'] + tmp['dot_count'] - ftmp['dot_count']
                            article.pattern_tree[tmp['depth']].append([tmp['type'],tmp['dot_count']])
                        else:
                            if "parentthese" in tmp["type"]:
                                isChild = checkFirst(tmp['inside'])
                            else:
                                isChild = checkFirst(tmp['prefix'])
                            logging.info(f"isChild为{isChild}")
                            if not isChild and tmp['dot_count'] <= ftmp['dot_count']:
                                #同级直接合并，暂时不处理1后面仍然接1的情况
                                lastcode = sum(ord(c) for c in tmp["prefix"])
                                neighborcode = None
                                neighbor_dir  ={}
                                if tmp['dot_count'] < ftmp['dot_count']:
                                    if chunks[-1].metadata['outline'][2][tmp['depth']] != []:
                                        tmp_neighbor = chunks[-1].metadata['outline'][2][tmp['depth']][0]
                                    else:
                                        logging.warning(f"无法分辨{total_splits[i].page_content}的层级，请检查序号。暂时作为正文加入切片中")
                                        chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i].page_content
                                        check_unique(chunks[-1], total_splits[i])
                                        i+=1
                                        continue
                                else:
                                    tmp_neighbor = chunks[-1].metadata['outline'][0]
                                if r'\s' in tmp_neighbor:
                                    neighbor_dir = extract_features(tmp_neighbor)
                                    tmp_neighbor = neighbor_dir['prefix']
                                    logging.info(f"neighbor_dir为{neighbor_dir}")
                                    if len(tmp_neighbor) != len(tmp['prefix']):
                                        #处理9-10之类转行的情况。
                                        def extract_prefix(s): #提取数字
                                            result = 0
                                            hasDigit = False
                                            for c in s:
                                                if c.isdigit():
                                                    result = result*10 + int(c)
                                                    hasDigit = True
                                            if hasDigit:
                                                return result
                                            return -1
                                        lastcode = extract_prefix(tmp['prefix'])
                                        neighborcode = extract_prefix(tmp_neighbor)
                                        if lastcode*neighborcode < 0:
                                            logging.warning(f"相同层级的目录序号结构不同，无法分辨{total_splits[i].page_content}的层级，请检查序号。暂时作为正文加入切片中")
                                            chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i].page_content
                                            check_unique(chunks[-1], total_splits[i])
                                            i+=1
                                            continue
                                else:
                                    neighborcode = sum(ord(c) for c in tmp_neighbor)
                                    neighbor_dir = {'head': tmp_neighbor}
                                if lastcode - neighborcode >=3 or lastcode - neighborcode <=0 and '~' not in neighbor_dir['head'] :
                                    #排除1.0.1~1.0.7， 1.0.8这种情况， 、只包含两个数，差值理论上不会大于等于3
                                    logging.info("同级标题差距太大，猜测是标题错误或为注释，作为正文加入当前分块")
                                    chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i].page_content
                                    check_unique(chunks[-1], total_splits[i])
                                    i+=1
                                    continue
                            if isChild:
                                logging.info("标题层次重复，直接作为正文")
                                lastcode = sum(ord(c) for c in tmp["prefix"])
                                
                                chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i].page_content
                                check_unique(chunks[-1], total_splits[i])
                                while True:
                                    stmp = extract_features(total_splits[i+1].page_content)
                                    tmpcode = sum(ord(c) for c in stmp["prefix"])
                                    logging.info(f"while循环中检查下一个元素: stmp={stmp}, tmp={tmp}")
                                    logging.info(f"tmpcode={tmpcode}, lastcode={lastcode}")
                                    if stmp['type'] == tmp['type'] and stmp['dot_count'] == tmp['dot_count']:
                                        logging.info(f"类型和dot_count相同，检查tmpcode条件")
                                        if tmpcode == lastcode + 1:
                                            logging.info(f"tmpcode == lastcode + 1，继续循环")
                                            chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i+1].page_content
                                            check_unique(chunks[-1], total_splits[i+1])
                                            lastcode = tmpcode
                                            i+=1
                                            continue
                                        else:
                                            logging.info(f"tmpcode != lastcode + 1，准备break")
                                    pattern_slice = article.pattern_tree[:ftmp['depth']+1]
                                    patterns_flat = [pattern for sublist in pattern_slice for pattern in sublist]
                                    logging.debug(f"pattern_tree切片: {pattern_slice}")
                                    logging.debug(f"检查模式: [stmp['type'],stmp['dot_count']] = {[stmp['type'],stmp['dot_count']]}")
                                    if stmp['type'] == 'content' or [stmp['type'],stmp["dot_count"]] not in patterns_flat:
                                        logging.debug(f"下一个元素是content或不在pattern_tree中，继续循环")
                                        chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i+1].page_content
                                        check_unique(chunks[-1], total_splits[i+1])
                                        i+=1
                                        continue
                                    logging.info(f"所有条件都不满足，执行break")
                                    break
                                i+=1
                                continue
                        addParentOutline()
                        continue


                        '''
                        elif tmp['depth'] >= ftmp['depth'] and tmp['dot_count'] < ftmp['dot_count']:
                            #纠正由于目录层级分布不全而造成的错误识别。例如本来是A->A.0.1, C->C.1,此时default_recognizer会将C.1与A.0.1识别为同一级别，如果此时再有D->D.1->D.0.1的情况，就应当进行纠正。
                            #由于识别是从顶层向底层，所以将D.0.1的深度纠正为D.1的深度+1
                            #格式相同的情况下，由于tree中只会有一个相同格式，无须纠正。
                            article.pattern_tree[ftmp['depth']].remove([ftmp['type'],ftmp['dot_count']])
                            article.addtree(tmp['depth'] + ftmp['dot_count'] - tmp['dot_count'], [ftmp['type'],ftmp['dot_count']])
                        '''
                        #目录结构修正会在从父目录切换到子目录时进行，无需在这里进行
                        
                    else:
                        addChild = False
                        if not tmp['depth']:
                            tmpdepth = ftmp['depth'] + tmp['dot_count'] - ftmp['dot_count']
                            if tmpdepth < PARENT_DEPTH + 2: #只将两层以内的子标题记为新的标题
                                tmp['depth'] = tmpdepth
                                article.addtree(tmp['depth'], [tmp['type'],tmp['dot_count']])
                                logging.info(f"添加子标题{[tmp['type'], tmp['depth']]}到深度{tmp['depth']}")
                                addChild = True
                        else:
                            if tmp['depth'] <= ftmp['depth'] :
                            #纠正可能由于识别时跳过目录而造成的错误识别。例如本来是A->A.1->A.1.1, B->B.0.1, C->C.1->C.1.1,但default读取掉了A.1.1与C.1.1导致A.1与B.0.1对应的格式在目录树中为同一级别，此时需要将C.1.1的深度纠正为C.1的深度+1。
                            #由于识别是从顶层向底层，所以将A.1.1的深度纠正为A.1的深度+1
                            #格式相同的情况下，由于tree中只会有一个相同格式，无须纠正。
                            #之后循环到的目录标题受到的影响在循环开头修正，此时的父标题深度与结构不会受到影响，如果有空缺，只需要将空缺部分加在父标题列表末尾即可。
                                article.pattern_tree[tmp['depth']].remove([tmp['type'],tmp['dot_count']])
                                article.addtree(ftmp['depth'] + tmp['dot_count'] - ftmp['dot_count'], [tmp['type'],tmp['dot_count']])
                                tmp['depth'] = ftmp['depth'] + tmp['dot_count'] - ftmp['dot_count']
                            addChild = True
                        if addChild:        
                            addChildOutline()
                            continue
                            #如果不在目录标题两层子标题以内，直接识别为正文。
                else:
                    #处理目录类型不同的情况
                    if "parentthese" in tmp["type"]:
                        isChild = checkFirst(tmp['inside'])
                    else:
                        isChild = checkFirst(tmp['prefix'])
                    if tmp['depth'] != None and ftmp['depth'] == tmp['depth']:
                        #两层标题之间插入新标题的情况
                        article.pattern_tree[tmp['depth']].remove([tmp['type'],tmp['dot_count']])
                        tmp["depth"] += 1
                        article.addtree(tmp['depth'], [tmp['type'],tmp['dot_count']])
                        isChild = True
                    if not isChild: #切到上级标题。
                        if tmp['depth'] != None:
                            tmpcode = sum(ord(c) for c in tmp["prefix"])
                            parenthead = chunks[-1].metadata['outline'][2][tmp['depth']][0]
                            if r'\s' in parenthead:
                                f = parenthead.split(r'\s')[0]
                                pcode = sum(ord(c) for c in f)
                            else:
                                pcode = sum(ord(c) for c in parenthead)
                        if tmp['depth'] == None or tmp["depth"] >= ftmp["depth"] or tmpcode-pcode>=3:
                            logging.warning(f"可能新加切片{total_splits[i].page_content}的序号层级过低或{chunks[-1].page_content}与新加切片同级的父标题没有被识别出来或误在开头字符中加入了空格，请检查正文，暂且将该序号加入正文中")
                            chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i].page_content
                            check_unique(chunks[-1], total_splits[i])
                            i+=1
                            continue
                        else:
                            addParentOutline()
                            continue
                    if isChild:
                        #如果不在目录标题两层子标题以内，直接识别为正文。
                        if tmp['depth'] == None:
                            tmp['depth'] = ftmp['depth'] + 1
                            #如果不在目录标题两层子标题以内，直接识别为正文。
                            if tmp['depth'] <= PARENT_DEPTH + 2:
                                article.addtree(tmp['depth'], [tmp['type'],tmp['dot_count']])
                        if tmp['depth'] > ftmp['depth'] and tmp['depth'] <= PARENT_DEPTH + 2:
                            #除去出现1之类同一标题种类在不同标题层级的情况。
                            #例如1->(1)->1,这种情况将该末端标题作为正文
                            #由于1->(1)->2的情况一定在处理前文时将1识别为（1）的父标题，不会出现1跟（1）在同一层的情况。
                            addChildOutline()
                            continue
                        else:
                            logging.info(f"可能新加切片{total_splits[i].page_content}的序号在父标题中出现过或层级太低，作为正文处理。")
                            lastcode = sum(ord(c) for c in tmp["prefix"])
                            chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i].page_content
                            check_unique(chunks[-1], total_splits[i])
                            while True:
                                stmp = extract_features(total_splits[i+1].page_content)
                                if stmp['type'] == tmp['type'] and stmp['dot_count'] == tmp['dot_count']:
                                    tmpcode = sum(ord(c) for c in stmp["prefix"])
                                    if tmpcode == lastcode + 1:
                                        chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i+1].page_content
                                        check_unique(chunks[-1], total_splits[i+1])
                                        lastcode = tmpcode
                                        i+=1
                                        continue
                                pattern_slice = article.pattern_tree[:ftmp['depth']+1]
                                patterns_flat = [pattern for sublist in pattern_slice for pattern in sublist]
                                if stmp['type'] == 'content' or [stmp['type'],stmp["dot_count"]] not in patterns_flat:
                                    chunks[-1].page_content = chunks[-1].page_content + "\n" + total_splits[i+1].page_content
                                    check_unique(chunks[-1], total_splits[i+1])
                                    i+=1
                                    continue
                                break
                            i+=1
                            continue
                    
            if chunks==[]: #开头必然是第一个标题，与之强行匹配
                #print(f"开头必然是第一个标题，与之强行匹配。")
                logging.warning("目录第一个标题的识别似乎有问题，请检查目录与目录后正文开头，但正文开头大概率是第一条目录，进行强制添加。")
                tmp = article.dir_features[num+1]
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
                ftmp = tmp
                continue
            #不属于任何标题的内容，归入当前chunk
            
            chunks[-1].page_content = chunks[-1].page_content+"\n"+total_splits[i].page_content
            #print(f"不属于任何标题的内容，归入当前chunk,当前chunk为{chunks[-1].page_content}")
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
    tmpj = 0
    insideHead = False
    tmppattern = []
    vision_pattern = []
    for pattern in OUTLINE_PATTERN[-1]:
        #print(f"pattern为{pattern}")
        leftmatches = list(re.finditer(r"\(", pattern))
        rightmatches = list(re.finditer(r"\)", pattern))
        if not leftmatches or not rightmatches or len(leftmatches) != len(rightmatches):
            logging.warning("未从目录结构最后一层检测到序号格式，请检查目录最后一层的捕获组设置。")
            vision_pattern = [r'^\d+\.\d+\.\d+(?:\s*[~\\~、]\s*\d+\.\d+\.\d+)*']
        else:
            for t, left in enumerate(leftmatches):
                if pattern[left.start()-1] != "\\":
                    tmppattern.append(pattern[left.start()+1:rightmatches[t].start()])
                    vision_pattern.append(rf'^{tmppattern[-1]}(?:\s*[~\\~、]\s*{tmppattern[-1]})*')
                    break
            if vision_pattern ==[]:
                logging.warning("未从目录结构最后一层检测到序号格式，请检查目录最后一层的捕获组设置。")
                vision_pattern = [r'^\d+\.\d+\.\d+(?:\s*[~\\~、]\s*\d+\.\d+\.\d+)*']
    logging.info(f"tmppattern为{tmppattern}")
    logging.info(f"vision_pattern为{vision_pattern}")
    for i, chunk in enumerate(add_chunks):
        if j == len(base_chunks):
            j = tmpj
        else:
            tmpj = j #存储当前j的值，处理上一条没找到的情况
        pure_chunk = chunk.page_content[chunk.metadata["outlineend"]:].strip()
        #print(f"pure_chunk为{pure_chunk}")
        space_index = pure_chunk.find(' ')
        chunk_head = chunk.metadata["outline"][0].strip()
        if space_index != -1:
            pure_content = pure_chunk[space_index+1:]
            head = pure_chunk[:space_index]
            #print("head为",head)
            if chunk_head == "":
                #chunk_head = re.match(r"^\d+\.\d+\.\d+", head)
                chunk_head, _ = extract_matching_parts(head,OUTLINE_PATTERN[-1],useCapture=True)#检测最后一层标题，只保留第一个捕获组（序号）
                if chunk_head == "":
                    logging.warning(f"无法识别附件说明切片的表头，目前切片为{chunk.page_content}")
                    continue
        elif chunk_head == "":
            #head = re.match(rf'^{version_pattern}(?:\s*[~\\~、]\s*{version_pattern})*', pure_chunk)
            head, vision = extract_matching_parts(pure_chunk,vision_pattern)
            if head != "" :
                logging.warning(f"无法识别附件说明切片的表头，目前切片为{chunk.page_content}")
                continue
            pure_content = pure_chunk[len(head):].strip()
            #head = head.group()
            if chunk_head == "":
                t = vision_pattern.index(vision)
                chunk_head = re.match(tmppattern[t], head)
                if chunk_head:
                    chunk_head = chunk_head.group()
                else:
                    logging.warning(f"无法识别附件说明切片的表头，目前切片为{chunk.page_content}")
                    continue
        #print(f"head为{head},原切片为{pure_chunk}")
        #if not re.match(r"^\d+\.\d+\.\d+", chunk_head):
        if extract_matching_parts(chunk_head,tmppattern)[0] == "":
            logging.info(f"没有在末端目录结构中找到{chunk_head}匹配的标题格式")
            head = chunk_head
            head = head.replace("付录", "附录").replace("符录", "附录")
            #print(f"head为{head},原切片为{pure_chunk}")
            while j<len(base_chunks):
                #print(base_chunks[j].metadata["outline"][2])
                outline_text = base_chunks[j].metadata["outline"][0]
                outline_text = outline_text.replace("付录", "附录").replace("符录", "附录")
                #print(f"outline_text为{base_chunks[j].metadata["outline"]}")
                if any(fuzzy_match(head,head, outline_item[0])[0] for outline_item in base_chunks[j].metadata["outline"][2] if outline_item):
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + head + '：' + pure_chunk
                    merge_2chunk(base_chunks[j], chunk)
                    #print(f"合并后为{base_chunks[j].page_content}")
                    break

                similar,_ = fuzzy_match(head, outline_text)
                if similar :
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + head + '：' + pure_chunk
                    merge_2chunk(base_chunks[j], chunk)
                    #print(f"合并后为{base_chunks[j].page_content}")
                    j+=1
                    break
                
                j+=1
            continue

        #print(f"序号范围是{head}")
        tmp = head.find('~')
        if tmp != -1:
            chunk_tail = head[tmp+1:].strip()
            #print(chunk_tail)
            #print(chunk_head)
            while j<len(base_chunks):
                #print(base_chunks[j].metadata["outline"][0])
                # 排除chunk_head是outline[0]前缀的情况
                outline_text = base_chunks[j].metadata["outline"][0]
                if chunk_head == outline_text or (chunk_head in outline_text and not outline_text.startswith(chunk_head)):
                    #print(f"匹配到起点分块，具体位置为{j},内容为{base_chunks[j].page_content}")
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_chunk
                    merge_2chunk(base_chunks[j], chunk)
                    #print(f"匹配到起点分块，具体位置为{j},内容为{base_chunks[j].page_content},additionstart为{base_chunks[j].metadata["additionstart"]}")
                    insideHead = True
                    tmpj = j
                    j+=1
                    continue
                if (chunk_tail in base_chunks[j].metadata["outline"][0] and not base_chunks[j].metadata["outline"][0].startswith(chunk_tail)) or chunk_tail == base_chunks[j].metadata["outline"][0]:
                    #try:
                        #print(f"匹配到终点分块，具体为{base_chunks[j].page_content}")
                    #except UnicodeEncodeError:
                        # 使用UTF-8编码直接输出到标准输出
                        #content = f"匹配到终点分块，具体为{base_chunks[j].page_content}\n"
                        #sys.stdout.buffer.write(content.encode('utf-8'))
                        #sys.stdout.flush()
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_chunk
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    insideHead = False
                    break
                if insideHead:
                    
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_chunk
                    merge_2chunk(base_chunks[j], chunk)
                    #print(f"当前j为{j},base_chunks长度为{len(base_chunks)}，分块{base_chunks[j].metadata["outline"][0]}在目录{head}的范围中,additionstart为{base_chunks[j].metadata["additionstart"]}")
                    j+=1
                    continue
                j+=1
                continue
            if insideHead == True:
                j = tmpj+1
                #print(f"当前j为{j}")
                insideHead = False
                logging.warning(f"无法找到条文说明{head}结尾对应的正文，请检查文件正文对应的标题是否有问题,暂且去掉{chunk_head}以后所有条目中的该说明。")
                k = tmpj +1
                while k<len(base_chunks):
                    #print(f"当前j为{k}，分块为{base_chunks[k].metadata["outline"][0]},additionstart为f{base_chunks[k].metadata["additionstart"]}")
                    base_chunks[k].page_content = base_chunks[k].page_content[:base_chunks[k].metadata["additionstart"][-1]]
                    base_chunks[k].metadata["additionstart"].pop()
                    k+=1
            continue
        tmp = head.find('、')
        if tmp != -1:
            chunk_tail = head[tmp+1:].strip()
            while j<len(base_chunks):
                #print(base_chunks[j].metadata["outline"][0])
                # 排除chunk_head是outline[0]前缀的情况
                outline_text = base_chunks[j].metadata["outline"][0]
                if chunk_head == outline_text or (chunk_head in outline_text and not outline_text.startswith(chunk_head)):
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_chunk
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    continue
                if (chunk_tail in base_chunks[j].metadata["outline"][0] and not base_chunks[j].metadata["outline"][0].startswith(chunk_tail)) or chunk_tail == base_chunks[j].metadata["outline"][0]:
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_chunk
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    break
                j+=1
            continue
        else:
            #print(f"开始循环的目录为{base_chunks[j].metadata["outline"]}")
            while j<len(base_chunks):
                #print(base_chunks[j].metadata["outline"][0])
                # 排除chunk_head是outline[0]前缀的情况
                outline_text = base_chunks[j].metadata["outline"][0]
                if chunk_head == outline_text or (chunk_head in outline_text and not outline_text.startswith(chunk_head)):
                    if "additionstart" not in base_chunks[j].metadata:
                        base_chunks[j].metadata["additionstart"] = [len(base_chunks[j].page_content)]
                    else:
                        base_chunks[j].metadata["additionstart"].append(len(base_chunks[j].page_content))
                    base_chunks[j].page_content = base_chunks[j].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_chunk    
                    merge_2chunk(base_chunks[j], chunk)
                    j+=1
                    break
                j+=1
            
        if j == len(base_chunks):
            j = tmpj
            logging.warning(f"无法找到条文说明{chunk_head}对应的正文，请检查文件正文对应的标题,暂且将该条文说明合并到上一条有说明的切片{base_chunks[j-1].metadata["outline"][0]}中（可以处理部分情况）。")
            if "additionstart" not in base_chunks[j-1].metadata:
                base_chunks[j-1].metadata["additionstart"] = [len(base_chunks[j-1].page_content)]
            else:
                base_chunks[j-1].metadata["additionstart"].append(len(base_chunks[j-1].page_content))
            base_chunks[j-1].page_content = base_chunks[j-1].page_content + '\n\n' + chunk.metadata["header"][0] + '：' + pure_chunk
            merge_2chunk(base_chunks[j-1], chunk)        



def merge_chunk(article:Article,total_splits):
    '''
    处理包含附件的文档
    '''
    threshold = 0.8
    i=0
    b = sum(len(addition.base_splits)for addition in article.additions)
    base_chunks = merge_chunk_through_outlines(article, article.base_splits[:len(article.base_splits)-b])
    #outputtest_file(base_chunks,"base.md")
    for i,chunk in enumerate(base_chunks):
        chunk.metadata["chunk_id"] = i
    for addition in article.additions:
        add_chunks = merge_chunk_through_outlines(addition, addition.base_splits)
        #outputtest_file(add_chunks,"addition.md")
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



#chunk_size暂定为字符串长度
def merge_size_chunk(article:Article, total_splits, chunk_size):
    basic = merge_chunk(article, total_splits)
    sized_chunk = []
    j = 0
    while j < len(basic)-1:
        if len(basic[j].page_content) > chunk_size*0.8:
            j+=1
            continue
        else:
            while j<len(basic)-1 and (len(basic[j].page_content) < chunk_size and len(basic[j+1].page_content)<chunk_size*0.8 \
                or len(basic[j].page_content) < chunk_size*0.4 ):
                if basic[j].metadata["outline"][2] == basic[j+1].metadata["outline"][2] or \
                    (isinstance(basic[j].metadata["outline"][2], list) and isinstance(basic[j+1].metadata["outline"][2], list) \
                        and len(basic[j].metadata["outline"][2]) <= len(basic[j+1].metadata["outline"][2]) \
                        and all(item in basic[j+1].metadata["outline"][2] for item in basic[j].metadata["outline"][2]) \
                        and basic[j].metadata["outline"][:2] in basic[j+1].metadata["outline"][2]):
                    if "additionstart" in basic[j].metadata and "additionstart" in basic[j+1].metadata and \
                        basic[j].metadata["additionstart"] != [] and basic[j+1].metadata["additionstart"] != []:
                        for i in range(len(basic[j].metadata["additionstart"])):#处理条文说明重复问题
                            if i == len(basic[j].metadata["additionstart"])-1:
                                o1description = basic[j].page_content[basic[j].metadata["additionstart"][i]:]
                            else:
                                o1description = basic[j].page_content[basic[j].metadata["additionstart"][i]:basic[j].metadata["additionstart"][i+1]]
                            odescription = o1description.strip()
                            #print(f"读取的odescription为：{odescription}")
                            ndescription = basic[j+1].page_content[basic[j+1].metadata["additionstart"][0]:].strip()
                            #print(f"读取的ndescription为：{ndescription}")
                            if odescription in ndescription:
                                basic[j].page_content = basic[j].page_content.replace(o1description, "")
                                for idx in range(i+1, len(basic[j].metadata["additionstart"])):
                                    basic[j].metadata["additionstart"][idx] -= len(o1description)
                    merge_2chunk(basic[j], basic[j+1])
                    if basic[j].metadata["outline"][2] == basic[j+1].metadata["outline"][2]:
                        overlap = len(basic[j].page_content)+2-len(struct(basic[j].metadata["outline"][2]))
                        basic[j].page_content = basic[j].page_content + "\n\n" + basic[j+1].page_content[len(struct(basic[j].metadata["outline"][2])):]
                        if not isinstance(basic[j].metadata["outline"][0], list):
                            basic[j].metadata["outline"][0] = [basic[j].metadata["outline"][0]]
                            basic[j].metadata["outline"][1] = [basic[j].metadata["outline"][1]]
                        basic[j].metadata["outline"][0].append(basic[j+1].metadata["outline"][0])
                        basic[j].metadata["outline"][1].append(basic[j+1].metadata["outline"][1])
                    else:
                        overlap = len(basic[j].page_content)+2-len(struct(basic[j].metadata["outline"][2])+struct([basic[j].metadata["outline"][:2]]))
                        basic[j].page_content = basic[j].page_content + "\n\n" + basic[j+1].page_content[len(struct(basic[j].metadata["outline"][2])+struct([basic[j].metadata["outline"][:2]])):]
                        basic[j].metadata["outline"] = basic[j+1].metadata["outline"]
                    if "additionstart" not in basic[j+1].metadata or basic[j+1].metadata["additionstart"] == []:
                        basic[j].metadata["additionstart"] = []
                    else:
                        basic[j].metadata["additionstart"] = [num+overlap for num in basic[j+1].metadata["additionstart"]] 
                    basic.pop(j+1)
                    continue
                else:
                    #print(f"{basic[j].metadata['outline'][0]}的父标题{basic[j].metadata['outline'][2]}与{basic[j+1].metadata['outline'][0]}的父标题{basic[j+1].metadata['outline'][2]}不同或不为包含关系")
                    break
            j += 1
    return basic
                        

#对目录前的部分进行简单的大小分割
def simple_size_chunk(base_splits, chunk_size):
    i = 0
    while i < len(base_splits)-1:
        if len(base_splits[i].page_content) > chunk_size*0.8:
            i += 1
            continue
        else:
            if base_splits[i].metadata["type"] not in ["table", "equation", "figure"]\
                and base_splits[i+1].metadata["type"] not in ["table", "equation", "figure"]\
                and base_splits[i].metadata["type"] !=  base_splits[i+1].metadata["type"]:
                i += 1
                continue
            else:
                while(len(base_splits[i].page_content)<chunk_size and i<len(base_splits)-1 and len(base_splits[i+1].page_content)<chunk_size*0.8) :
                    base_splits[i].page_content = base_splits[i].page_content + "\n\n" + base_splits[i+1].page_content
                    check_unique(base_splits[i], base_splits[i+1])
                    base_splits.pop(i+1)
                i += 1
    return base_splits  



if __name__ == "__main__":
    file_path = r"GB50023-2009：建筑抗震鉴定标准.md"
    # 使用TextLoader保持原始markdown格式
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    file_name = Path(file_path).name               
    documents[0].metadata['source'] = file_name
    chunks = mdfile_recognizer(documents[0],chunk_size = 500)
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
    with open("finalresult.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))