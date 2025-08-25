from dataclasses import dataclass
from nt import TMP_MAX
import os
from pickle import TRUE
import re 
import copy
import asyncio
from symtable import Class
from dotenv import load_dotenv
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from langchain_core.runnables.config import P
from langchain_text_splitters import CharacterTextSplitter

from cache_manager import get_embeddings, get_faiss, get_bm25, get_doc_cache, update_faiss_cache, update_bm25_cache, update_doc_cache

load_dotenv()

# 设置默认的向量数据库存储路径
URI = os.getenv("URI", "./saved_files")
HEAD_PATTERN = (
    ("cover", r"#.*\n公告\n"),
    ("notice", r"# 前言\n"),
    ("forehead", r"# 目次\n"),
    ("outlines_cn", r"# CONTENTS\n"),
    ("outlines_en", r"([a-zA-Z]+\n+)#? ?(?:\d+ )?[\u4e00-\u9fa5]+")
)#存储正文前各部分的名字与各部分结尾的正则表达式（即下一部分的开头）
OUTLINE_PATTERN = ((r"\d+ [\u4e00-\u9fa5a-zA-Z0-9]+",r"附录[A-Za-z0-9] [\u4e00-\u9fa5a-zA-Z0-9]+", r"[\u4e00-\u9fa5a-zA-Z0-9]+ +\.+" ),
                   (r"\d+\.\d+ [\u4e00-\u9fa5a-zA-Z0-9]+"),
                   (r"\d+\.\d+\.\d+ [\u4e00-\u9fa5a-zA-Z0-9]+"))


NOTICE_PATTERN = r"#.*\n公告\n"
FOREWORD_PATTERN = r"# 前言\n"
OUTLINES_CN_PATTERN = r"# 目次\n"
OUTLINES_EN_PATTERN = r"# CONTENTS\n"
CONTENT_PATTERN = r"[a-zA-Z]+\n+#? ?(?:\d+ )?[\u4e00-\u9fa5]+"


class Article:
    cover: Document
    notice: Document
    forehead: Document
    outlines_cn: Document
    outlines_en: Document
    content: Document
    additions: list[Article]


    def __init__(self, ducument):
        Start = 0
        for i,pattern in enumerate(HEAD_PATTERN):
            name,p = pattern
            End = re.search(p, document.page_content[start:]).start
            if name == "outlines_en":
                End = re.search(p, document.page_content[start:]).end(1)
            if end == -1:
                eval("self."+name+" ="" ")
                continue
            else:
                eval("self."+name+" = Document(page_content=document.page_content[Start:End],metadata=document.metadata)")
                eval("self."+name+".matadata['type'] = name")
                Start = End
        self.content = Document(page_content=document.page_content[Start:],metadata=document.metadata)

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
        for line in lines:
            line = line.strip()
            enline = enline.strip()
            if line == "":
                continue
            tail = re.search(r"\.+",line).start
            if tail == -1:
                tail = re.search(r"\d+(?=$)",line).start
            line = re.sub(r"^# +?","",line)
            cn.append(line[:tail].strip())


        for enline in enlines:
            line = enline.strip()
            if line == "":
                continue
            tail = re.search(r"\d+(?=$)",line).start()
            line = re.sub(r"^# +?","",line)
            en.append(line[:tail].strip())

        if not use_en or len(cn) != len(en):
            outline.append([line,"",""] for line in cn)
            use_en = False
            if len(cn) != len(en):
                print("中英文目录对照存在问题，请检查文件目录内容。")
        else:
            for i in range(len(cn)):
                outline.append([cn[i],en[i],""])

        depth = -1
        f =[] #记录父标题
        '''
        def struct(t):#构建用于chunk的标题格式
            res = ""
            if t == []:
                return res
            if use_en:
                for tmp in t:
                    res += tmp[0] + "："
            else:
                for tmp in t:
                    res += tmp[0] + "（" + tmp[1] + "）" + "："

            return res
        '''

        for i, t in enumerate(outline):
            for j ,p in enumerate(OUTLINE_PATTERN):
                for OUTLINE in p:
                    if re.match(OUTLINE, t[0]):
                        if i > depth:
                            t[2] = f
                            f.append(t)
                        else:
                            f = (f[:i] if i>0 else [])
                            t[2] = f
                            f.append(t)
                        depth = i
                        break
                if depth>=0:
                    break

        return outline



def mdfile_recognizer(document:Document, use_en = True, has_chunk_size = False, chunk_size = -1):

    #读取正文前的内容，包括封面、公告、前言、目录、英文目录
    document.metadata["type"] = "text"
    Start = 0
    '''
    for i,pattern in enumerate(HEAD_PATTERN):
        name,p = pattern
        End = re.search(p, document.page_content[start:]).start
        if name == "outlines_en":
            End = re.search(p, document.page_content[start:]).end(1)
        if end == -1:
            eval(name+" ="" ")
            continue
        else:
            eval(name+" = Document(page_content=document.page_content[Start:End],metadata=document.metadata)")
            eval(name+".matadata['type'] = name")
            Start = End
    '''
    article = Article(document)
    
    #识别目录结构
    outlines = article.outline_recognize(use_en)
    for outline in outlines[::-1]:
        if re.match(r"附：[\u4e00-\u9fa5a-zA-Z0-9]",outline[0]) : #附件
            tail = re.search(outline[0][2:],article.content.page_content).start()
            doc = Document(page_content=article.content.page_content[tail:],metadata=article.content.metadata)
            article.content.page_content = article.content.page_content[:tail]
            doc.metadata["header"] = outline[0][2:]
            article.additions.append(Article(doc))
            continue
        break
    article.additions = article.additions[::-1]


    base_splitter = CharacterTextSplitter(
        separator="\n{2,}",
        is_separator_regex=True
    )
    

    base_splits = base_splitter.split_documents([article.content])
    for split in base_splits:
        if bool(re.fullmatch(r"^-?\d+(\.\d+)?$", split.page_content)):
            base_splits.remove(split) #删除误识别的页码
    figures = check_figures(base_splits)
    equations = check_equations(base_splits)
    tables = check_table(base_splits)

    if has_chunk_size:
        chunks = merge_size_chunk(base_splits,outlines, chunk_size)
    else:
        chunks = merge_chunk(base_splits,outlines)


    



def check_figures(base_splits):
    figures = []
    i = 0
    while i < len(base_splits):
        if base_splits[i].page_content.startswith("!["):
            base_splits.metadata["type"] = "figure"
            image_link = re.search(r"images/([\s\S]*?)\.jpg",base_splits[i].page_content).group()
            image_name = re.search(r"图[\s　]+(\d+[.\d-]*\d+)\s+(.+)",base_splits[i].page_content).group()
            base_splits[i].metadata["image_link"] = image_link
            base_splits[i].page_content = image_name
            figures.append(copy.deepcopy(base_splits[i]))
        i+=1
    return figures


def check_equations(base_splits):
    equations = []
    i = 0
    while i<len(base_splits):
        if base_splits[i].page_content.startswith("$$"):
            base_splits[i].metadata["type"] = "equation"
            if base_splits[i-1].page_content.endwith("："):
                base_splits[i].page_content = base_splits[i-1].page_content +"\n\n"+ base_splits[i].page_content
                base_splits.pop(i-1)
                i-=1
            while True:
                if i+1 >= len(base_splits):
                    break
                if re.match(r"(\$?)([\s\S]*?)(\$?)\s*——\s*(.+)",base_splits[i+1].page_content) \
                    or base_splits[i+1].page_content.startswith("$$")\
                    or re.match(r"^(注|主)：",base_splits[i+1].page_content):
                    base_splits[i].page_content = base_splits[i].page_content +"\n\n"+ base_splits[i+1].page_content
                    base_splits.pop(i+1)
                else:
                    break
            equations.append(copy.deepcopy(base_splits[i]))
        i += 1
    return equations       

def check_table(base_splits):
    '''
    识别表的标题、备注、续表，进行合成
    '''
    tables = []
    i=0
    hasheader = False
    while i < len(base_splits):
        if base_splits[i].page_content.startswith("<table>"):
            base_splits[i].metadata["type"] = "table"
            uptitle = min (i-5,0)
            for j in range(uptitle, i+1):
                if re.match(r"^表 ",base_splits[j].page_content):
                    uptitle = j
                    hasheader = True
                    break
            if hasheader:
                table_name = base_splits[uptitle].page_content
                while i > uptitle:
                    base_splits[i].page_content = base_splits[i-1].page_content + "\n\n" + base_splits[i].page_content
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
                            base_splits.pop(j)
                    else: 
                        print(table_name + "的续表格式错误或不存在,请检查原文档")
                    continue
                if re.match(r"^(注|主)：",base_splits[i+1].page_content):
                    base_splits[i].page_content = base_splits[i].page_content + "\n\n" + base_splits[i+1].page_content
                    base_splits.pop(i+1)
                    continue
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
        return [cell.get_text(separator=" ").strip() for cell in cells]
    l = min(len(rows1),len(rows2))
    merge = soup1.new_tag('table')
    merge.extend(rows1)
    for i in range(l):
        if get_row_content(rows1[i]) != get_row_content(rows2[i]):
            merge.extend(rows2[i:])
            break
    ntable = header + str(merge) + tail
    return ntable

def merge_chunk(base_splits,outlines):
    chunks = []
    for split in base_splits:
        if split.metadata["type"] == "table":
            for outline in outlines:
                if split.metadata["page"] == outline.metadata["page"]:
                    outline.page_content = merge_table(outline.page_content,split.page_content)
                    break
        else:
            chunks.append(split)
    return chunks




        
    
    

  