import os
import re 
import asyncio
from dotenv import load_dotenv
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
OUTLINE_PATTERN = (r"#? ?\d+", 
                   r"#? ?\d+\.\d+",
                   r"#? ?\d+\.\d+\.\d+",
                   r"#? ?附录[A-Z]",
                   r"#? ?[\u4e00-\u9fa5]+ \.+")



NOTICE_PATTERN = r"#.*\n公告\n"
FOREWORD_PATTERN = r"# 前言\n"
OUTLINES_CN_PATTERN = r"# 目次\n"
OUTLINES_EN_PATTERN = r"# CONTENTS\n"
CONTENT_PATTERN = r"[a-zA-Z]+\n+#? ?(?:\d+ )?[\u4e00-\u9fa5]+"

def get_cover(document:Document):
    cover = re.search(NOTICE_PATTERN, document.page_content)
    if cover:
      return cover.start
    return -1


def mdfile_recognizer(document:Document):

    #读取正文前的内容，包括封面、公告、前言、目录、英文目录
    Start = 0
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
            Start = End
    
    outlines = outline_recognize(outlines_cn)
    content = Document(page_content=document.page_content[Start:],metadata=document.metadata)

    base_splitter = CharacterTextSplitter(
        separator="\n{2,}",
        is_separator_regex=True
    )
    
    base_splits = base_splitter.split_documents([document])
    cover = []
    outline = []
    headers = []
    tables = []
    for s in base_splits:
        continue

    def outline_recognize(outlines_cn:Document):
        outline = []
        for line in outlines_cn.page_content.splitlines():
            if line.startswith("#"):
                outline.append(line)

        pass

  