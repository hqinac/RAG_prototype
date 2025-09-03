from pathlib import Path
import re
import Levenshtein
from langchain_core.documents import Document

#将切片打印到指定mdfile里
def outputtest_file(chunks,filename):
    
    # 检查文件是否存在
    file_exists = Path(filename).exists()
    
    output_content = []
    
    # 如果文件存在，添加分隔符
    if file_exists:
        output_content.append("\n" + "="*80)
        output_content.append("=== 召回切片 ===")
        output_content.append("="*80)
        output_content.append("")
    
    for i, chunk in enumerate(chunks):       
        # 添加分界线（除了第一个切片）
        if i > 0:
            output_content.append("="*80)  # 分界线
            output_content.append("")  # 空行
        
        # 添加切片编号
        output_content.append(f"=== 文档 {i+1} ===")
        output_content.append("")  # 空行
        
        # page_content
        output_content.append(chunk.page_content)
        output_content.append("")  # 空行
        
        # metadata信息
        output_content.append("--- Metadata ---")
        metadata_lines = []
        for key, value in chunk.metadata.items():
            metadata_lines.append(f"{key}: {value}")
        output_content.extend(metadata_lines)
        output_content.append("")  # 空行
    
    # 根据文件是否存在选择写入模式
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, encoding='utf-8') as f:
        f.write('\n'.join(output_content))


def struct(t):#构建用于chunk的标题格式
    res = ""
    if not t or t == []:
            return res
    for tmp in t:
        if(tmp[1] == ""):
            res += tmp[0] + "："
        else:
            res += tmp[0] + "（" + tmp[1] + "）" + "："
    return res



def fuzzy_match(outline:str,split:str,threshold=0.8):
    '''
    对目录结构进行模糊匹配
    '''
    end = 0
    # 首先尝试精确匹配
    tmp = re.match(r'^' + re.escape(outline),split)
    #print(tmp)
    if tmp:
        end = tmp.end()
        #print(f"精确匹配成功，end位置为{end},split[end]为{split[end-1]}")
        return True, end
    
    # 如果精确匹配失败，尝试忽略空格的匹配
    outline_no_space = re.sub(r'\s+', '', outline)
    split_no_space = re.sub(r'\s+', '', split[:len(outline)*2])  # 限制搜索范围
    if outline_no_space == split_no_space[:len(outline_no_space)]:
        # 找到原始split中对应的结束位置
        char_count = 0
        for i, char in enumerate(split):
            if char != ' ':
                char_count += 1
                if char_count == len(outline_no_space):
                    end = i + 1
                    #print(f"忽略空格匹配成功，end位置为{end}")
                    return True, end
    if re.match(r'^\d+\.\d+\.\d+',split):
        return False, -1
    sign = re.search(r'\s*\S*?([^\w\s]|\n)', split)
    if sign and sign.group(1) in ["。","！","？","，",","]:
        return False, -1
    if '.' in outline:
        space_outline = outline.split(' ')[0]
        space_split = split.split(' ')[0]
        if space_outline == space_split:
            return True, min(len(outline),len(split))
    distance = Levenshtein.distance(outline,split[:len(outline)])
    similarity = 1 - distance / len(outline)
    bestscore = similarity
    end = len(outline)
    if similarity >= threshold:
        window_sizes = range(max(1, len(outline) - 2), min(len(split),len(outline) + 3))
        for i in window_sizes:
            td = Levenshtein.distance(outline,split[:i])
            score = 1- td/max(len(outline),i)
            if score > bestscore:
                bestscore = score
                end = i
        return True, end
    return False, -1

#检查当前切片是否有特殊格式
def check_unique(chunk, split):
    match split.metadata.get("type", "text"):
        case "table":
            chunk.metadata["has_table"] = True
            if "table_name" in split.metadata:
                chunk.metadata["table_names"].append(split.metadata["table_name"])
        case "equation":
            chunk.metadata["has_equation"] = True
            if "equation_name" in split.metadata:
                chunk.metadata["equation_names"].append(split.metadata["equation_name"])
        case "figure":
            chunk.metadata["has_figure"] = True
            if "image_link" in split.metadata:
                chunk.metadata["figure_links"].append(split.metadata["image_link"])
            if "image_name" in split.metadata:
                chunk.metadata["figure_names"].append(split.metadata["image_name"])

    #if chunk.metadata["has_figure"] == True:
        #print("图片加入情况为",chunk,'\n')

    return

def delete_addition_splits(title_to_remove, addition):
    j=0
    while j < len(addition.base_splits):
        if addition.base_splits[j].page_content == title_to_remove:
            addition.base_splits.pop(j)
            return True
        j+=1
    return False

def merge_2chunk (chunk1, chunk2): #将chunk2的元数据合并到chunk1
    if chunk1.metadata["has_table"] and chunk2.metadata["has_table"]:
        for name in chunk2.metadata["table_names"]:
            if name not in chunk1.metadata["table_names"]:
                chunk1.metadata["table_names"].append(name)
    if chunk1.metadata["has_equation"] and chunk2.metadata["has_equation"]:
        for name in chunk2.metadata["equation_names"]:
            if name not in chunk1.metadata["equation_names"]:
                chunk1.metadata["equation_names"].append(name)
    if chunk1.metadata["has_figure"] and chunk2.metadata["has_figure"]:
        for name in chunk2.metadata["figure_names"]:
            if name not in chunk1.metadata["figure_names"]:
                chunk1.metadata["figure_names"].append(name)
        for link in chunk2.metadata["figure_links"]:
            if link not in chunk1.metadata["figure_links"]:
                chunk1.metadata["figure_links"].append(link)
    chunk1.metadata["has_table"] = chunk1.metadata["has_table"] or chunk2.metadata["has_table"]
    chunk1.metadata["has_equation"] = chunk1.metadata["has_equation"] or chunk2.metadata["has_equation"]
    chunk1.metadata["has_figure"] = chunk1.metadata["has_figure"] or chunk2.metadata["has_figure"]
    #chunk1.metadata["table_names"] += chunk2.metadata["table_names"]
    #chunk1.metadata["equation_names"] += chunk2.metadata["equation_names"]
    #chunk1.metadata["figure_names"] += chunk2.metadata["figure_names"]
    #chunk1.metadata["figure_links"] += chunk2.metadata["figure_links"]

def extract_matching_parts(text,last_pattern_tuple, useCapture = False):
    """
    检测给定字符串是否符合 OUTLINE_PATTERN 最后一行中的正则表达式格式，
    并提取出所有符合格式的部分。
    """
    # 获取 OUTLINE_PATTERN 的最后一行（即最后一个元组）中的正则表达式
    
    
    # 遍历最后一个元组中的所有正则表达式
    for pattern_str in last_pattern_tuple:
        try:
            # 编译正则表达式
            pattern = re.compile(pattern_str)
            # 查找所有匹配项
            matches = pattern.match(text)
            if matches:
                #print(f"字符串 '{text}' 匹配正则表达式 '{pattern_str}'，提取到的部分：{matches}")
                if useCapture:
                    return matches.group(1),pattern_str
                return matches.group(),pattern_str
        except re.error as e:
            print(f"正则表达式 '{pattern_str}' 无效: {e}")
    
    #print(f"字符串 '{text}' 不符合 OUTLINE_PATTERN 最后一行中的任何格式。")
    return "",None