from pathlib import Path
import re
import Levenshtein
from langchain_core.documents import Document

#将切片打印到指定mdfile里
def outputtest_file(chunks,filename):
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
    with open(filename, 'w', encoding='utf-8') as f:
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
    tmp = re.match(r'^' + re.escape(outline),split)
    if tmp:
        end = tmp.end()
        return True, end

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
            if "figure_link" in split.metadata:
                chunk.metadata["figure_links"].append(split.metadata["figure_link"])
            chunk.metadata["figure_names"].append(split.page_content)
    return