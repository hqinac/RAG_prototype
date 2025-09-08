import re
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
from utils import struct

HEADER_PATTERN = {
    'number': r'^\d+$',
    'cn_number': r'^[一二三四五六七八九十]+$',
    'upper_cn_number': r'^[壹贰叁肆伍陆柒捌玖拾]+$',
    'lower_alpha': r'^[a-z]+$',
    'upper_alpha': r'^[A-Z]+$',
    'upper_roma': r"^[IVX\u2160-\u217F]+$",
    'lower_roma': r"^[ivxⅰ-ⅿ]+$"
}

def extract_features(dir_str: str, usePlain = False) -> Dict:
    """提取目录项的层级特征：前缀类型、点数量、基值等"""
    # 匹配数字前缀（如1.、2.1、3.1.1）
    num_match = re.match(r'^(\d+(\.\d+)*)(.*?)?\s+', dir_str)

    cn_num_match = re.match(r'^([一二三四五六七八九十]+)(.*?)?\s+', dir_str)

    upper_cn_num_match = re.match(r'^([壹贰叁肆伍陆柒捌玖拾]+)(.*?)?\s+', dir_str)

    circle_match = re.match(r'^([\u2460-\u2473\u3251-\u325F\u32B1-\u32BF]+)(.*?)?\s+', dir_str)

    # 匹配字母前缀（如a 、b ）
    lower_alpha_match = re.match(r'^([a-z]+(?:\.[a-zA-Z0-9]+)*)(.*?)?\s+', dir_str)

    upper_alpha_match = re.match(r'^([A-Z]+(?:\.[a-zA-Z0-9]+)*)(.*?)?\s+', dir_str)

    upper_roma_match = re.match(r'^([IVX\u2160-\u217F]+)(.*?)?\s+', dir_str)

    lower_roma_match = re.match(r"^([ivxⅰ-ⅿ]+)(.*?)\s+", dir_str)

    parenttheses_match = re.match(r'^([（\(][^）\)]+[）\)])(.*?)\s+', dir_str)
    
    right_parenttheses_match = re.match(r"(^.{1,3}[）\)])(.*?)\s+", dir_str)

    appendix_match = re.match(r"^((附|付|符)\s*录\s*[A-Za-z0-9])\s+", dir_str)

    annex_match = re.match(r"^((附|付|符)(:|：)\s*(.*?))(?:\s+|\n|$)", dir_str)

    def structmatch(match):
        logging.info(f"匹配到{match}")
        prefix = match.group(1)
        matches = list(re.finditer(r'\.', prefix))
        if matches == []:
            dot_count = 0
        else:
            start = matches[-1].end()
            if start == len(prefix):
                dot_count = len(matches)-1
            else:
                dot_count = len(matches)
        p = re.compile(f'^({re.escape(prefix)}[^\s]*?\s+)')
        head_match = p.match(dir_str)
        if head_match:
            head = head_match.group(1)
        else:
            head = dir_str
        return prefix, dot_count, head


    if parenttheses_match:
        prefix, dot_count, head = structmatch(parenttheses_match)
        insider = prefix[1:-1].strip()
        insidetype = None
        for name, pattern in HEADER_PATTERN.items():
            if re.match(pattern, insider):
                insidetype = name
                break
        if not insidetype:
            insidetype = 'other'
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        if(len(insider)<=3):
            return {
                'type': insidetype+"_parentthese",
                'prefix': prefix,
                'inside': insider,
                'dot_count': dot_count,
                'depth': None,
                'head': head.strip(),
                'content': content
            }
    
    if right_parenttheses_match:
        prefix, dot_count, head = structmatch(right_parenttheses_match)
        insider = prefix[:-1].strip()
        insidetype = None
        for name, pattern in HEADER_PATTERN.items():
            if re.match(pattern, insider):
                insidetype = name
                break
        if not insidetype:
            insidetype = 'other'
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': insidetype+"_right_parentthese",
            'inside': insider,
            'prefix': prefix,
            'dot_count': dot_count,
            'depth': None,
            'head': head.strip(),
            'content': content
        }
        
    if appendix_match:
        prefix, dot_count, head = structmatch(appendix_match)
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'appendix',
            'prefix': prefix,
            'dot_count': dot_count,
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
    if annex_match:
        prefix, dot_count, head = structmatch(annex_match)
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'annex',
            'prefix': prefix,
            'dot_count': dot_count,
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
    if num_match:
        prefix, dot_count, head = structmatch(num_match)
        # 基值（最前面的数字，用于匹配父目录）
        #base_num = int(re.findall(r'\d+', prefix)[0])
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'number',
            'prefix': prefix,
            'dot_count': dot_count,
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
    if cn_num_match:
        prefix, dot_count, head = structmatch(cn_num_match)
        # 基值（最前面的数字，用于匹配父目录）
        #base_num = int(re.findall(r'\d+', prefix)[0])
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'cn_number',
            'prefix': prefix,
            'dot_count': dot_count,  # 字母前缀通常依赖上下文
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
    if upper_cn_num_match:
        prefix, dot_count, head = structmatch(upper_cn_num_match)
        # 基值（最前面的数字，用于匹配父目录）
        #base_num = int(re.findall(r'\d+', prefix)[0])
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'upper_cn_number',
            'prefix': prefix,
            'dot_count': dot_count,  # 字母前缀通常依赖上下文
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
    if circle_match:
        prefix, dot_count, head = structmatch(circle_match)
        # 基值（最前面的数字，用于匹配父目录）
        #base_num = int(re.findall(r'\d+', prefix)[0])
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'circle',
            'prefix': prefix,
            'dot_count': dot_count,  # 字母前缀通常依赖上下文
            'depth': None,
            'head': head.strip(),
            'content': content
        }

    if lower_alpha_match:
        prefix, dot_count, head = structmatch(lower_alpha_match)
        # 基值（最前面的数字，用于匹配父目录）
        #base_num = int(re.findall(r'\d+', prefix)[0])
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'lower_alpha',
            'prefix': prefix,
            'dot_count': dot_count,  # 字母前缀通常依赖上下文
            'depth': None,
            'head': head.strip(),
            'content': content
        }

    if upper_alpha_match:
        prefix, dot_count, head = structmatch(upper_alpha_match)
        # 基值（最前面的数字，用于匹配父目录）
        #base_num = int(re.findall(r'\d+', prefix)[0])
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'upper_alpha',
            'prefix': prefix,
            'dot_count': dot_count,  # 字母前缀通常依赖上下文
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
    if upper_roma_match:
        prefix, dot_count, head = structmatch(upper_roma_match)
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'upper_roma',
            'prefix': prefix,
            'dot_count': dot_count,  # 字母前缀通常依赖上下文
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
        # 无明显前缀（如"参考文献"）
    
    if lower_roma_match:
        prefix, dot_count, head = structmatch(lower_roma_match)
        if usePlain:
            content = dir_str
        else:
            content = checkcontent(head, dir_str)
        return {
            'type': 'lower_roma',
            'prefix': prefix,
            'dot_count': dot_count,  # 字母前缀通常依赖上下文
            'depth': None,
            'head': head.strip(),
            'content': content
        }
    
    if usePlain:
        return {
            'type': 'plain',
            'prefix': '',
            'dot_count': 0,  # 标记为顶级
            'depth': None,
            'content': dir_str
        }
    return {
        'type': 'content',
        'prefix': '',
        'dot_count': 0,  
        'depth': None,
        'content': dir_str
    }


def checkcontent(head: str, content: str):
    #logging.info(f"传入的head为{head}")
    tmphead = content[len(head):]
    tmphead = re.match(r"^(.*?)(?:\n|$)",tmphead)
    if tmphead:
        tmphead = tmphead.group(1)
    #logging.info(f"正则匹配后的tmphead为: '{tmphead}'")
    tmp = re.search(r"\s*([,!?;。，！？；\s+]).*?(?=\n|$)",tmphead)
    #logging.info(f"标点符号匹配结果tmp为: {tmp}")
    if tmp :
        return head.strip()
    return content[:len(head)+len(tmphead)+1].strip()


def infer_hierarchy(directories: List[str], en_outlines=[], useen = True):
    """根据特征和顺序推断目录层级，返回带层级信息的列表"""
    # 提取所有目录的特征
    dir_features = [extract_features(dir_str, usePlain = True) for dir_str in directories]
    
    # 存储每个目录的层级（0为顶级）和父目录索引
    outlines = []
    depths = []
    # 用栈记录可能的父目录（按层级从深到浅）
    parent_dic = {}
    pattern_tree = []

    def addoutline(features: Dict, i: int, parent):
        logging.info(f"新标题{features['content']}深度为{features['depth']}")
        if useen:
            outlines.append([features['content'], en_outlines[i], parent])
        else:
            outlines.append([features['content'], "", parent])
    
    for i, features in enumerate(dir_features):
        logging.info(f"处理标题{features['content']}")
        if parent_dic == {}:
            features['depth']=0
            addoutline(features, i, [])
            parent_dic = features
            pattern_tree.append([[features['type'], features['dot_count']]])
            continue
        #处理附录附件情况：固定放在最高层
        if features['type'] in ['plain','appendix','annex']:
            if [features['type'], features['dot_count']] not in pattern_tree[0]:
                pattern_tree[0].append([features['type'], features['dot_count']])
            features['depth']=0
            addoutline(features, i, [])
            parent_dic = features
            continue
        #标题种类相同，点数区分点数层级（1,1.1）
        if features['type'] == parent_dic['type']:
            if features['dot_count'] == parent_dic['dot_count']:
                features['depth'] = parent_dic['depth']
                addoutline(features, i, outlines[-1][2])
                parent_dic = features
                continue
            elif features['dot_count'] < parent_dic['dot_count']:
                ndepth = parent_dic['depth']-parent_dic['dot_count']+features['dot_count']
                exists = False
                for j, patterns in enumerate(pattern_tree[:ndepth+1]):
                    if [features['type'], features['dot_count']] in patterns:
                        ndepth = j
                        exists = True
                        break
                if not exists:
                    pattern_tree[ndepth].append([features['type'], features['dot_count']])
                features['depth'] = ndepth
                addoutline(features, i, outlines[-1][2][:ndepth])
                parent_dic = features
                continue
            else:
                ndepth = parent_dic['depth']-parent_dic['dot_count']+features['dot_count']
                if len(pattern_tree) <= ndepth:
                    for j in range(len(pattern_tree),ndepth+1):
                        pattern_tree.append([])
                exists = False
                for patterns in pattern_tree[ndepth:]:
                    if [features['type'], features['dot_count']] in patterns:
                        exists = True
                        break
                if not exists:
                    pattern_tree[ndepth].append([features['type'], features['dot_count']])
                #if [features['type'], features['dot_count']] not in pattern_tree[parent_dic['depth']+1]:
                    #pattern_tree[ndepth].append([features['type'], features['dot_count']])
                features['depth'] = ndepth
                addoutline(features, i, outlines[-1][2]+[outlines[-1][:2]])
                parent_dic = features
                continue
        else:
            tmppattern = [features['type'], features['dot_count']]
            exists, tmpdepth = False, parent_dic['depth']+1
            for j, patterns in enumerate(pattern_tree[parent_dic['depth']+1:]):
                if tmppattern in patterns:
                    exists = True
                    tmpdepth = j+parent_dic['depth']+1
                    break
            if exists:
                features['depth'] = tmpdepth
                addoutline(features, i, outlines[-1][2]+[outlines[-1][:2]])
                parent_dic = features
                continue

            check = False
            if "parentthese" in features["type"]:
                check = checkFirst(features['inside'])
            else:
                check = checkFirst(features['prefix'])
            if check: #子层级
                if tmppattern in pattern_tree[parent_dic['depth']]:
                    pattern_tree[parent_dic['depth']].remove(tmppattern)
                if len(pattern_tree) == parent_dic['depth']+1:
                    pattern_tree.append([tmppattern])
                    features['depth'] = parent_dic['depth']+1
                else: 
                    exists = False
                    for j, patterns in enumerate(pattern_tree[parent_dic['depth']+1:]):
                        if tmppattern in patterns:
                            features['depth'] = j+parent_dic['depth']+1
                            exists = True
                            break
                    if not exists:
                        pattern_tree[parent_dic['depth']+1].append(tmppattern)
                        features['depth'] = parent_dic['depth']+1
                addoutline(features, i, outlines[-1][2]+[outlines[-1][:2]])
                parent_dic = features
                continue
            else: #上级（默认相邻同级标题除了第一级一定是同类型的，已经处理过了）
                for j, patterns in enumerate(pattern_tree[:parent_dic['depth']]):
                    if tmppattern in patterns:
                        features['depth'] = j
                        addoutline(features, i, outlines[-1][2][:j])
                        parent_dic = features
                        break
            if features['depth'] != None:
                continue
            logging.warning(f"未识别标题{features['content']},上级标题为{parent_dic}，当前格式树为{pattern_tree}，检查原因")

    for i, outline in enumerate(outlines):
        j = dir_features[i]['depth']
        logging.debug(f"处理标题格式{[dir_features[i]['type'], dir_features[i]['dot_count']]}")
        for patterns in pattern_tree[dir_features[i]['depth']:]:
            if [dir_features[i]['type'], dir_features[i]['dot_count']] in patterns:
                break
            j+=1
        logging.debug(f"{outline[0]}记录深度为{dir_features[i]['depth']},目前检测深度为{j}")
        if j>dir_features[i]['depth']:
            n=0
            while n<j:
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
            dir_features[i]['depth'] = j
    return outlines, pattern_tree,dir_features

def checkFirst(head):
    matches = list(re.finditer(r'\.', head))
    if matches == []:
        start = 0
    else:
        start = matches[-1].end()
        if start == len(head):
            if len(matches) == 1:
                return re.match(r"^[1一壹aAIⅠiⅰ\u2460]$",head[:-1])
            return re.match(r"^[1一壹aAIⅠiⅰ\u2460]$",head[matches[-2].end():-1])

    return re.match(r"^[1一壹aAIⅠiⅰ\u2460]$",head[start:])

'''
def print_hierarchy(hierarchy: List[Dict]):
    """可视化层级结构（用缩进表示层级）"""
    print("推断的目录层级结构：")
    for item in hierarchy:
        indent = '  ' * item['level']  # 每级缩进2个空格
        print(f"{indent}- {item['dir_str']}")
'''

# 示例用法
if __name__ == "__main__":
    sample_directories = [
        "1. 第一章", "1.1 引言" , "1.1.1 研究背景",
        "1.2 方法", "a 方法一",
        "2. 第二章", "2.1 实验设计", "2.1.1 简介", 
        "2.2 结果分析", "a 统计", "2.2.1 数据统计",
        "3. 第三章", "3.1 讨论", 
        "附录A 补充材料", "附录B 荣誉证书",
        "参考文献"  
    ]
    
    # 推断层级
    hierarchy, pattern_tree = infer_hierarchy(sample_directories, useen=False)
    
    # 打印结果
    for item in hierarchy:
        print(item)
        
    for pattern in pattern_tree:
        print(pattern)


