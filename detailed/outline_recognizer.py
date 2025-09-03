import re
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

def extract_features(dir_str: str) -> Dict:
    """提取目录项的层级特征：前缀类型、点数量、基值等"""
    # 匹配数字前缀（如1.、2.1、3.1.1）
    num_match = re.match(r'^(\d+)(\.\d+)*\s+', dir_str)
    # 匹配字母前缀（如a 、b ）
    alpha_match = re.match(r'^([a-zA-Z])\s+', dir_str)

    parenttheses_match = re.match(r'^([（\(][^）\)]+[）\)])\s+', dir_str)
    
    right_parenttheses_match = re.match("^.{1,2}[）\)]", dir_str)
    
    if num_match:
        prefix = num_match.group(0)
        # 点数量（1. 含0个点，1.1 含1个点，用于表示层级深度）
        dot_count = prefix.count('.')
        # 基值（最前面的数字，用于匹配父目录）
        base_num = int(re.findall(r'\d+', prefix)[0])
        return {
            'type': 'number',
            'prefix': prefix,
            'dot_count': dot_count,
            'base_num': base_num,
            'content': dir_str[len(prefix):]
        }
    elif alpha_match:
        prefix = alpha_match.group(0)
        return {
            'type': 'alpha',
            'prefix': prefix,
            'dot_count': None,  # 字母前缀通常依赖上下文
            'base_num': None,
            'content': dir_str[len(prefix):]
        }
    else:
        # 无明显前缀（如"参考文献"）
        return {
            'type': 'plain',
            'prefix': '',
            'dot_count': -1,  # 标记为顶级
            'base_num': None,
            'content': dir_str
        }

def infer_hierarchy(directories: List[str]) -> List[Dict]:
    """根据特征和顺序推断目录层级，返回带层级信息的列表"""
    # 提取所有目录的特征
    dir_features = [extract_features(dir_str) for dir_str in directories]
    
    # 存储每个目录的层级（0为顶级）和父目录索引
    result = []
    # 用栈记录可能的父目录（按层级从深到浅）
    parent_stack = []
    
    for i, features in enumerate(dir_features):
        dir_str = directories[i]
        level = 0
        parent_idx = -1  # -1表示无父目录
        
        if features['type'] == 'number':
            # 数字前缀目录：通过dot_count和base_num找父目录
            current_dot = features['dot_count']
            current_base = features['base_num']
            
            # 栈中找最近的、dot_count = current_dot - 1且base_num匹配的目录
            for j in reversed(range(len(parent_stack))):
                parent_idx_candidate, parent_features = parent_stack[j]
                if (parent_features['type'] == 'number' and 
                    parent_features['dot_count'] == current_dot - 1 and 
                    parent_features['base_num'] == current_base):
                    level = parent_features['dot_count'] + 1
                    parent_idx = parent_idx_candidate
                    break
            # 如果没找到父目录，默认顶级（dot_count=0时通常为顶级）
            if level == 0:
                level = current_dot + 1  # dot_count=0 → level=1（顶级）
        
        elif features['type'] == 'alpha':
            # 字母前缀目录：通常是前一个目录的子目录
            if i > 0:
                prev_features = dir_features[i-1]
                level = result[i-1]['level'] + 1
                parent_idx = i - 1
        
        else:  # plain类型
            # 无前缀目录：默认顶级，或根据前一个目录层级调整
            level = 0
        
        # 更新结果和父栈
        result.append({
            'dir_str': dir_str,
            'level': level,
            'parent_idx': parent_idx,
            'features': features
        })
        
        # 维护父栈：只保留当前目录之前、可能作为后续目录父级的项
        # 清除栈中层级 >= 当前层级的项（它们不可能是后续目录的父级）
        while parent_stack and parent_stack[-1][1]['dot_count'] >= features.get('dot_count', -1):
            parent_stack.pop()
        parent_stack.append((i, features))
    
    return result

def print_hierarchy(hierarchy: List[Dict]):
    """可视化层级结构（用缩进表示层级）"""
    print("推断的目录层级结构：")
    for item in hierarchy:
        indent = '  ' * item['level']  # 每级缩进2个空格
        print(f"{indent}- {item['dir_str']}")

# 示例用法
if __name__ == "__main__":
    sample_directories = [
        "1. 第一章", "1.1 引言" , "1.1.1 研究背景",
        "1.2 方法", "a 方法一",
        "2. 第二章", "2.1 实验设计", "a 设计思路", "2.1.1 简介", 
        "2.2 结果分析", "a 统计", "2.2.1 数据统计",
        "3. 第三章", "3.1 讨论", 
        "附录A 补充材料", "附录B 荣誉证书",
        "参考文献"  
    ]
    
    # 推断层级
    hierarchy = infer_hierarchy(sample_directories)
    
    # 打印结果
    print_hierarchy(hierarchy)
    
    # 额外输出各层级的格式特征
    print("\n各层级格式特征：")
    level_formats = defaultdict(list)
    for item in hierarchy:
        level_formats[item['level']].append(item['features']['type'])
    
    for level in sorted(level_formats.keys()):
        types = level_formats[level]
        type_counts = defaultdict(int)
        for t in types:
            type_counts[t] += 1
        print(f"层级{level}：{dict(type_counts)}（示例：{[item['dir_str'] for item in hierarchy if item['level']==level][:2]}）")


