from detailed.utils import fuzzy_match
from detailed.tablerecognizer import merge_chunk_through_outlines
import sys
sys.path.append('D:/pythonprojects/RAG_prototype')

# 模拟一个简单的split对象
class MockSplit:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# 测试实际的page_content内容
title = '6.2 A 类钢筋混凝土房屋抗震鉴定'
text_with_source = '6.2 A 类钢筋混凝土房屋抗震鉴定\n\nsource: GB50023-2009：建筑抗震鉴定标准.md'
text_without_source = '6.2 A 类钢筋混凝土房屋抗震鉴定'

print("=== 测试带source的情况 ===")
result1 = fuzzy_match(title, text_with_source)
print(f'匹配结果: {result1}')
print(f'标题长度: {len(title)}')
if result1[0]:
    outlineend = result1[1]
    print(f'outlineend: {outlineend}')
    print(f'文本前{outlineend}个字符: "{text_with_source[:outlineend]}"')
    print(f'从outlineend开始的剩余内容: {repr(text_with_source[outlineend:])}')
    print(f'剩余内容strip后: {repr(text_with_source[outlineend:].strip())}')

print("\n=== 测试不带source的情况 ===")
result2 = fuzzy_match(title, text_without_source)
print(f'匹配结果: {result2}')
if result2[0]:
    outlineend = result2[1]
    print(f'outlineend: {outlineend}')
    print(f'文本前{outlineend}个字符: "{text_without_source[:outlineend]}"')
    print(f'从outlineend开始的剩余内容: {repr(text_without_source[outlineend:])}')
    print(f'剩余内容strip后: {repr(text_without_source[outlineend:].strip())}')