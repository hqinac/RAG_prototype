from detailed.utils import fuzzy_match

# 测试修改后的fuzzy_match函数
outline = '9.2 A类单层砖柱厂房抗震鉴定'
split = '9.2 A 类单层砖柱厂房抗震鉴定'

print(f"outline: '{outline}'")
print(f"split: '{split}'")
print(f"outline长度: {len(outline)}")
print(f"split长度: {len(split)}")

# 测试fuzzy_match函数
print("\n测试fuzzy_match函数:")
result = fuzzy_match(outline, split)
print(f"fuzzy_match结果: {result}")

if result[0]:
    end_pos = result[1]
    print(f"匹配成功，end位置: {end_pos}")
    print(f"split[:end_pos]: '{split[:end_pos]}'")
    print(f"split[end_pos:]: '{split[end_pos:]}'")
else:
    print("匹配失败")

# 测试另一个案例
print("\n测试另一个案例:")
outline2 = '6.2 A 类钢筋混凝土房屋抗震鉴定'
split2 = '6.2 A 类钢筋混凝土房屋抗震鉴定：定'
result2 = fuzzy_match(outline2, split2)
print(f"fuzzy_match结果: {result2}")

if result2[0]:
    end_pos2 = result2[1]
    print(f"匹配成功，end位置: {end_pos2}")
    print(f"split2[:end_pos2]: '{split2[:end_pos2]}'")
    print(f"split2[end_pos2:]: '{split2[end_pos2:]}'")
else:
    print("匹配失败")