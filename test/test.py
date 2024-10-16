import re

def reassign_qxx_labels(code):
    # 使用正则表达式匹配所有的 Qxx 指令
    qxx_pattern = re.compile(r'Q(\d{1,2})')
    matches = qxx_pattern.findall(code)

    # 将找到的 Qxx 指令去重并排序
    unique_qxx = sorted(set(matches), key=lambda x: int(x))

    # 创建一个映射表，将原来的 Qxx 映射到新的编号
    qxx_mapping = {old: new for new, old in enumerate(unique_qxx)}

    # 定义替换函数
    def replace_qxx(match):
        old_qxx = match.group(1)
        new_qxx = qxx_mapping[old_qxx]
        return f'Q{new_qxx}'

    # 使用正则表达式替换原代码中的 Qxx 指令
    new_code = qxx_pattern.sub(replace_qxx, code)

    return new_code

# 示例代码
code = """
CZ Q1 Q26
Q1
Q26
Q14
CZ Q14 Q1
"""

# 调用函数
new_code = reassign_qxx_labels(code)
print(new_code)
