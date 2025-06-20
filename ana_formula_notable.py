from typing import List, Dict, Any
import re

def is_heading(line: str, heading_levels: List[int]) -> tuple[bool, int, str]:
    """
    判断一行是否是指定深度的标题，并返回标题级别和标题文本。

    :param line: Markdown 中的一行文本
    :param heading_levels: 需要识别的标题深度列表，如 [1, 2]
    :return: (是否匹配, 标题深度, 标题文本)
    """
    if not line.startswith('#'):
        return False, 0, ""

    # 计算标题深度（即 '#' 的数量）
    level = 0
    for char in line:
        if char == '#':
            level += 1
        else:
            break

    # 判断是否在配置的深度列表中
    if level in heading_levels:
        # 提取标题文本（去掉 '#' 和空格）
        title = line[level:].strip()
        return True, level, title
    else:
        return False, level, ""

def read_md_by_heading_depth(file_path: str, heading_levels: List[int]) -> Dict[str, Any]:
    """
    按照指定的标题深度读取 Markdown 文件，并返回结构化数据。

    :param file_path: Markdown 文件路径
    :param heading_levels: 需要读取的标题深度列表，如 [1, 2]
    :return: 结构化的字典树
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    root = {"title": "root", "heading_level": 0, "content": "", "children": []}
    stack = [root]  # 使用栈来维护当前所在的层级
    current_content = ""  # 当前累积的内容

    for line in lines:
        line = line.rstrip('\r\n')  # 去掉换行符
        is_h, level, title = is_heading(line, heading_levels)

        if is_h:
            # 如果当前有累积的内容，添加到栈顶节点
            if current_content.strip():
                stack[-1]["content"] += current_content + "\n"
                current_content = ""

            # 如果当前标题深度小于等于栈顶的深度，则需要回退栈
            while stack and level <= stack[-1]["heading_level"]:
                stack.pop()

            # 创建新的节点
            new_node = {
                "title": title,
                "heading_level": level,
                "content": "",
                "children": []
            }

            # 将新节点添加到当前栈顶的 children 中
            stack[-1]["children"].append(new_node)

            # 将新节点压入栈
            stack.append(new_node)
        else:
            # 如果不是标题行，累积内容
            current_content += line + "\n"

    # 处理文件末尾可能剩余的内容
    if current_content.strip():
        stack[-1]["content"] += current_content

    return root

def extract_chapter_prefix(text: str):
    """
    在文本中查找符合 XX--XX--XX 格式的字符串，并将其拆分为三段数字的元组列表。

    :param text: 输入的文本字符串
    :return: 匹配到的三段数字的元组列表，如 [(12, 34, 56), (1, 23, 45)]
    """
    # 定义正则表达式模式
    pattern = r'\d+-\d+-\d+'

    # 编译正则表达式
    regex = re.compile(pattern)

    # 查找所有匹配项
    matches = regex.findall(text)

    # 将匹配到的字符串拆分为三段数字，并转换为整数元组
    result = ""
    for match in matches:
        part1, part2, part3 = match.split('-')  # 按 -- 分割字符串
        num1, num2, num3 = int(part1), int(part2), int(part3)  # 转换为整数
        result = result + str(num1)+"-" + str(num2)+"-" + str(num3) 
        break

    return result

def print_structure(file_prefix, node: Dict[str, Any], indent: int = 0):
    """
    打印结构化数据，方便查看结果
    """
    prefix = "  " * indent
    
    
    
    # print(f"{prefix}📌 {node['title']} (Level {node.get('heading_level', 0)})")
    if node["content"].strip():
        
        # print ("length is: ", len(node["content"]))
        # print(f"{prefix}   node['title'], 💬 {node['content'][:50].replace('\n', ' ')}...")  # 只打印前50个字符
        a = analyze_formulas_outside_tables(node["content"])
        # print ("contains formula? ", a)
        # print(a)
        if a['formula_outside_tables_ratio'] > 0.3:
            print(len(node['content']), a['formula_outside_tables_ratio'] , node['title'], f"{prefix}   💬 {node['content'][:50].replace('\n', ' ')}...")
            # chapter_prefix = node['title'].replace(".","-")
            chapter_prefix = extract_chapter_prefix(node['title'].replace(".","-"))
            save_path = "./sub_md/" + file_prefix + "_" + chapter_prefix + ".md"
            with open(save_path, 'w', encoding='utf-8') as fout:
                fout.write(node['content'])
    for child in node["children"]:
        print_structure(file_prefix, child, indent + 1)

def analyze_formulas_outside_tables(md_text: str) -> Dict[str, Any]:
    """
    分析 Markdown 文本中的公式，判断是否存在公式且公式不在表格内。

    :param md_text: Markdown 格式的字符串
    :return: 包含分析结果的字典
    """
    lines = md_text.split('\n')
    n = len(lines)

    # 初始化变量
    in_table = False  # 是否在表格中
    table_start_line = -1  # 表格起始行
    table_end_line = -1    # 表格结束行

    # 用于存储表格外的公式信息
    formulas_outside_tables = []  # 存储公式所在的行号

    # 第一步：识别表格范围
    for i, line in enumerate(lines):
        if line.strip().startswith('|'):
            # 如果当前行以 | 开头，认为进入表格
            if not in_table:
                in_table = True
                table_start_line = i
        else:
            # 如果当前行不以 | 开头，且之前是在表格中，则认为表格结束
            if in_table:
                in_table = False
                table_end_line = i - 1

    # 处理文件末尾可能仍在表格中的情况
    if in_table:
        table_end_line = n - 1

    # 第二步：在表格范围之外查找公式
    in_table_range = False
    for i, line in enumerate(lines):
        # 判断当前行是否在表格范围内
        if table_start_line != -1 and table_end_line != -1:
            if table_start_line <= i <= table_end_line:
                in_table_range = True
            else:
                in_table_range = False

        if not in_table_range:
            # 当前行不在表格范围内，检查是否包含公式
            if is_formula_in_line(line):
                formulas_outside_tables.append(i)

    # 统计表格外的公式数量
    num_formulas_outside_tables = len(formulas_outside_tables)

    # 判断是否存在表格外的公式
    contains_formula_outside_tables = num_formulas_outside_tables > 0
    formula_ratio = float(num_formulas_outside_tables /n)

    # 返回结果
    return {
        "contains_formula_outside_tables": contains_formula_outside_tables,
        "formula_outside_tables_ratio": formula_ratio,
        "num_formulas_outside_tables": num_formulas_outside_tables,
        "formulas_outside_tables_lines": formulas_outside_tables
    }

def is_formula_in_line(line: str) -> bool:
    """
    判断一行中是否包含数学公式（$...$ 或 $$...$$），不使用正则表达式。

    :param line: 一行文本
    :return: 如果包含公式返回 True，否则返回 False
    """
    # 检查块级公式 $$...$$
    dollar_dollar_index = line.find('$$')
    while dollar_dollar_index != -1:
        # 找到 $$ 的起始位置
        start = dollar_dollar_index
        end = line.find('$$', start + 2)
        if end != -1:
            # 找到了闭合的 $$，确认是块级公式
            return True
        else:
            # 没有找到闭合的 $$，退出循环
            break
        dollar_dollar_index = line.find('$$', end + 2)

    # 检查行内公式 $...$
    dollar_index = line.find('$')
    while dollar_index != -1:
        # 找到 $ 的起始位置
        start = dollar_index
        end = line.find('$', start + 1)
        if end != -1:
            # 找到了闭合的 $，确认是行内公式
            return True
        else:
            # 没有找到闭合的 $，退出循环
            break
        dollar_index = line.find('$', end + 1)

    # 如果没有找到任何公式，返回 False
    return False

if __name__ == "__main__":
    md_file_path = "./b19159_fix/b19159.md"
    heading_levels_to_read = [3]
    structured_data = read_md_by_heading_depth(md_file_path, heading_levels_to_read)
    prefix = md_file_path.split("/")[-1][:-3]
    print ("prefix is: ", prefix)
    print_structure(prefix, structured_data)    