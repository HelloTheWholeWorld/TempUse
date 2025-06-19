from typing import List, Dict, Any

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

def print_structure(node: Dict[str, Any], indent: int = 0):
    """
    打印结构化数据，方便查看结果
    """
    prefix = "  " * indent
    print(f"{prefix}📌 {node['title']} (Level {node.get('heading_level', 0)})")
    if node["content"].strip():
        print(f"{prefix}   💬 {node['content'][:50].replace('\n', ' ')}...")  # 只打印前50个字符
    for child in node["children"]:
        print_structure(child, indent + 1)

# ============================
# 示例用法
# ============================

if __name__ == "__main__":
    # 替换为你的 Markdown 文件路径
   
    # 配置需要读取的标题深度，比如只读取 h1 和 h2
    heading_levels_to_read = [3]
    md_file_path = "./b19159_fix/b19159.md"

    # 读取并解析 Markdown 文件
    structured_data = read_md_by_heading_depth(md_file_path, heading_levels_to_read)

    # 打印结构化结果
    print_structure(structured_data)
