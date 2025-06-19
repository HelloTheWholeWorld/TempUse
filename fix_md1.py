from pathlib import Path
def fix_markdown_headings(md_text):
    """
    修正Markdown文本中的标题等级，使其编号与正确的标题等级匹配
    
    参数:
        md_text (str): 输入的Markdown文本
        
    返回:
        str: 修正后的Markdown文本
    """
    lines = md_text.split('\n')
    result = []
    
    for line in lines:
        #print ("Processing: ", line)
        if line.startswith('#'):
            # 计算当前标题的等级
            # print ("bef replace the line is: ", line)
            # line = line.replace(" ","")
            line = line.replace("．",".").replace(" . ",".").replace(". ",".").replace(" .",".")
            
            heading_level = 0
            # print ("the line start with # is: ", line)
            # print ("len of line is: ", len(line))
            while heading_level < len(line) and line[heading_level] == '#':
                heading_level += 1
            # print ("heading_level is: ", heading_level)
            
            # 提取标题文本
            heading_text = line[heading_level:].lstrip()
            # print ("heading_text is: ", heading_text)
            # 检查标题是否以数字编号开头
            if heading_text and heading_text[0].isdigit():
                # 分割数字部分和剩余文本
                num_part = []
                rest_part = []
                in_num_part = True
                
                for char in heading_text:
                    if in_num_part and (char.isdigit() or char == '.'):
                        num_part.append(char)
                    else:
                        in_num_part = False
                        rest_part.append(char)
                
                num_str = ''.join(num_part).strip()
                rest_str = ''.join(rest_part).strip()
                # print ("num_str is: ",num_str)
                # print ("rest_str is: ",rest_str)
                
                # 计算数字编号的深度(通过点的数量)
                if '.' in num_str:
                    num_depth = num_str.count('.') + 1
                else:
                    num_depth = 1
                
                # 调整标题等级
                new_heading_level = min(6, max(1, num_depth))  # 确保在1-6范围内
                # print ("the adjusted level is: ", new_heading_level)
                # 重建标题行
                new_line = '#' * new_heading_level + ' ' + num_str + ' ' + rest_str
                # print ("the new_line is:", new_line)
                # print ("=================================================================")
                result.append(new_line)
                continue
        
        # 如果不是标题或不需要修改，直接添加
        result.append(line)
    
    return '\n'.join(result)

# if __name__ == "__main__":
#     sample_md = """
# # 1 一级标题
# ## 1.1 二级标题
# ### 1.1.1 三级标题
# #### 1.1.1.1 四级标题
# ##### 1.1.1.1.1 五级标题
# ###### 1.1.1.1.1.1 六级标题
# ####### 1.1.1.1.1.1.1 不应该存在的七级标题
# ## 2 另一个二级标题
# ### 2.1 三级标题
# """

#     fixed_md = fix_markdown_headings(sample_md)
#     print(fixed_md)

def batch_fix_md_files(dir_path):
    for md_file in Path(dir_path).glob("*.md"):
        print ("processing: ", md_file)
        content = md_file.read_text(encoding="utf-8")
        fixed_text =fix_markdown_headings(content)
        # print (fixed_text)
        new_path = "./b19159_fix/" + str(md_file).split("\\")[-1]
        with open(new_path, 'w', encoding='utf-8') as file:
            file.write(fixed_text)
        # formatted_text = format_heading_numbering(fixed_text)
        # print (formatted_text )
        # fixed_content = fix_md_headings(content)
        #md_file.write_text(fixed_content, encoding="utf-8")

# 示例：修正当前目录下所有MD文件
batch_fix_md_files("./b19159_new/")