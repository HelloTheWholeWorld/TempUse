import jsonlines
import time
import os
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama  # 更新导入路径
from langchain_core.output_parsers import StrOutputParser
import tqdm
import json
import re
# 1. 创建提示词模版





def doc2qa(paper_text):
    try:
        prompt_template = PromptTemplate.from_template("Attached is a detailed wikipedia article.\n\n{text}\n\nYour task is to formulate up to a maximum of {qa_num} sophisticated Q&A pairs that delve into the underlying knowledge presented in this article. Focus on questions that require a deeper understanding of the subject matter, especially those  relating to the the main concepts, key findings, and overall themes of the article.Ensure diversity in your Q&A pairs, avoiding any duplication. Answers should be rich in detail. Strive for clarity and depth in your responses. Return the generated questions and answers in JSON list format, where each item in the list contains two keys: the question is placed under the key 'Q', and the answer is placed under the key 'A'. Do not use any knowledge outside the provided text. If you find that the provided text is insufficient to generate the required number of QA pairs,then generate fewer questions,with a minimum of 1. Please strictly output in JSON format, do not output irrelevant content. "
)
        #{[{\"Q\":\"为什么...\" , \"A\":\"因为...\"},{\"Q\":\"Quesiton\",\"A\":\"Answer\"}]}
        prompt_template = PromptTemplate.from_template(

        """
        附上一篇详细的维基百科文章。

        {text}

        你的任务是根据这篇文章生成尽可能多的问题和答案，以便作为知识用于大模型预训练语料， 关注文章每句表述里的知识点，所有的概念、关键发现和整体主题相关的问题。确保你的问答对具有多样性，避免重复。答案应详细且富有细节。力求在回答中体现清晰性和深度，以增强读者对文章中复杂概念的理解。

        请注意：
        问题应避免使用“这篇文章”或类似的指代，而是应追求通用知识，使其适用于更广泛的背景。 答案也应避免提及“这篇文章”，而是直接针对问题提供详细的解释和分析。
        生成的JSON结构中，文本中的双引号必须被单引号替代，以确保JSON格式的正确性。

        请以JSON列表格式返回生成的问题和答案，列表是一个字符串列表，每一项是一个字符串而不是一个字典，问题和答案连接为一个完整的单行字符串，字符串格式如:"Question:...?Answer:..."
        ***示例输出JSON格式***
        ["Question:...Answer:...","Question:...Answer:...","Question:...Answer:..."]


        请严格以JSON格式输出，且所有内容必须为中文。
        """

        )
        prompt_template = PromptTemplate.from_template(

        """
        附上一篇详细的维基百科文章。

        {text}

        你的任务是根据这篇文章生成尽可能多的问题和答案，以便作为知识用于大模型预训练语料， 关注文章每句表述里的知识点，所有的概念、关键发现和整体主题相关的问题。确保你的问答对具有多样性，避免重复。答案应简洁清晰且有细节。

        请注意：
        问题应避免使用“这篇文章”或类似的指代，而是应追求通用知识，使其适用于更广泛的背景。 答案也应避免提及“这篇文章”，而是直接针对问题提供详细的解释和分析。

        返回的多个问答对,每个回答对单独成一行，问答对之间分行。
        ***示例输出格式***
        Question:...Answer:...
        Question:...Answer:...
        Question:...Answer:...

        请严格以上述格式输出，且所有内容必须为中文,并且不要输出额外内容，尤其不要输出思考过程。
        """

        )



        # 格式化提示
        formatted_prompt = prompt_template.format(text=paper_text)

        model_name = "qwen2.5:7b"
        model_name = "qwen3:8b"
        # 2. 创建模型
        #   gpu_num   参数用于指定模型在推理过程中使用的 GPU 数量。这个参数特别适用于多 GPU 系统，可以帮助你更高效地利用多个 GPU 的计算能力。
        chatLLM = ChatOllama(model=model_name)  # 更新为 ChatOllama
        # 3. 创建解析器

        # 调用 LLM 并生成回答
        response = chatLLM.invoke(formatted_prompt)
        parser = StrOutputParser()
        parsed_content_old = parser.invoke(response)  # 使用解析器提取内容
        # 案例："Question: "数学"一词的起源和意义是什么？Answer: 古希腊语中的"μάθημα"(máthēma)意为“学习”、“学问”或“科学”，在西塞罗翻译到拉丁文时，被用作形容词"mathēmatikós"，指与学习有关的或是用功的意思。",
        # 4. 创建链
        # 此时生成的格式带:```json 和```
        #parsed_content = parsed_content.replace("```json","").replace("```","")
        # 使用正则表达式匹配并删除 <think> 和 </think> 之间的内容
        parsed_content_old = re.sub(r'<think>.*?</think>', '', parsed_content_old)
        parsed_content_old = parsed_content_old.replace("<think>","")
        parsed_content_old = parsed_content_old.replace("</think>","")
        # 努尔哈赤在什么战役中被明朝将领袁崇焕击败，进而逝世？ Answer：1626年，在与明袁崇焕交战的宁远战役中受挫后数月，努尔哈赤逝世。
        parsed_content = parsed_content_old.replace('"','\"').replace("Question：","Question:").replace("Answer：","Answer:")

        #Cyc的名称来源是什么？ Answer"Cyc"名字的来源是"encyclopedia"，发音很像"syke"，是美国德克萨斯州奥斯丁的Cycorp公司的有效注册商标。
        #因为我们是处理中文语料，所以快
        if "Answer:" not in parsed_content and "Answer" in parsed_content:
            parsed_content = parsed_content.replace("Answer","Answer:")

        parsed_content_list = parsed_content.split("Question:")

        parsed_content = parsed_content_old.replace('"','\"')

        parsed_content_json = []
        for qa in parsed_content_list:
            qa = qa.replace("\n"," ")
            if len(qa) < 10:
                continue
            if "Answer:" not in qa:
                print("bad qa3:",qa)
                continue
            qa_splits = qa.split("Answer:")
            if len(qa_splits) != 2:
                print("bad qa4:",qa)
                continue
            qa_question = qa_splits[0]
            qa_answer = qa_splits[1]
            qa_question = qa_question.strip()
            qa_answer = qa_answer.strip()
            parsed_content_json.append({"Q":qa_question,"A":qa_answer})
        return parsed_content_json 
    except Exception as e:
        print(parsed_content)
        print(e)
        return []

import jsonlines
import multiprocessing

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


CHUNK_SIZE = 2000
CHUNK_OVERLAP_SIZE = 100

# 每个CHUNK的最少的文本数
CHUNK_MINIMUM_SIZE = 100 

# 提取单个 JSON 对象中的 mmdpath 字段
def extract_mmdpath(item):
    time0 = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP_SIZE,
        keep_separator=True,
        separators=[
            "\n\n",
            "\n",
            ".",
            "\uff0e",  # Fullwidth full stop
        ],
        # Existing args
    )
    # 分割文本
    text = item["text"]
    text_size = len(text)
    chunks = text_splitter.split_text(text)
    text_id = item["id"]


    new_chunks = []
    cur_chunk = ""
    for i, chunk in enumerate(chunks):
        cur_chunk += chunk
        if len(cur_chunk) > CHUNK_MINIMUM_SIZE:
            new_chunks.append(cur_chunk)
            cur_chunk = ""

    # 打印分割后的片段
    # 如果wiki的原文很多换行，那么会出现chunk 太小的情况,这时候要合并多个chunk来回答
    qa_total = []
    for i, chunk in enumerate(new_chunks):
        if len(chunk) < CHUNK_MINIMUM_SIZE:
            print(f"text_id {text_id}, chunk too small")
            continue
        qa_chunk = doc2qa(chunk)
        qa_total.extend(qa_chunk)

    time1 = time.time()
    time_cost = time1-time0
    print(f"text_id:{text_id} text size: {text_size} time cost: {time_cost:.2f}   chunk: {len(chunks)}  qa_n: {len(qa_total)}")
    item["qas"] = qa_total   
    return item



import jsonlines
import tqdm
import argparse

# 定义命令行参数解析
parser = argparse.ArgumentParser(description='Process a JSON Lines file and generate QA pairs.')
parser.add_argument('input_file', type=str, help='Path to the input JSON Lines file')
args = parser.parse_args()

# 从命令行参数获取输入文件路径
input_file = args.input_file

# 根据输入文件路径生成输出文件路径
output_file = input_file.replace(".json", "_qa.json")
# 定义文件路径


# 主函数
def main():
    # 使用 jsonlines 读取文件中的所有 JSON 对象，并边处理边保存
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode="w") as writer:
        for item in tqdm.tqdm(reader):
            # 处理每个 JSON 对象
            processed_item = extract_mmdpath(item)
            # 将处理后的 JSON 对象写入输出文件
            writer.write(processed_item)




# 运行主函数
if __name__ == "__main__":
    main()
    
