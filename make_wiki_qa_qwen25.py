import jsonlines
import time
import os
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama  # 更新导入路径
from langchain_core.output_parsers import StrOutputParser
import tqdm
import json

# 1. 创建提示词模版


import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
import tqdm
import argparse
import jsonlines
import multiprocessing

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 定义命令行参数解析
parser = argparse.ArgumentParser(description='Process a JSON Lines file and generate QA pairs.')
parser.add_argument('input_gpu', type=str, help='GPU  ID')
parser.add_argument('input_file', type=str, help='Path to the input JSON Lines file')
args = parser.parse_args()

# 从命令行参数获取输入文件路径
input_file = args.input_file

# 根据输入文件路径生成输出文件路径
output_file = input_file.replace(".json", "_qa.json")
# 定义文件路径

if os.path.exists(output_file):
    print("skip ",output_file)
    exit() 

GPU_ID = int(args.input_gpu)%4

model_name = "~/models/Qwen2.5-0.5B-Instruct"
model_name = "~/models/Qwen2.5-3B-Instruct"
model_name = "/mnt/gpudisk/models_home/Qwen2.5-7B-Instruct"

# 指定运行模型的 GPU 设备
device = torch.device("cuda:"+str(GPU_ID))  # 假设你想在第一个 GPU 上运行模型

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    #device_map="auto",
    device_map={"": device}  # 将模型加载到指定的 GPU 设备
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def doc2qa(paper_text):
    try:
        system_prompt = """ 
        你是一个你是从文本中提取问答对的专家。你的任务是从提供的文本中识别并提取问题及其对应的答案。确保提取的问答对准确且完整。用户提供的是一篇维基百科文章的片段，你的任务是根据这篇文章片段生成尽可能多的问题和答案，以便作为知识用于大模型预训练语料， 关注文章每句表述里的知识点，所有的概念、关键发现和整体主题相关的问题。确保你的问答对具有多样性，避免重复。答案应简洁清晰且有细节。

        请注意：
        问题应避免使用“这篇文章”或类似的指代，而是应追求通用知识，使其适用于更广泛的背景。 答案也应避免提及“这篇文章”，而是直接针对问题提供详细的解释和分析。

        返回的多个问答对,每个回答对单独成一行，问答对之间分行。
        ***示例输出格式***
        Question:...Answer:...
        Question:...Answer:...
        Question:...Answer:...

        请严格以上述格式输出，且所有内容必须为中文,并且不要输出额外内容。
        """

        # 对输入文本进行分词处理

        
        # "You are an expert in extracting QA pairs from text. Your task is to identify and extract questions and their corresponding answers from the provided text. Ensure that the extracted QA pairs are accurate and complete."

        messages = [
            {"role": "system","content": system_prompt},
            {"role": "user","content": paper_text}
        ]


        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # 为了让输出的QA很多，所以要让输出变很多
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=5000
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]


        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 调用 LLM 并生成回答
        parser = StrOutputParser()
        parsed_content_old = parser.invoke(response)  # 使用解析器提取内容
        # 案例："Question: "数学"一词的起源和意义是什么？Answer: 古希腊语中的"μάθημα"(máthēma)意为“学习”、“学问”或“科学”，在西塞罗翻译到拉丁文时，被用作形容词"mathēmatikós"，指与学习有关的或是用功的意思。",
        # 4. 创建链
        # 此时生成的格式带:```json 和```
        #parsed_content = parsed_content.replace("```json","").replace("```","")

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



CHUNK_SIZE = 2000
CHUNK_OVERLAP_SIZE = 100

# 每个CHUNK的最少的文本数
CHUNK_MINIMUM_SIZE = 100 

# 提取单个 JSON 对象中的 mmdpath 字段
def extract_mmdpath(text_inc,item):
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
    time_cost = int((time1-time0)*100)/100
    print(f"text_inc:{text_inc:5d} text_id:{text_id:4} text size: {text_size:8} time cost: {time_cost:.2f}   chunk: {len(chunks)}  qa_n: {len(qa_total)}")
    model_name_min = model_name.split("/")[-1]
    item["qas"] = qa_total   
    item["meta"] = {"model":model_name_min,"tcost":time_cost,"chunks":len(chunks)}
    return item







# 主函数
def main():
    # 使用 jsonlines 读取文件中的所有 JSON 对象，并边处理边保存
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode="w") as writer:
        text_inc = 0
        for item in reader:
            # 处理每个 JSON 对象
            processed_item = extract_mmdpath(text_inc,item)
            # 将处理后的 JSON 对象写入输出文件
            writer.write(processed_item)




# 运行主函数
if __name__ == "__main__":
    main()
    
