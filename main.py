import json
import logging
import random
import os
from typing import List, Dict, Tuple
import glob
import re
from config import CONFIG
from jinja2 import Template
from batch_processor import ModelAPIClient, AsyncBatchProcessor
from utils import (get_random_question_types, 
                   extract_sections, 
                   extract_new_task_types, 
                   get_processed_files,
                   find_md_files,
                   parse_generated_content,
                   build_result_dict,
                   mark_file_processed,
                   update_task_file
                   )
import asyncio

# 配置日志
# logging_level = logging.DEBUG if CONFIG['DEBUG'] else logging.INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MAIN")

def save_results(
    result_data_list: List[Dict],
    output_dir: str
):
    """按模型和问题类型分类保存结果"""
    for result_data in result_data_list:
        # 创建分类目录
        base_dir = os.path.join(
            output_dir,
            result_data["model_name"].replace(" ", "_"),
            result_data["question_type"].replace(" ", "_")
        )
        os.makedirs(base_dir, exist_ok=True)
        
        # 创建安全文件名
        filename = f"{result_data['question_id']}.json"
        filepath = os.path.join(base_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果保存至: {filepath}")

def prepare_prompts(
    chunks: List[Tuple[str, str]],
    task_data: Dict,
) -> List[str]:
    """准备提示列表"""
    # 加载模板
    with open(CONFIG["PROMPT_TEMPLATE"], "r", encoding="utf-8") as f:
        template_str = f.read()
    template = Template(template_str)
    
    # 准备生成器
    task_gen = get_random_question_types(task_data, CONFIG['TASK_SEED_NUM'])
    prompts = []
    
    for chunk_id, chunk_text in chunks:
        # 获取随机示例
        task_examples = next(task_gen)
        prompts.append(template.render({
            "text": chunk_text,
            "n": CONFIG['N_QUESTIONS_PER_CHUNK'],
            "task_type_examples": task_examples
        }))
    
    return prompts

def process_md_file(file_path: str) -> List[Tuple[str, str]]:
    """处理单个MD文件并生成带有ID的分块，返回文件名和chunks"""
    chunks = extract_sections(
        file_path,
        heading_levels=CONFIG["HEADING_LEVELS"],
        chunk_max_length=4096,
        min_chunk_length=256,
        debug=CONFIG["DEBUG"]
    )
    
    # 应用DEBUG限制
    if CONFIG["DEBUG"] and CONFIG["DEBUG_LIMIT"] > 0:
        chunks = chunks[:CONFIG["DEBUG_LIMIT"]]
    
    # 为每个分块生成ID
    result = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{base_name}_{i}"
        result.append((chunk_id, chunk_text))
    return result

async def main():
    # 确保输出目录存在
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    logger.info(f"输出目录: {os.path.abspath(CONFIG['OUTPUT_DIR'])}")
    
    # 断点续传处理
    processed_files = get_processed_files(CONFIG["OUTPUT_DIR"], CONFIG["PROCESSED_RECORD_FILE"]) if not CONFIG["DEBUG"] else set()
    
    # 创建API客户端
    qwen_client = ModelAPIClient(
        model_type="qwen",
        api_key="sk-426c613869b34e508a78e0153e3198e6",
        model_name="qwen3-4b",
        max_concurrent=CONFIG["MAX_CONCURRENT"],
        save_results=False,
        enable_thinking=False
    )
    
    ollama_client = ModelAPIClient(
        model_type="ollama",
        model_name="qwen3-0.6b",
        max_concurrent=4,
        save_results=False,  # 启用结果保存
        num_ctx=32*1024,
        enable_thinking=False
    )

    # 初始化批量处理器
    async with AsyncBatchProcessor(api_clients=[qwen_client, ollama_client]) as processor:
        # 读取所有MD文件
        md_files = find_md_files(CONFIG["INPUT_FOLDER"])
        logger.info(f'{CONFIG["INPUT_FOLDER"]}下共{len(md_files)}个markdown文件')
        
        # 加载现有任务数据
        task_data = {}
        if os.path.exists(CONFIG["SEED_TASK_FILE"]):
            with open(CONFIG["SEED_TASK_FILE"], "r", encoding="utf-8") as f:
                task_data = json.load(f)
        
        new_task_types = {}
        
        # 处理每个MD文件
        for md_id, file_path in enumerate(md_files):
            if CONFIG['DEBUG']:
                if md_id >= CONFIG['DEBUG_DOCS']:
                    break

            # 断点续传检查
            if file_path in processed_files:
                logger.info(f"跳过已处理文件: {file_path}")
                continue
            
            # 处理文件
            chunks = process_md_file(file_path)

            prompts = prepare_prompts(
                chunks,
                task_data
            )
            
            # 处理批量请求
            results = await processor.process_batch(prompts)
            
            # 收集当前文件的结果
            file_results = []
            
            for (chunk_id, chunk_text), api_response in zip(chunks, results):
                # 解析API响应
                generated_instances = parse_generated_content(api_response)
                
                # 构建结果字典
                chunk_results = build_result_dict(
                    chunk_id,
                    chunk_text,
                    generated_instances,
                    qwen_client.model_name
                )
                
                # 添加到文件结果
                file_results.extend(chunk_results)
                
                # 提取新任务类型
                file_new_types = extract_new_task_types(generated_instances, task_data)
                for category, types in file_new_types.items():
                    if category not in new_task_types:
                        new_task_types[category] = []
                    new_task_types[category].extend(types)
                
                # 更新任务数据 (为下一个分块添加新的类型)
                for category, types in file_new_types.items():
                    if category not in task_data:
                        task_data[category] = []
                    task_data[category].extend(types)
            
            # 保存当前文件的所有结果
            save_results(file_results, CONFIG["OUTPUT_DIR"])
            
            # 标记文件为已处理
            if not CONFIG["DEBUG"]:
                mark_file_processed(file_path, CONFIG["OUTPUT_DIR"], CONFIG['PROCESSED_RECORD_FILE'])
                logger.info(f"文件处理完成: {file_path}")
            
            # 更新任务文件
            if new_task_types:
                update_task_file(new_task_types, CONFIG['SEED_TASK_FILE'], CONFIG["NEW_TASK_FILE"])

if __name__ == "__main__":
    asyncio.run(main())