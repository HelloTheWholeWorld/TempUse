import json
import os
import random
from tqdm import tqdm
from datetime import datetime

# 假设以下模块已实现（根据您之前的描述）
from data_loader import JsonDataLoader
from prompt_builder import PromptBuilder
from batch_processor import BatchProcessor  # 假设已实现
from result_processor import JsonResultProcessor  # 假设已实现

def process_json_data(input_file):
    """处理JSON数据：切分、采样并保存"""
    print(f"开始处理源文件: {input_file}")
    
    # 1. 读取原始JSON数据
    with open(input_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    if not isinstance(full_data, list):
        raise ValueError("JSON文件应包含列表格式数据")
    
    total_count = len(full_data)
    print(f"数据总量: {total_count:,} 条")
    
    # 2. 随机切分成两半
    random.shuffle(full_data)
    split_index = total_count // 2
    half1 = full_data[:split_index]
    half2 = full_data[split_index:]
    
    # 3. 从每半中采样20%
    sample_size = int(total_count * 0.2)
    sample_half1 = random.sample(half1, min(sample_size, len(half1)))
    sample_half2 = random.sample(half2, min(sample_size, len(half2)))
    
    # 4. 保存所有切分文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    file_paths = {
        "half1": f"{base_name}_half1_{timestamp}.json",
        "half2": f"{base_name}_half2_{timestamp}.json",
        "sample1": f"{base_name}_sample1_{timestamp}.json",
        "sample2": f"{base_name}_sample2_{timestamp}.json"
    }
    
    for name, data in [("half1", half1), ("half2", half2), 
                      ("sample1", sample_half1), ("sample2", sample_half2)]:
        with open(file_paths[name], 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"文件已保存: {list(file_paths.values())}")
    return file_paths

def translate_instructions(sample_file, output_file):
    """翻译JSON文件中的instruction字段"""
    print(f"开始翻译文件: {sample_file}")
    
    # 1. 加载数据
    loader = JsonDataLoader(sample_file)
    data_entries = list(loader.process_data())
    
    # 2. 准备翻译
    translator = PromptBuilder("请将以下英文问题翻译成中文：{{ instruction }}")
    batch_processor = BatchProcessor()  # 假设已实现API调用
    result_processor = JsonResultProcessor(
        output_dir="translated_results",
        parser_func=lambda x: {"translation": x.strip()}
    )
    
    # 3. 批量翻译
    batch_size = 50
    translated_data = []
    
    for i in tqdm(range(0, len(data_entries), batch_size), desc="翻译进度"):
        batch = data_entries[i:i+batch_size]
        
        # 构建prompt并处理
        prompts = [translator.build_single({
            "instruction": entry['raw_data']['instruction']
        }) for entry in batch]
        
        translations = batch_processor.process(prompts)
        
        # 重组数据结构
        for j, entry in enumerate(batch):
            original = entry['raw_data']
            translated_entry = {
                "original_instruction": original["instruction"],
                "translated_instruction": translations[j],
                "output": original["output"],
                "source_file": sample_file,
                "batch_index": i + j
            }
            translated_data.append(translated_entry)
    
    # 4. 保存翻译结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print(f"翻译完成! 结果保存至: {output_file}")
    return output_file

def main():
    # 配置参数
    INPUT_FILE = "original_data.json"  # 包含10w条数据的JSON文件
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 第一阶段：数据切分与采样
        processed_files = process_json_data(INPUT_FILE)
        
        # 第二阶段：翻译采样文件
        translate_instructions(
            processed_files["sample1"],
            f"translated_sample1_{TIMESTAMP}.json"
        )
        
        translate_instructions(
            processed_files["sample2"],
            f"translated_sample2_{TIMESTAMP}.json"
        )
        
        print("所有处理步骤已完成!")
    
    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
