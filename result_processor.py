import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional

class BaseResultProcessor(ABC):
    """结果处理器基类"""
    
    def __init__(self, 
                 output_dir: str, 
                 parser_func: Optional[Callable[[str], Dict]] = None,
                 storage_hierarchy: Optional[list] = None):
        """
        :param output_dir: 结果输出根目录
        :param parser_func: LLM输出解析函数
        :param storage_hierarchy: 存储分层规则 (如["date", "source"])
        """
        self.output_dir = output_dir
        self.parser_func = parser_func or self.default_parser
        self.storage_hierarchy = storage_hierarchy or []
        os.makedirs(output_dir, exist_ok=True)
    
    def default_parser(self, llm_output: str) -> Dict:
        """默认解析器：直接返回原始文本"""
        return {"raw_output": llm_output}
    
    def process(self, 
               input_data: Dict, 
               llm_output: str,
               checkpoint_info: Optional[Dict] = None) -> Dict:
        """
        处理结果的主流程
        :param input_data: 原始输入数据
        :param llm_output: LLM原始输出
        :param checkpoint_info: 断点续传所需信息
        :return: 封装后的结果数据
        """
        # 1. 解析LLM输出
        parsed_data = self.parse_output(llm_output)
        
        # 2. 封装元数据
        wrapped_data = self.wrap_data(input_data, parsed_data, checkpoint_info)
        
        # 3. 分层保存结果
        self.save_result(wrapped_data)
        
        return wrapped_data
    
    def parse_output(self, llm_output: str) -> Dict:
        """解析LLM结构化输出"""
        try:
            return self.parser_func(llm_output)
        except Exception as e:
            print(f"解析失败: {str(e)}，使用原始输出")
            return {"raw_output": llm_output}
    
    def wrap_data(self, 
                 input_data: Dict, 
                 parsed_data: Dict,
                 checkpoint_info: Optional[Dict] = None) -> Dict:
        """封装数据信息"""
        # 基础元数据
        wrapped = {
            "timestamp": datetime.now().isoformat(),
            "source": input_data.get("source", "unknown"),
            "input": input_data.get("raw_data", input_data),
            "output": parsed_data,
            "checkpoint": checkpoint_info or {}
        }
        
        # 添加子类特有的扩展字段
        return self.add_custom_fields(wrapped, input_data, parsed_data)
    
    @abstractmethod
    def add_custom_fields(self, 
                         wrapped_data: Dict, 
                         input_data: Dict,
                         parsed_data: Dict) -> Dict:
        """子类实现：添加自定义字段"""
        pass
    
    def save_result(self, wrapped_data: Dict):
        """分层保存结果"""
        # 生成分层目录路径
        save_path = self.output_dir
        for field in self.storage_hierarchy:
            field_value = str(wrapped_data.get(field, "unknown"))
            save_path = os.path.join(save_path, field_value)
        
        os.makedirs(save_path, exist_ok=True)
        
        # 生成文件名（时间戳+来源）
        filename = f"{wrapped_data['timestamp']}_{wrapped_data['source']}".replace(":", "-")
        full_path = os.path.join(save_path, f"{filename}.{self.get_file_extension()}")
        
        # 子类实现具体保存逻辑
        self.save_to_file(full_path, wrapped_data)
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """子类实现：返回文件扩展名"""
        pass
    
    @abstractmethod
    def save_to_file(self, file_path: str, data: Dict):
        """子类实现：具体保存方法"""
        pass


class JsonResultProcessor(BaseResultProcessor):
    """JSON格式结果处理器"""
    
    def add_custom_fields(self, wrapped_data, input_data, parsed_data):
        """添加JSON特有字段"""
        wrapped_data["format"] = "json"
        wrapped_data["schema_version"] = "1.0"
        return wrapped_data
    
    def get_file_extension(self) -> str:
        return "json"
    
    def save_to_file(self, file_path: str, data: Dict):
        """保存为JSON文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class TextResultProcessor(BaseResultProcessor):
    """文本格式结果处理器"""
    
    def add_custom_fields(self, wrapped_data, input_data, parsed_data):
        """添加文本特有字段"""
        wrapped_data["format"] = "text"
        wrapped_data["content_type"] = "plain/text"
        return wrapped_data
    
    def get_file_extension(self) -> str:
        return "txt"
    
    def save_to_file(self, file_path: str, data: Dict):
        """保存为文本文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {data['timestamp']}\n")
            f.write(f"Source: {data['source']}\n")
            f.write("Input:\n")
            f.write(json.dumps(data['input'], ensure_ascii=False) + "\n\n")
            f.write("Output:\n")
            f.write(json.dumps(data['output'], ensure_ascii=False))


class PydanticResultProcessor(BaseResultProcessor):
    """Pydantic结构化结果处理器[9,10](@ref)"""
    
    def __init__(self, 
                 output_dir: str, 
                 pydantic_model: Any,
                 storage_hierarchy: Optional[list] = None):
        """
        :param pydantic_model: Pydantic模型类
        """
        super().__init__(output_dir, self.pydantic_parser, storage_hierarchy)
        self.pydantic_model = pydantic_model
    
    def pydantic_parser(self, llm_output: str) -> Dict:
        """使用Pydantic解析LLM输出"""
        try:
            # 尝试直接解析为Pydantic模型
            if isinstance(llm_output, str) and llm_output.startswith("{"):
                json_data = json.loads(llm_output)
                return self.pydantic_model(**json_data).dict()
            return self.pydantic_model.parse_raw(llm_output).dict()
        except Exception as e:
            print(f"Pydantic解析失败: {str(e)}")
            return {"raw_output": llm_output}
    
    def add_custom_fields(self, wrapped_data, input_data, parsed_data):
        """添加Pydantic特有字段"""
        wrapped_data["model_name"] = self.pydantic_model.__name__
        wrapped_data["schema"] = self.pydantic_model.schema()
        return wrapped_data
    
    def get_file_extension(self) -> str:
        return "json"
    
    def save_to_file(self, file_path: str, data: Dict):
        """保存为JSON文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# 示例领域特定子类
class MedicalResultProcessor(JsonResultProcessor):
    """医疗领域结果处理器"""
    
    def add_custom_fields(self, wrapped_data, input_data, parsed_data):
        wrapped_data = super().add_custom_fields(wrapped_data, input_data, parsed_data)
        # 添加医疗领域特有字段
        wrapped_data["patient_id"] = input_data.get("patient_id", "unknown")
        wrapped_data["diagnosis_code"] = parsed_data.get("diagnosis", {}).get("code", "")
        wrapped_data["confidence_score"] = parsed_data.get("confidence", 0.0)
        return wrapped_data
    
    def save_result(self, wrapped_data: Dict):
        """医疗领域特殊存储规则：按患者ID分层"""
        original_hierarchy = self.storage_hierarchy
        self.storage_hierarchy = ["patient_id"]  # 覆盖为按患者ID分层
        super().save_result(wrapped_data)
        self.storage_hierarchy = original_hierarchy  # 恢复原始设置


if __name__ == '__main__':
    # Pydantic解析器使用示例
    from pydantic import BaseModel

    class DiagnosisResult(BaseModel):
        disease: str
        confidence: float
        treatment_plan: list

    medical_parser = PydanticResultProcessor(
        output_dir="data/medical",
        pydantic_model=DiagnosisResult,
        storage_hierarchy=["date", "disease"]
    )

    # 创建质量评估器
    quality_evaluator = LLMEvaluator(...)

    # 处理流程
    def full_pipeline(data):
        # 数据加载 -> 提示构造 -> LLM调用 -> 质量评估 -> 结果处理
        prompt = prompt_builder.build_single(data)
        llm_output = batch_processor.process([prompt])[0]
        quality_score = quality_evaluator.evaluate(data, llm_output)
        
        result_processor.process(
            input_data=data,
            llm_output=llm_output,
            checkpoint_info={"index": data["index"]},
            extra_fields={"quality_score": quality_score}
        )