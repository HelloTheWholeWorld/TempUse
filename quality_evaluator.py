import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Union
import requests
import numpy as np
from rouge import Rouge
from bert_score import score as bert_score

class BaseQualityEvaluator(ABC):
    """数据质量评估器基类"""
    
    def __init__(self, 
                 evaluator_type: str, 
                 config: Dict[str, Any] = None):
        """
        :param evaluator_type: 评估器类型标识
        :param config: 评估配置字典
        """
        self.evaluator_type = evaluator_type
        self.config = config or {}
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    @abstractmethod
    def evaluate(self, 
                 input_data: Union[str, Dict], 
                 output_data: str) -> Dict[str, Any]:
        """执行质量评估
        :return: 包含评分和元数据的字典
        """
        pass
    
    def batch_evaluate(self, 
                       data_pairs: List[Dict]) -> List[Dict]:
        """批量评估接口"""
        return [self.evaluate(pair['input'], pair['output']) 
                for pair in data_pairs]

class LLMEvaluator(BaseQualityEvaluator):
    """基于LLM的智能评估器"""
    
    def __init__(self, 
                 api_endpoint: str,
                 api_key: str,
                 eval_prompt_template: str,
                 **kwargs):
        """
        :param api_endpoint: LLM API地址
        :param api_key: API认证密钥
        :param eval_prompt_template: 评估用的Prompt模板
        """
        super().__init__("llm_scoring", kwargs)
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.eval_prompt_template = eval_prompt_template
        
    def _call_llm_api(self, prompt: str) -> str:
        """调用LLM API"""
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": self.config.get("model", "gpt-4-turbo"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.get("temperature", 0.2)
        }
        
        try:
            response = requests.post(self.api_endpoint, 
                                    json=payload, 
                                    headers=headers,
                                    timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            raise RuntimeError(f"LLM API调用失败: {str(e)}")
    
    def evaluate(self, input_data: Union[str, Dict], output_data: str) -> Dict:
        # 构造评估Prompt
        if isinstance(input_data, dict):
            context = json.dumps(input_data, ensure_ascii=False)
        else:
            context = input_data
            
        prompt = self.eval_prompt_template.format(
            context=context,
            output=output_data
        )
        
        # 获取LLM评估结果
        llm_response = self._call_llm_api(prompt)
        
        try:
            # 解析结构化评估结果
            eval_result = json.loads(llm_response)
            return {
                "score": eval_result.get("score", 0),
                "dimensions": eval_result.get("dimensions", {}),
                "evaluation_text": eval_result.get("evaluation_text", ""),
                "evaluator": self.evaluator_type,
                "timestamp": self.timestamp
            }
        except json.JSONDecodeError:
            # 非结构化结果处理
            return {
                "evaluation_text": llm_response,
                "evaluator": self.evaluator_type,
                "timestamp": self.timestamp
            }

class MetricEvaluator(BaseQualityEvaluator):
    """基于传统指标的评估器"""
    
    METRIC_FUNCTIONS = {
        'rouge': lambda ref, hyp: Rouge().get_scores(hyp, ref)[0],
        'bert_score': lambda ref, hyp: bert_score([hyp], [ref], lang='zh')[2].mean(),
        'exact_match': lambda ref, hyp: 1 if hyp.strip() == ref.strip() else 0
    }
    
    def __init__(self, 
                 metric_names: List[str], 
                 reference_field: str = None,
                 **kwargs):
        """
        :param metric_names: 使用的指标名称列表 ['rouge', 'bert_score'...]
        :param reference_field: 输入数据中参考文本的字段名
        """
        super().__init__("metric_scoring", kwargs)
        self.metric_names = metric_names
        self.reference_field = reference_field
        
    def evaluate(self, input_data: Dict, output_data: str) -> Dict:
        # 获取参考文本
        reference = input_data.get(self.reference_field, "") if self.reference_field else ""
        
        results = {}
        for metric in self.metric_names:
            if metric in self.METRIC_FUNCTIONS:
                try:
                    results[metric] = self.METRIC_FUNCTIONS[metric](reference, output_data)
                except Exception as e:
                    results[metric] = -1  # 错误标识
                    print(f"指标[{metric}]计算失败: {str(e)}")
        
        return {
            "scores": results,
            "evaluator": self.evaluator_type,
            "timestamp": self.timestamp,
            "reference": reference
        }

class CustomRuleEvaluator(BaseQualityEvaluator):
    """基于自定义规则的评估器"""
    
    def __init__(self, 
                 rule_function: Callable[[str, str], float],
                 **kwargs):
        """
        :param rule_function: 自定义评估函数 (input, output) -> score
        """
        super().__init__("custom_rule", kwargs)
        self.rule_function = rule_function
        
    def evaluate(self, input_data: Dict, output_data: str) -> Dict:
        try:
            score = self.rule_function(input_data, output_data)
            return {
                "score": float(score),
                "evaluator": self.evaluator_type,
                "timestamp": self.timestamp
            }
        except Exception as e:
            print(f"自定义规则执行失败: {str(e)}")
            return {
                "error": str(e),
                "evaluator": self.evaluator_type,
                "timestamp": self.timestamp
            }

class PluginEvaluator(BaseQualityEvaluator):
    """插件式评估器（预留扩展接口）"""
    # 实现原理与CustomRuleEvaluator类似，支持从外部加载评估插件
    pass

# ================= 使用示例 =================
if __name__ == "__main__":
    # 示例1：LLM评估器
    llm_evaluator = LLMEvaluator(
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_key="your_api_key",
        eval_prompt_template="""
        请评估以下文本质量(1-10分)，返回JSON格式:
        {{
          "score": 分数, 
          "dimensions": {{"准确性":0-5, "流畅性":0-5}},
          "evaluation_text": "评估说明"
        }}
        
        ## 输入数据
        {context}
        
        ## 生成结果
        {output}
        """
    )
    
    # 示例2：指标评估器
    metric_evaluator = MetricEvaluator(
        metric_names=['rouge', 'bert_score'],
        reference_field="reference_text"
    )
    
    # 示例3：自定义规则评估器
    def length_rule(input_data, output):
        """输出长度需在输入长度的80%-120%之间"""
        input_len = len(input_data.get('text', ''))
        output_len = len(output)
        if 0.8 * input_len <= output_len <= 1.2 * input_len:
            return 1
        return 0.3
        
    custom_evaluator = CustomRuleEvaluator(rule_function=length_rule)
    
    # 执行评估
    sample_data = {
        "text": "大模型数据蒸馏技术解析",
        "reference_text": "本文详细讲解了大模型数据蒸馏的实现方法"
    }
    llm_result = llm_evaluator.evaluate(
        input_data=sample_data, 
        output_data="大模型蒸馏技术介绍"
    )
    metric_result = metric_evaluator.evaluate(
        input_data=sample_data, 
        output_data="大模型蒸馏技术介绍"
    )
    
    print("LLM评估结果:", json.dumps(llm_result, indent=2, ensure_ascii=False))
    print("指标评估结果:", json.dumps(metric_result, indent=2, ensure_ascii=False))