"""
航天领域知识抽取与指令数据集生成系统
核心功能：从航天领域文档中抽取实体-属性-关联上下文，生成多任务指令微调数据集
"""
import json
import re
import os
import jinja2
from typing import List, Dict, Any, Tuple
import requests  # 用于调用Ollama API

# Ollama API配置（根据实际部署修改）
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:instruct"  # 推荐使用指令优化模型

class SpaceKnowledgeExtractor:
    """航天领域知识抽取核心类"""
    
    def __init__(self, template_dir: str = "templates"):
        # 初始化Jinja2模板环境
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape()
        )
        # 预加载关键模板
        self.entity_prompt_template = self.template_env.get_template("entity_extraction.jinja2")
        self.attribute_prompt_template = self.template_env.get_template("attribute_extraction.jinja2")
        
        # 航天领域实体类型定义（可扩展）
        self.space_entity_types = [
            "航天器", "运载火箭", "卫星系统", "探测器", "空间站模块",
            "航天机构", "航天任务", "关键技术", "有效载荷", "地面设施"
        ]
        
        # 航天领域属性类型定义（基于领域知识）
        self.space_attribute_types = [
            "发射时间", "研制单位", "任务目标", "技术参数", "服役状态",
            "轨道参数", "科学载荷", "任务时长", "重量", "能源系统"
        ]

    def call_ollama_api(self, prompt: str, max_tokens=2000) -> str:
        """调用Ollama API获取大模型响应"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3  # 较低温度保证事实准确性
        }
        
        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Ollama API调用失败: {e}")
            return ""

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从航天文本中抽取领域实体及关联上下文"""
        # 渲染实体抽取模板
        prompt = self.entity_prompt_template.render(
            text=text,
            entity_types=self.space_entity_types
        )
        
        # 调用大模型API
        response = self.call_ollama_api(prompt)
        
        # 解析模型响应（JSON格式）
        try:
            entities = json.loads(response)
            # 添加关联上下文
            for entity in entities:
                entity["context"] = self._find_entity_context(text, entity["name"])
            return entities
        except json.JSONDecodeError:
            print("实体解析失败，响应格式错误")
            return []

    def _find_entity_context(self, text: str, entity_name: str, window_size=200) -> str:
        """在文本中定位实体周围的上下文"""
        start_idx = text.find(entity_name)
        if start_idx == -1:
            return ""
        
        # 扩展上下文窗口
        start = max(0, start_idx - window_size)
        end = min(len(text), start_idx + len(entity_name) + window_size)
        return text[start:end].replace('\n', ' ').strip()

    def extract_entity_attributes(self, entity: Dict[str, Any], full_text: str) -> List[Dict[str, str]]:
        """抽取实体在航天领域的详细属性"""
        # 渲染属性抽取模板
        prompt = self.attribute_prompt_template.render(
            entity_name=entity["name"],
            entity_type=entity["type"],
            context=entity["context"],
            attribute_types=self.space_attribute_types
        )
        
        # 调用大模型API
        response = self.call_ollama_api(prompt)
        
        # 解析属性数据
        attributes = []
        try:
            attr_dict = json.loads(response)
            for attr_name, attr_value in attr_dict.items():
                if attr_value:  # 过滤空值
                    attributes.append({
                        "attribute": attr_name,
                        "value": attr_value,
                        "source_context": self._find_attribute_context(full_text, attr_value)
                    })
            return attributes
        except json.JSONDecodeError:
            print("属性解析失败，采用备用解析策略")
            return self._fallback_attribute_parsing(response)

    def _find_attribute_context(self, text: str, value: str) -> str:
        """在文本中定位属性值出现的上下文"""
        if not value:
            return ""
        
        # 查找最接近的属性值出现位置
        match = re.search(re.escape(value), text)
        if match:
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            return text[start:end]
        return ""

    def _fallback_attribute_parsing(self, raw_response: str) -> List[Dict[str, str]]:
        """当JSON解析失败时的备用解析方法"""
        attributes = []
        # 尝试识别key: value格式
        lines = raw_response.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                attr = parts[0].strip()
                value = parts[1].strip()
                if attr and value:
                    attributes.append({"attribute": attr, "value": value})
        return attributes

class InstructionDatasetGenerator:
    """指令微调数据集生成器"""
    
    def __init__(self, extractor: SpaceKnowledgeExtractor):
        self.extractor = extractor
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates"),
            autoescape=jinja2.select_autoescape()
        )
        # 加载任务模板
        self.templates = {
            "single_qa": self.template_env.get_template("single_qa.jinja2"),
            "multi_choice": self.template_env.get_template("multi_choice.jinja2"),
            "multi_turn": self.template_env.get_template("multi_turn.jinja2"),
            "cloze": self.template_env.get_template("cloze.jinja2"),
            "summarization": self.template_env.get_template("summarization.jinja2"),
            "info_extraction": self.template_env.get_template("info_extraction.jinja2")
        }
    
    def process_document(self, doc_path: str) -> List[Dict[str, Any]]:
        """处理单个航天领域文档"""
        # 读取文档内容（实际实现需支持PDF/DOCX等格式）
        text = self._read_document(doc_path)
        if not text:
            return []
        
        # 抽取实体和属性
        entities = self.extractor.extract_entities(text)
        knowledge_graph = []
        
        for entity in entities:
            attributes = self.extractor.extract_entity_attributes(entity, text)
            knowledge_graph.append({
                "entity": entity["name"],
                "type": entity["type"],
                "context": entity["context"],
                "attributes": attributes
            })
        
        # 基于知识图谱生成指令数据
        dataset = []
        for kg in knowledge_graph:
            # 单轮问答（事实性长回复）
            dataset.append(self._generate_single_qa(kg))
            
            # 多项选择题
            if len(kg["attributes"]) >= 3:
                dataset.append(self._generate_multi_choice(kg))
            
            # 完形填空
            dataset.append(self._generate_cloze(kg))
            
            # 信息抽取指令
            dataset.append(self._generate_info_extraction(kg))
            
            # 多轮对话（基于属性链式提问）
            dataset.extend(self._generate_multi_turn(kg))
        
        # 文档级摘要任务
        dataset.append(self._generate_summarization(text, knowledge_graph))
        
        return dataset
    
    def _generate_single_qa(self, kg: Dict) -> Dict:
        """生成单轮问答数据"""
        question = f"请详细介绍{kg['entity']}的基本情况"
        instruction = self.templates["single_qa"].render(
            entity=kg["entity"],
            entity_type=kg["type"]
        )
        # 构建完整答案
        answer = f"{kg['entity']}是{kg['type']}，主要信息包括：\n"
        for attr in kg["attributes"]:
            answer += f"- {attr['attribute']}: {attr['value']}\n"
        
        return {
            "type": "single_qa",
            "instruction": instruction.strip(),
            "input": "",
            "output": answer.strip(),
            "source": kg["context"][:500]  # 截断长上下文
        }
    
    def _generate_multi_choice(self, kg: Dict) -> Dict:
        """生成多项选择题"""
        # 随机选择3个真实属性和1个干扰项
        real_attrs = kg["attributes"][:3]
        fake_attr = {"attribute": "干扰属性", "value": "错误信息"}
        
        options = [{"id": "A", "text": f"{attr['attribute']}: {attr['value']}"} 
                  for attr in real_attrs]
        options.append({"id": "D", "text": f"{fake_attr['attribute']}: {fake_attr['value']}"})
        
        question = f"关于{kg['entity']}的描述，以下哪项是错误的？"
        instruction = self.templates["multi_choice"].render(
            entity=kg["entity"],
            question=question,
            options=options
        )
        
        return {
            "type": "multi_choice",
            "instruction": instruction.strip(),
            "input": "",
            "output": "D",
            "options": options
        }
    
    def _generate_multi_turn(self, kg: Dict) -> List[Dict]:
        """生成多轮对话数据"""
        conversations = []
        # 第一轮：实体介绍
        conversations.append({
            "role": "user",
            "content": f"请介绍一下{kg['entity']}"
        })
        conversations.append({
            "role": "assistant",
            "content": f"{kg['entity']}是{kg['type']}，主要应用于{kg['attributes'][0]['value'] if kg['attributes'] else '航天领域'}"
        })
        
        # 后续轮次：基于属性提问
        for i, attr in enumerate(kg["attributes"][:3]):  # 限制轮次
            conversations.append({
                "role": "user",
                "content": f"它的{attr['attribute']}是什么？"
            })
            conversations.append({
                "role": "assistant",
                "content": attr["value"]
            })
        
        return [{
            "type": "multi_turn",
            "conversations": conversations,
            "source_entity": kg["entity"]
        }]
    
    def _generate_cloze(self, kg: Dict) -> Dict:
        """生成完形填空数据"""
        # 构建含空缺的文本
        context = kg["context"][:300]  # 使用实体上下文
        if not context:
            return None
        
        # 随机选择一个属性值进行遮盖
        if kg["attributes"]:
            target_attr = kg["attributes"][0]
            blank_text = context.replace(target_attr["value"], "______")
            question = f"请在空白处填写正确的{target_attr['attribute']}值"
        else:
            return None
        
        return {
            "type": "cloze",
            "instruction": self.templates["cloze"].render(question=question),
            "input": blank_text,
            "output": target_attr["value"]
        }
    
    def _generate_info_extraction(self, kg: Dict) -> Dict:
        """生成信息抽取指令"""
        instruction = self.templates["info_extraction"].render(
            entity=kg["entity"]
        )
        # 构建结构化输出
        structured_info = {
            "entity": kg["entity"],
            "type": kg["type"],
            "attributes": {attr["attribute"]: attr["value"] for attr in kg["attributes"]}
        }
        
        return {
            "type": "info_extraction",
            "instruction": instruction.strip(),
            "input": kg["context"][:500],
            "output": json.dumps(structured_info, ensure_ascii=False)
        }
    
    def _generate_summarization(self, text: str, kg: List[Dict]) -> Dict:
        """生成知识摘要任务"""
        # 构建知识图谱摘要
        summary = f"文档核心知识摘要：\n"
        for entity in kg[:3]:  # 摘要前3个关键实体
            summary += f"- {entity['entity']}：{entity['type']}，"
            if entity["attributes"]:
                summary += f"关键属性：{entity['attributes'][0]['attribute']}={entity['attributes'][0]['value']}\n"
        
        return {
            "type": "summarization",
            "instruction": self.templates["summarization"].render(),
            "input": text[:1000],  # 使用文档前1000字符作为输入
            "output": summary
        }
    
    def _read_document(self, path: str) -> str:
        """读取不同格式的航天文档（简化版，实际需扩展）"""
        if path.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        # 实际应添加PDF/DOCX解析逻辑
        else:
            print(f"未支持的文档格式：{path}")
            return ""

# === 模板文件示例 ===
# templates/entity_extraction.jinja2 内容：
"""
您是一个航天领域知识抽取专家，请从以下文本中识别航天领域的特定实体：
{% for type in entity_types %}- {{ type }}
{% endfor %}
文本内容：
{{ text }}

请严格按JSON格式输出：[{
  "name": "实体名称",
  "type": "实体类型"
}]
"""

# templates/attribute_extraction.jinja2 内容：
"""
您正在分析航天器『{{ entity_name }}』（类型：{{ entity_type }}），请从上下文提取以下属性：
{% for attr in attribute_types %}- {{ attr }}
{% endfor %}
上下文：
{{ context }}

请输出JSON字典：{"属性名": "属性值"}
"""

if __name__ == "__main__":
    # 初始化组件
    extractor = SpaceKnowledgeExtractor()
    generator = InstructionDatasetGenerator(extractor)
    
    # 处理航天文档（示例路径）
    dataset = generator.process_document("天问一号技术文档.txt")
    
    # 保存为llama-factory兼容格式
    with open("space_instruction_data.jsonl", "w", encoding="utf-8") as f:
        for data in dataset:
            # 统一格式化处理
            if data["type"] == "multi_turn":
                f.write(json.dumps({
                    "type": data["type"],
                    "conversations": data["conversations"]
                }, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps({
                    "instruction": data["instruction"],
                    "input": data.get("input", ""),
                    "output": data["output"]
                }, ensure_ascii=False) + '\n')
    
    print(f"成功生成{len(dataset)}条航天领域指令数据")

{
    "entity": "玉兔号",
    "entity_type": "",
    "entity_context": "2023年发射的遥五火箭...",
    "entity_attributes": [
        {"entity": "玉兔号", "attribute": "发射时间", "value": "2013年12月", "attribute_type":""},
        {"entity": "玉兔号", "attribute": "研制单位", "value": "航天五院", "attribute_type":""}
    ]
}