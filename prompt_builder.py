from jinja2 import Environment, Template, Undefined, meta
from typing import Dict, List, Union
import os
import warnings

class SilentUndefined(Undefined):
    """自定义未定义变量处理策略，静默返回空字符串而不报错[8](@ref)"""
    def __str__(self):
        return ""
    
    def __iter__(self):
        return iter([])

class PromptBuilder:
    """基于Jinja2的Prompt批量构造器"""
    
    def __init__(self, template_source: Union[str, os.PathLike]):
        """
        初始化模板引擎
        :param template_source: 模板字符串或模板文件路径
        """
        self.env = Environment(undefined=SilentUndefined)  # 静默处理未定义变量[8](@ref)
        
        if os.path.exists(template_source):
            with open(template_source, 'r', encoding='utf-8') as f:
                self.template_str = f.read()
        else:
            self.template_str = template_source
            
        self.template = self.env.from_string(self.template_str)
        self.required_vars = self._detect_required_variables()

    def _detect_required_variables(self) -> List[str]:
        """解析模板依赖的变量[6](@ref)"""
        parsed_content = self.env.parse(self.template_str)
        return sorted(meta.find_undeclared_variables(parsed_content))

    def validate_data(self, data: Dict) -> bool:
        """验证数据是否包含模板所需变量"""
        missing_vars = [var for var in self.required_vars if var not in data]
        if missing_vars:
            warnings.warn(f"缺少必要变量: {', '.join(missing_vars)}")
            return False
        return True

    def build_single(self, data: Dict) -> str:
        """单条数据生成Prompt"""
        if not self.validate_data(data):
            return ""
        return self.template.render(**data)

    def build_batch(self, data_list: List[Dict]) -> List[str]:
        """批量生成Prompt列表[7](@ref)"""
        results = []
        for data in data_list:
            if self.validate_data(data):
                results.append(self.template.render(**data))
        return results

    def generate_dynamic_template(self, base_prompt: str, context: Dict) -> str:
        """动态生成模板（支持嵌套模板）[3](@ref)"""
        dynamic_template = Template(base_prompt, undefined=SilentUndefined)
        return dynamic_template.render(**context)
    

if __name__ == "__main__":
    # 单条数据处理
    builder = PromptBuilder("请分析新闻: {{ title }}\n附加背景: {{ context }}")
    data = {"title": "经济新政", "context": "2025年税收改革"}
    print(builder.build_single(data))

    # 批量处理（高效渲染）
    batch_data = [{"title": "科技突破", "context": "AI新模型"}, ...]
    prompts = builder.build_batch(batch_data)



    # 创建基础模板
    base_template = "{{ header }}\n问题: {{ question }}"
    # 动态生成具体场景模板
    dynamic_prompt = builder.generate_dynamic_template(
        base_template,
        {"header": "医疗报告分析", "question": "诊断依据是什么？"}
    )
    print(dynamic_prompt)  # 输出: 医疗报告分析\n问题: 诊断依据是什么？


    # 创建基础模板
    base_template = "{{ header }}\n问题: {{ question }}"

    # 动态生成具体场景模板
    dynamic_prompt = builder.generate_dynamic_template(
        base_template,
        {"header": "医疗报告分析", "question": "诊断依据是什么？"}
    )
    print(dynamic_prompt)  # 输出: 医疗报告分析\n问题: 诊断依据是什么？





    # 自动检测缺失变量
    data = {"title": "不完整数据"}
    builder.build_single(data)  # 触发警告: 缺少必要变量: context



    

    # 定义核查模板
    template = """
    请核查新闻真实性[1](@ref):
    **标题**: {{ title }}
    **来源**: {{ source }}
    **关键声明**: {{ claim }}

    附加要求:
    {% if historical_context %}历史背景: {{ historical_context }}{% endif %}
    {% if reliable_sources %}参考可信源: {{ reliable_sources|join(', ') }}{% endif %}
    """

    # 批量生成核查指令
    builder = PromptBuilder(template)
    news_batch = [
        {
            "title": "火星移民计划启动",
            "source": "宇宙时报",
            "claim": "NASA将在2026年送人类登陆火星",
            "reliable_sources": ["NASA官网", "科学杂志"]
        },
        # ...更多新闻数据
    ]
    prompts = builder.build_batch(news_batch)