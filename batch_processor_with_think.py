import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime
from dashscope.aigc.generation import GenerationResponse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Dict, Any, List, Optional, Union
from http import HTTPStatus
import dashscope  # 导入DashScope库

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelAPIEngine")

class BaseModelAPIClient:
    """API客户端基类，处理通用功能"""
    
    def __init__(
        self,
        model_name: str = "default",
        max_retries: int = 3,
        timeout: int = 60,
        max_concurrent: int = 10,
        save_results: bool = False,
        output_dir: str = "results",
        **kwargs
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model_kwargs = kwargs
        self.save_results = save_results
        self.output_dir = output_dir
        
        # 确保输出目录存在
        if save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"结果将保存到目录: {os.path.abspath(self.output_dir)}")
            
        logger.info(f"初始化客户端，模型: {model_name}")
    
    def _save_response(self, prompt: str, response: str) -> None:
        """保存响应到文件"""
        if not self.save_results or not response:
            return
            
        safe_model_name = "".join(c for c in self.model_name if c.isalnum() or c in (' ', '_')).rstrip()
        safe_model_name = safe_model_name.replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_model_name}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("\n=== Prompt ===\n")
                f.write(f"{prompt}\n")
                f.write("\n=== Response ===\n")
                f.write(f"{response}\n")
            logger.info(f"响应已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存响应失败: {str(e)}")
    
    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """生成文本"""
        try:
            result = await self._make_async_request(session, prompt)
            self._save_response(prompt, result)
            return result
        except Exception as e:
            logger.error(f"生成失败: {prompt[:50]}... 错误: {str(e)}")
            return ""
    
    async def _make_async_request(self, session: aiohttp.ClientSession, prompt: str) -> Dict:
        """抽象方法，由子类实现具体请求逻辑"""
        raise NotImplementedError("子类必须实现此方法")

class QwenAPIClient(BaseModelAPIClient):
    """Qwen模型专用API客户端"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen-max",
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.base_url = base_url or "https://dashscope.aliyuncs.com"
    
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_async_request(self, session: aiohttp.ClientSession, prompt: str) -> Dict:
        """使用DashScope库实现Qwen API调用"""
        enable_thinking = self.model_kwargs.get("enable_thinking", False)
        
        # 创建DashScope格式的消息
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        
        # 构建请求参数
        params = {
            "model": self.model_name,
            "api_key": self.api_key,
            "messages": messages,
            "result_format": "message",
            "stream": False
        }
        
        # 添加可选参数
        for param in ["temperature", "top_p", "enable_search"]:
            if param in self.model_kwargs:
                params[param] = self.model_kwargs[param]
                
        if enable_thinking:
            params["enable_thinking"] = True
            params["stream"] = True
            if "thinking_budget" in self.model_kwargs:
                params["thinking_budget"] = self.model_kwargs["thinking_budget"]
        else:
            params["stream"] = False
        
        # 使用DashScope进行API调用
        responses = dashscope.Generation.call(**params)
        reasoning_content = ""
        content = ""
        if params['enable_thinking']:
            for response in responses:
                if response.status_code == HTTPStatus.OK:
                    # print(response)
                    reasoning_content += response.output["choices"][0]["message"]["reasoning_content"]
                    content += response.output["choices"][0]["message"]["content"]
                else:
                    error_msg = f"Qwen API错误: [{response.status_code}] {response.message}"
                    logger.error(error_msg)
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_msg
                    )
        else:
            assert type(responses) == GenerationResponse
            if responses.status_code == HTTPStatus.OK:
                content = responses.output["choices"][0]["message"]["content"]
            else:
                error_msg = f"Qwen API错误: [{responses.status_code}] {responses.message}"
                logger.error(error_msg)
                raise aiohttp.ClientResponseError(
                    responses.request_info,
                    responses.history,
                    status=responses.status,
                    message=error_msg
                )
        return {
            'thinking': '<think>\n' + reasoning_content + '\n</think>\n',
            'content': content
        }

class OllamaAPIClient(BaseModelAPIClient):
    """Ollama模型专用API客户端"""
    
    def __init__(
        self,
        model_name: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.endpoint = "/api/generate"
    
    def _build_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}
    
    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        """构建请求负载"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # 添加可选参数
        options = {}
        for param in ["temperature", "top_p", "num_ctx"]:
            if param in self.model_kwargs:
                options[param] = self.model_kwargs[param]
                
        if options:
            payload["options"] = options
            
        # 处理思维链模式
        if self.model_kwargs.get("enable_thinking", False):
            payload["prompt"] += " /think"
        else:
            payload["prompt"] += " /no_think"
            
        return payload
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_async_request(self, session: aiohttp.ClientSession, prompt: str) -> Dict:
        """执行异步API请求"""
        url = f"{self.base_url}{self.endpoint}"
        payload = self._build_payload(prompt)
        headers = self._build_headers()
        
        async with self.semaphore:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                
                if "error" in response_data:
                    error_msg = response_data["error"]
                    logger.error(f"Ollama API错误: {error_msg}")
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_msg
                    )
                    
                # 处理响应并返回
                return self._parse_response(response_data.get("response", ""))
    
    def _parse_response(self, content: str) -> Dict:
        """解析Ollama响应，处理思维链"""
        if self.model_kwargs.get("enable_thinking", False):
            # 提取思维链部分
            if content.startswith("<think>") and "</think>" in content:
                thinking_end = content.find("</think>")
                return {
                    "thinking": content[7:thinking_end].strip(),
                    "content": content[thinking_end+7:].strip()
                }
        return {
                    "thinking": "",
                    "content": content
                }
class AsyncBatchProcessor:
    """异步批量处理器"""
    
    def __init__(self, api_clients: List[BaseModelAPIClient]):
        self.api_clients = api_clients
        self.session = None
        logger.info(f"初始化批量处理器，{len(api_clients)}个API客户端")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()
    
    async def process_batch(self, prompts: List[str]) -> List[str]:
        if not self.session:
            raise RuntimeError("请使用异步上下文管理器 (async with)")
        
        client_cycle = self._client_cycle()
        tasks = []
        
        for prompt in prompts:
            client = next(client_cycle)
            task = asyncio.create_task(client.generate(self.session, prompt))
            tasks.append(task)
            logger.debug(f"已提交任务: {prompt[:30]}...")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._process_results(results)
    
    def _process_results(self, results):
        """处理结果和异常"""
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"任务失败: {str(res)}")
                final_results.append("")
            else:
                final_results.append(res)
        return final_results
    
    def _client_cycle(self):
        """循环提供API客户端"""
        while True:
            for client in self.api_clients:
                yield client

# 使用示例
async def main():
    # 创建Qwen客户端（启用思维链模式）
    qwen_client = QwenAPIClient(
        api_key="sk-426c613869b34e508a78e0153e3198e6",
        model_name="qwen3-0.6b",
        max_concurrent=3,
        save_results=True,
        output_dir="./qwen_results",
        enable_thinking=True,
        thinking_budget=4000
    )
    
    # # 创建Ollama客户端（启用思维链模式）
    # ollama_client = OllamaAPIClient(
    #     model_name="llama3:8b",
    #     base_url="http://localhost:11434",
    #     max_concurrent=3,
    #     save_results=True,
    #     output_dir="./ollama_results",
    #     enable_thinking=True
    # )
    
    # 初始化批量处理器
    async with AsyncBatchProcessor(
        api_clients=[qwen_client]
    ) as processor:
        prompts = [
            "解释量子计算的基本原理",
            # "计算：如果一辆车在2小时内行驶了120公里，那么它的平均速度是多少公里/小时？",
            # "动物园有15只狮子和20只老虎。如果将3只狮子转移到另一个动物园，还剩下多少只大型猫科动物？"
        ]
        
        results = await processor.process_batch(prompts)
        
        # 输出结果
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"提示 {i+1}: {prompt}")
            
            if isinstance(result, dict) and "thinking" in result:
                print(f"思考过程: {str(result['thinking'])[:50]}")
                print(f"最终回答: {str(result['content'])[:50]}")
            else:
                print(f"回答: {result}")
                
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())