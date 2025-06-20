import asyncio
import aiohttp
import json
import logging
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Dict, Any, List, Optional
from config import CONFIG

# 配置日志
logging_level = logging.DEBUG if CONFIG['DEBUG'] else logging.INFO
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelAPIEngine")

class ModelAPIClient:
    """统一多模型API客户端，支持云端API和本地Ollama"""
    
    def __init__(
        self,
        model_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4-turbo",
        max_retries: int = 3,
        timeout: int = 60,
        max_concurrent: int = 10,
        save_results: bool = False,
        output_dir: str = "results",
        **kwargs
    ):
        """
        初始化API客户端
        :param model_type: 模型类型 (openai/deepseek/volcano/ollama/qwen)
        :param api_key: API密钥（云端API需要）
        :param base_url: 自定义基础URL（可选）
        :param model_name: 模型名称
        :param max_retries: 最大重试次数
        :param timeout: 请求超时时间（秒）
        :param max_concurrent: 最大并发请求数
        :param save_results: 是否保存结果文件
        :param output_dir: 结果保存目录
        :param kwargs: 模型特定参数，如temperature, top_p等
        """
        self.model_type = model_type.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model_kwargs = kwargs  # 存储模型特定参数
        self.save_results = save_results
        self.output_dir = output_dir
        
        # 确保输出目录存在
        if save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"结果将保存到目录: {os.path.abspath(self.output_dir)}")
        
        # 设置API基础URL
        self.base_url = self._resolve_base_url(base_url)
        
        # 设置特定API端点
        self.endpoints = {
            "openai": "/v1/chat/completions",
            "deepseek": "/v1/chat/completions",
            "volcano": "/api/v1/chat",
            "ollama": "/api/generate",
            "qwen": "/api/v1/services/aigc/text-generation/generation"
        }

        # 模型特定参数模板
        self.payload_templates = {
            "openai": {
                "model": model_name,
                "messages": [{"role": "user", "content": ""}],
                "temperature": 0.7,
                "top_p": 1.0,
                "max_tokens": None,
                "stream": False
            },
            "deepseek": {
                "model": model_name,
                "messages": [{"role": "user", "content": ""}],
                "temperature": 0.7,
                "top_p": 1.0,
                "max_tokens": None,
                "stream": False
            },
            "volcano": {
                "model": model_name,
                "messages": [{"role": "user", "content": ""}],
                "temperature": 0.7,
                "top_p": 1.0,
                "max_tokens": None,
                "stream": False
            },
            "qwen": {
                "model": model_name,
                "input": {
                    "messages": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": ""}
                    ]
                },
                "parameters": {
                    "result_format": "message",
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "enable_search": False,
                    "stream": False,
                    "enable_thinking": False
                }
            },
            "ollama": {
                "model": model_name,
                "prompt": "",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "num_ctx": 2048
                }
            }
        }
        
        logger.info(f"Initialized {model_type.upper()} client for model: {model_name}")

    def _resolve_base_url(self, base_url: Optional[str]) -> str:
        """解析基础URL"""
        if base_url:
            return base_url.rstrip('/')
        
        # 默认URL配置
        defaults = {
            "openai": "https://api.openai.com",
            "deepseek": "https://api.deepseek.com",
            "volcano": "https://open.volcengineapi.com",
            "ollama": "http://localhost:11434",
            "qwen": "https://dashscope.aliyuncs.com"
        }
        return defaults.get(self.model_type, "")
    
    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        """构建模型特定的请求负载"""
        if self.model_type not in self.payload_templates:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        payload = self.payload_templates[self.model_type].copy()

        # 更新payload中的模型特定参数
        if self.model_type == "ollama":
            payload["prompt"] = prompt
            # 更新Ollama的options参数
            for key in ["temperature", "top_p", "num_ctx",]:
                if key in self.model_kwargs:
                    payload["options"][key] = self.model_kwargs[key]
            if "enable_thinking" in self.model_kwargs and not self.model_kwargs['enable_thinking']:
                payload["prompt"] = '/no_think' + prompt
        elif self.model_type == "qwen":
            # 更新Qwen的消息内容
            payload["input"]["messages"][1]["content"] = prompt

            # 更新参数
            for key in ["temperature", "top_p", "enable_search", "stream", "enable_thinking"]:
                if key in self.model_kwargs:
                    payload["parameters"][key] = self.model_kwargs[key]
        else:
            payload["messages"][0]["content"] = prompt
            # 更新通用参数
            for key in ["temperature", "top_p", "max_tokens", "stream"]:
                if key in self.model_kwargs:
                    payload[key] = self.model_kwargs[key]
        return payload
    
    def _build_headers(self) -> Dict[str, str]:
        """构建请求头，包含认证信息"""
        headers = {"Content-Type": "application/json"}
        
        if self.model_type == "openai" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.model_type == "deepseek" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.model_type == "volcano" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.model_type == "qwen" and self.api_key:
            # 添加官方要求的额外头部
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-DashScope-Async"] = "disable"  # 确保同步调用

        # Ollama不需要认证头

        return headers
    
    def _extract_response(self, response_data: Dict) -> str:
        """从不同模型的响应中提取文本内容"""
        if self.model_type == "ollama":
            return response_data.get("response", "")
        elif self.model_type in ["openai", "deepseek"]:
            return response_data["choices"][0]["message"]["content"]
        elif self.model_type == "volcano":
            return response_data["result"]["choices"][0]["message"]["content"]
        elif self.model_type == "qwen":
            # 根据官方格式提取响应内容
            return response_data["output"]["choices"][0]["message"]["content"]
        else:
            return json.dumps(response_data)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry #{retry_state.attempt_number} for request due to"
        )
    )
    async def _make_async_request(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """执行异步API请求（带重试机制）"""
        url = f"{self.base_url}{self.endpoints[self.model_type]}"
        payload = self._build_payload(prompt)
        headers = self._build_headers()
        
        async with self.semaphore:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    logger.debug(response_data)

                    # 处理API返回的错误
                    if "error" in response_data:
                        error_msg = response_data["error"].get("message", "Unknown API error")
                        logger.error(f"API error: {error_msg}")
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=error_msg
                        )
                    return self._extract_response(response_data)
            
            except aiohttp.ClientResponseError as e:
                logger.debug(f"Request URL: {url}")
                logger.debug(f"Request Headers: {headers}")
                logger.debug(f"Request Payload: {payload}")
                logger.error(f"HTTP error {e.status}: {e.message}")
                error_detail = await e.text()
                logger.error(f"ClientResponseError API error {e.status}: {error_detail}")
                raise
            except asyncio.TimeoutError:
                logger.debug(f"Request URL: {url}")
                logger.debug(f"Request Headers: {headers}")
                logger.debug(f"Request Payload: {payload}")
                logger.warning("Request timed out")
                raise
            except Exception as e:
                logger.debug(f"Request URL: {url}")
                logger.debug(f"Request Headers: {headers}")
                logger.debug(f"Request Payload: {payload}")
                logger.error(f"Unexpected error: {str(e)}")
                raise

    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """
        生成文本内容
        
        :param session: aiohttp会话对象
        :param prompt: 输入的提示文本
        :return: 模型生成的文本
        """
        try:
            result = await self._make_async_request(session, prompt)
            return result
        except Exception as e:
            logger.error(f"所有重试失败: {prompt[:50]}... 错误: {str(e)}")
            return ""

class AsyncBatchProcessor:
    """异步批量处理器"""
    
    def __init__(self, api_clients: List[ModelAPIClient]):
        """
        初始化批量处理器
        
        :param api_clients: 配置好的API客户端列表
        """
        self.api_clients = api_clients
        self.session = None
        logger.info(f"Initialized batch processor with {len(api_clients)} API clients")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()
    
    async def process_batch(self, prompts: List[str]) -> List[str]:
        """
        处理一批提示文本
        
        :param prompts: 提示文本列表
        :return: 生成结果列表
        """
        if not self.session:
            raise RuntimeError("Use async context manager (async with)")
        
        # 轮询使用不同的API客户端
        client_cycle = self._client_cycle()
        
        tasks = []
        for prompt in prompts:
            client = next(client_cycle)
            task = asyncio.create_task(client.generate(self.session, prompt))
            tasks.append(task)
            logger.debug(f"Submitted task for prompt: {prompt[:30]}...")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Task failed: {str(res)}")
                final_results.append("")
            else:
                final_results.append(res)
        
        return final_results
    
    def _client_cycle(self):
        """无限循环提供API客户端"""
        while True:
            for client in self.api_clients:
                yield client

