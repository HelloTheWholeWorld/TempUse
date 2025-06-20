# ... 前面代码保持不变 ...

class AsyncBatchProcessor:
    """异步批量处理器"""
    
    def __init__(self, api_clients: List[ModelAPIClient]):
        # ... 保持不变 ...
    
    async def process_batch(self, prompts: List[str]) -> List[Tuple[str, str]]:
        """
        处理一批提示文本，返回结果和对应的模型名称
        
        :param prompts: 提示文本列表
        :return: 列表，每个元素是元组 (生成内容, 模型名称)
        """
        if not self.session:
            raise RuntimeError("Use async context manager (async with)")
        
        # 轮询使用不同的API客户端
        client_cycle = self._client_cycle()
        
        tasks = []
        for prompt in prompts:
            client = next(client_cycle)
            # 创建任务时保存当前客户端信息
            task = asyncio.create_task(self._generate_and_record(session, client, prompt))
            tasks.append(task)
            logger.debug(f"Submitted task for prompt: {prompt[:30]}...")
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Task failed: {str(res)}")
                # 异常时返回空结果和模型名称占位符
                final_results.append(("", ""))
            else:
                final_results.append(res)
        
        return final_results
    
    async def _generate_and_record(self, session: aiohttp.ClientSession, client: ModelAPIClient, prompt: str) -> Tuple[str, str]:
        """调用生成方法并记录模型名称"""
        try:
            result = await client.generate(session, prompt)
            return (result, client.model_name)
        except Exception as e:
            logger.error(f"生成失败: {prompt[:50]}... 错误: {str(e)}")
            return ("", client.model_name)
    
    # ... 其他方法保持不变 ...

# ... 后面代码保持不变 ...






# ... 在main函数中修改结果处理部分 ...

async def main():
    # ... 前面代码保持不变 ...
    
    async with AsyncBatchProcessor(api_clients=[qwen_client, ollama_client]) as processor:
        # ... 前面代码保持不变 ...
        
        # 处理批量请求 - 现在返回元组列表（结果，模型名称）
        batch_results = await processor.process_batch(prompts)
        
        # 收集当前文件的结果
        file_results = []
        
        for (chunk_id, chunk_text), (api_response, model_name) in zip(chunks, batch_results):
            # 解析API响应
            generated_instances = parse_generated_content(api_response)
            
            # 构建结果字典时使用任务对应的模型名称
            chunk_results = build_result_dict(
                chunk_id,
                chunk_text,
                generated_instances,
                model_name  # 这里使用实际的模型名称
            )
            
            # 添加到文件结果
            file_results.extend(chunk_results)
            
            # ... 其他处理保持不变 ...
    
    # ... 后面代码保持不变 ...
