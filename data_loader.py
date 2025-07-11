import os
import json
import csv
from abc import ABC, abstractmethod
from typing import Dict, Iterator

class BaseDataLoader(ABC):
    """数据加载器基类，支持断点续传功能"""
    
    def __init__(self, file_path: str, state_file: str = "loader_state.json"):
        """
        初始化数据加载器
        :param file_path: 数据文件路径
        :param state_file: 断点状态记录文件路径
        """
        self.file_path = file_path
        self.state_file = state_file
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """加载断点续传状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"状态文件加载失败: {e}")
        return {}
    
    def _save_state(self, current_index: int):
        """保存当前处理进度"""
        self.state[self.file_path] = current_index
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"状态保存失败: {e}")
    
    def get_start_index(self) -> int:
        """获取当前文件的起始处理位置"""
        return self.state.get(self.file_path, 0)
    
    def process_data(self) -> Iterator[Dict]:
        """数据处理主流程，支持断点续传"""
        start_index = self.get_start_index()
        print(f"从索引 {start_index} 开始处理文件: {self.file_path}")
        
        try:
            for i, item in enumerate(self._read_file()):
                if i < start_index:
                    continue
                
                # 统一输出格式
                result = {
                    "raw_data": item,
                    "source": self.file_path,
                    "index": i
                }
                
                # 更新处理进度
                self._save_state(i + 1)
                
                yield result
                
        except Exception as e:
            print(f"文件处理中断: {e}")
            raise
        finally:
            print(f"文件处理完成: {self.file_path}, 最后处理索引: {self.get_start_index()}")

    @abstractmethod
    def _read_file(self) -> Iterator[Dict]:
        """子类需实现的具体文件读取方法"""
        pass


class JsonDataLoader(BaseDataLoader):
    """JSON格式数据加载器"""
    
    def _read_file(self) -> Iterator[Dict]:
        """读取JSON文件（支持jsonlines格式）"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 尝试解析为jsonlines（每行一个JSON对象）
            for i, line in enumerate(f):
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"第 {i} 行JSON解析失败，跳过")
            
            # 如果不是jsonlines格式，尝试解析为完整JSON数组
            f.seek(0)
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
            except json.JSONDecodeError:
                raise ValueError("文件不是有效的JSON格式")


class TsvDataLoader(BaseDataLoader):
    """TSV格式数据加载器"""
    
    def _read_file(self) -> Iterator[Dict]:
        """读取TSV文件"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                yield row