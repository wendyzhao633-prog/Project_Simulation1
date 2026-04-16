"""
LLM Provider 抽象基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..models import GenerationResult


class LLMProvider(ABC):
    """LLM Provider 抽象基类"""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        decoding: Dict[str, Any] | None = None
    ) -> GenerationResult:
        """
        生成回复
        
        Args:
            messages: OpenAI 样式消息列表 [{"role": "system/user/assistant", "content": "..."}]
            decoding: 解码参数，包含 temperature, top_p, max_tokens 等
        
        Returns:
            GenerationResult
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """返回 provider 名称"""
        pass

