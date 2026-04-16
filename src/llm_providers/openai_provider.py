"""
OpenAI Provider

Uses httpx to call OpenAI API with retry logic
"""

import time
import httpx
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .base import LLMProvider
from ..models import GenerationResult


# Model name mapping
MODEL_MAP = {
    "openai_gpt4o_mini": "gpt-5.4-2026-03-05",
    "openai_gpt4o": "gpt-4o",
    "gpt-5.4-2026-03-05": "gpt-5.4-2026-03-05",
    "gpt-4o": "gpt-4o",
}


class RateLimitError(Exception):
    """Rate limit exception"""
    pass


class OpenAIProvider(LLMProvider):
    """OpenAI Provider with retry logic"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4-2026-03-05",
        base_url: str = "https://api.openai.com/v1",
        timeout_connect: int = 10,
        timeout_read: int = 60,
        max_retries: int = 3,
        sleep_on_rate_limit: float = 2.0
    ):
        """
        Args:
            api_key: OpenAI API Key
            model: Model name (or key in MODEL_MAP)
            base_url: API base URL
            timeout_connect: Connection timeout (seconds)
            timeout_read: Read timeout (seconds)
            max_retries: Maximum retry attempts
            sleep_on_rate_limit: Sleep duration after rate limit (seconds)
        """
        self.api_key = api_key
        # Map model name if needed
        self.model = MODEL_MAP.get(model, model)
        self.base_url = base_url.rstrip('/')
        self.timeout = httpx.Timeout(timeout_connect, read=timeout_read)
        self.max_retries = max_retries
        self.sleep_on_rate_limit = sleep_on_rate_limit
    
    def _make_retry_decorator(self):
        """Create retry decorator with configured parameters"""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=self.sleep_on_rate_limit, max=10),
            retry=retry_if_exception_type((RateLimitError, httpx.HTTPError))
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        decoding: Dict[str, Any] | None = None
    ) -> GenerationResult:
        """Generate response with retry logic"""
        decoding = decoding or {}
        
        @self._make_retry_decorator()
        def _generate_with_retry():
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": decoding.get("temperature", 0.7),
                "top_p": decoding.get("top_p", 1.0),
                "max_tokens": decoding.get("max_tokens", 400)
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                # Check rate limit
                if response.status_code == 429:
                    print(f"  Rate limit hit, retrying after {self.sleep_on_rate_limit}s...")
                    time.sleep(self.sleep_on_rate_limit)
                    raise RateLimitError("Rate limit exceeded")
                
                # Check server errors
                if response.status_code >= 500:
                    print(f"  Server error {response.status_code}, retrying...")
                    raise httpx.HTTPError(f"Server error: {response.status_code}")
                
                response.raise_for_status()
                
                result = response.json()
            
            duration = time.time() - start_time
            
            # Parse result
            choice = result["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
            
            usage = result.get("usage", {})
            response_model = result.get("model", self.model)
            
            return GenerationResult(
                text=text,
                model=response_model,  # Use actual model from response
                provider=self.get_provider_name(),
                duration_seconds=duration,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                finish_reason=finish_reason,
                decoding_params=decoding
            )
        
        return _generate_with_retry()
    
    def get_provider_name(self) -> str:
        return "openai"
