from langchain.llms.base import LLM
from typing import Any, List, Dict, Optional, Mapping
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

class NvidiaLLM(LLM):
    """LangChain Custom LLM for NVIDIA API."""
    
    model: str = "meta/llama3-70b-instruct"
    temperature: float = 0.5
    top_p: float = 0.9
    max_tokens: int = 2048
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: Optional[str] = None
    client: Optional[OpenAI] = None

    def __init__(self, **data: Any):
        """Initialize the LLM."""
        super().__init__(**data)
        
        if not self.api_key:
            self.api_key = os.getenv("NVIDIA_API_KEY")
            if not self.api_key:
                raise ValueError("NVIDIA API key not found")
                
        if not self.client:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

    @property
    def _llm_type(self) -> str:
        return "nvidia"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url
        }

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Makes the API call to NVIDIA."""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=stop,
                **kwargs
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in NVIDIA LLM call: {str(e)}")
            raise

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Synchronous version of _acall."""
        import asyncio
        return asyncio.run(self._acall(prompt, stop, **kwargs)) 