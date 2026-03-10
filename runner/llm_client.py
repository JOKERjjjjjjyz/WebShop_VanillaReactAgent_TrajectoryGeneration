import os
import backoff
from typing import List, Dict, Optional

try:
    import openai
except ImportError:
    openai = None


class OpenAIChatClient:
    """
    A wrapper for OpenAI-compatible chat APIs.
    Works with:
      - OpenAI API:       OpenAIChatClient(model_name="gpt-4o-mini", api_key="sk-...")
      - Local LLM server: OpenAIChatClient(model_name="Qwen3.5-4B",
                                           base_url="http://localhost:8000/v1",
                                           api_key="dummy")
    """

    def __init__(
        self,
        model_name: str = "Qwen3.5-4B",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # For local server, api_key can be anything (e.g. "dummy")
        # For OpenAI, set via env var OPENAI_API_KEY or pass directly
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")

        if openai is None:
            raise ImportError("Please install openai: pip install openai")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,  # None = use OpenAI's default endpoint
        )
        self.base_url = base_url

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def chat(self, history: List[Dict[str, str]]) -> str:
        """
        history format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        Returns the assistant's text response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

