from openai import OpenAI

from .config import Settings


class LLMEvalToolKit:
    def __init__(self, api_key: str = Settings.API_KEY, base_url: str = Settings.BASE_URL):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, query: str, model_name: str = Settings.MODEL_NAME):
        messages = [{"role": "user", "content": query}]
        completions = self.client.chat.completions.create(messages=messages, model=model_name)
        return completions.choices[0].message.content
