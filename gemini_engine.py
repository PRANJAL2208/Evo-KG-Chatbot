import os
from dotenv import load_dotenv
import google.generativeai as genai
from kani.models import ChatMessage  
import asyncio

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEngine:
    def __init__(self, model_name="models/gemini-2.5-pro-exp-03-25"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.chat_session = self.model.start_chat(history=[])
        self.max_context_size = 32768  # example limit
        self.token_reserve = 1024      # required by kani


    async def achat(self, messages, **kwargs):
        prompt = "\n".join([msg.content for msg in messages])
        return await self._async_chat(prompt)

    async def _async_chat(self, prompt):
        try:
            response = self.chat_session.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Gemini API error: {e}"

    def completion_to_text(self, completion):
        return completion

    def message_len(self, message):
        return len(message.content.split())  # or estimate via char length

    def function_token_reserve(self, functions: list) -> int:
    # Return a fixed token reserve — 1024 is safe, or adjust as needed
        return 1024

    async def stream(self, messages: list[ChatMessage], functions=None, **kwargs):
        prompt = "\n".join(m.content for m in messages)
        stream = self.model.generate_content(prompt, stream=True)

        # Stream chunks asynchronously
        for chunk in stream:
            await asyncio.sleep(0)  # Yield control to event loop
            yield chunk.text
            
    # In the method where the error occurs:
    def add_completion_to_history(self, completion):
        # Check if `completion.prompt_tokens` is None and set it to 0 if it is
        prompt_tokens = completion.prompt_tokens if completion.prompt_tokens is not None else 0
        
        # Now you can safely add it to `tokens_used_prompt`
        self.tokens_used_prompt += prompt_tokens
