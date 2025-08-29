import openai
import os
from dotenv import load_dotenv
import json

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class OneShotPromptHandler:
    def __init__(self):
        self.model = "gpt-3.5-turbo"

    def summarize_text(self, text: str):
        """
        One-shot prompting: Provide one example before asking
        the model to summarize a new input.
        """
        system_prompt = (
            "You are a professional text summarizer. "
            "Always return a valid JSON with:\n"
            "{ 'summary': 'string', 'confidence': float }"
        )

        # 👇 One example included here
        example = (
            "Example Input: 'Artificial intelligence is a rapidly growing field. "
            "It is changing healthcare, finance, and education.'\n"
            "Example Output: { 'summary': 'AI is transforming industries like healthcare, finance, and education.', 'confidence': 0.95 }"
        )

        user_prompt = f"{example}\n\nNow summarize this text:\n{text}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        content = response["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON output from model")
