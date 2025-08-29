import openai
import os
from dotenv import load_dotenv
import json

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ZeroShotPromptHandler:
    def __init__(self):
        self.model = "gpt-3.5-turbo"

    def classify_sentiment(self, text: str):
        """
        Zero-shot prompting: Directly ask the model to classify sentiment
        without giving any examples.
        """
        system_prompt = (
            "You are an expert sentiment classifier. "
            "Always respond in JSON format with:\n"
            "{ 'sentiment': 'positive | negative | neutral', 'confidence': float }"
        )

        user_prompt = f"Classify the sentiment of the following text:\n{text}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        content = response["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON output from model")
