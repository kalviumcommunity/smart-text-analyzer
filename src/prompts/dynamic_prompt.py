import openai
import os
from dotenv import load_dotenv
import json

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class DynamicPromptHandler:
    def __init__(self):
        self.model = "gpt-3.5-turbo"

    def analyze_text(self, text: str):
        """
        Dynamic prompting: Choose different prompt styles
        depending on the input text length.
        """

        if len(text.split()) > 20:
            # For long texts → detailed summary and sentiment
            system_prompt = (
                "You are an expert text analyst. "
                "Provide detailed analysis in JSON format:\n"
                "{ 'summary': 'string', 'sentiment': 'positive|negative|neutral', 'key_points': ['list'], 'confidence': float }"
            )
            user_prompt = f"Analyze the following long text:\n{text}"
        else:
            # For short texts → quick sentiment only
            system_prompt = (
                "You are an expert sentiment detector. "
                "Respond in JSON format:\n"
                "{ 'sentiment': 'positive|negative|neutral', 'confidence': float }"
            )
            user_prompt = f"Classify the sentiment of this short text:\n{text}"

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
            raise ValueError("Model did not return valid JSON.")
