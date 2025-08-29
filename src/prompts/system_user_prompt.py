import openai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptHandler:
    def __init__(self):
        self.model = "gpt-3.5-turbo"

    def analyze_text(self, user_input: str):
        """
        Creates a system + user prompt for text analysis
        using the RTFC framework.
        """
        system_prompt = (
            "You are an intelligent text analysis system. "
            "Follow the RTFC framework:\n"
            "Role: Text Analysis Expert\n"
            "Tasks: Sentiment, summary, categorization, semantic similarity, quality assessment\n"
            "Format: JSON output with confidence scores and metadata\n"
            "Context: Adapt to domain, user preferences, and maintain consistency"
        )

        user_prompt = f"Analyze the following text:\n{user_input}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        return response["choices"][0]["message"]["content"]
