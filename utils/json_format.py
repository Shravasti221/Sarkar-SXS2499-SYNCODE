import os
import json
import re
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Type, Optional


class JsonFormat:
    def __init__(self, pydantic_model: Type[BaseModel], expected_format: str, max_retries: int = 3, llm: Optional[ChatOpenAI] = None, verbose: bool = True):
        self.max_retries = max_retries
        self.verbose = verbose
        self.pydantic_model = pydantic_model
        if expected_format is not None:
            self.expected_format = expected_format
        else:
            self.expected_format = json.dumps(pydantic_model.model_json_schema(), indent=2)


        self.llm = llm or ChatOpenAI(
            model="openai/gpt-oss-20b",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0,
            reasoning_effort="low",
            model_kwargs={
                "stream": False,
                # "response_format": {"type": "json_object"},
            },
        )

    def _evaluate(self, text: str) -> tuple[float, str]:
        """Ask model to rate format compliance."""
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            return 0, "No JSON object found in the text."
        try:
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return 0.0, f"Invalid JSON: {str(e)}"
        try:
            self.pydantic_model(**data)  # Validate using the Pydantic model
            return 1.0, json_str
        except ValueError as e:
            return 0.0, f"Validation failed: {str(e)}"

    def _correct(self, text: str, feedback: str) -> str:
        """Ask model to correct format issues."""
        prompt = f"""
        Correct the following response so it matches the expected JSON format.

        Expected format:
        {self.expected_format}

        Response:
        {text}

        Feedback:
        {feedback}

        Return only the corrected text, no commentary.
        """
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as e:
            print("Correction error:", e)
            return text

    def refine(self, text: str) -> str:
        """Run evaluator–corrector loop up to max_retries."""
        # print("\n=== FormatRefiner ===")
        best_text, best_score = text, 0

        for i in range(1, self.max_retries + 1):
            # print(f"\n--- Pass {i}/{self.max_retries} ---")
            score, feedback = self._evaluate(best_text)
            # print(f"Score: {score}, Feedback/Clean text: {feedback}")

            if score >= 1:
                best_text = feedback
                return best_text

            # corrected = self._correct(best_text, feedback)
            best_text = self._correct(best_text, feedback)
        return best_text
