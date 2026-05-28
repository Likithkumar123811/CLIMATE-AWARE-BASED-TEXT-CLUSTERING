"""
DeepSeek Integration Module
Uses DeepSeek API for climate category prediction and explanations
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
import os


class DeepSeekReasoning:
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-reasoner"):

        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url
        self.model = model

        if not self.api_key:
            print("Warning: No API key provided.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.response_cache = {}

    # ------------------------------------------------------------------

    def _make_request(self,
                      prompt: str,
                      max_tokens: int = 1000,
                      temperature: float = 0.7) -> Optional[Dict]:

        if not self.api_key:
            return None

        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval)

        cache_key = f"{prompt}_{max_tokens}_{temperature}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )

            self.last_request_time = time.time()

            if response.status_code == 200:
                result = response.json()
                self.response_cache[cache_key] = result
                return result

        except Exception as e:
            print("API error:", e)

        return None

    # ------------------------------------------------------------------

    def predict_category(self,
                         text: str,
                         available_categories: List[str]) -> Dict[str, Any]:

        categories_str = ", ".join(available_categories)

        # ✅ MODIFIED PROMPT (CLIMATE)
        prompt = f"""
You are an expert climate event classification system.

Available categories: {categories_str}

Event: "{text}"

Respond in JSON:
{{
  "predicted_category": "",
  "confidence": 0.0,
  "reasoning": "Explain why this belongs to this climate category",
  "key_indicators": []
}}
"""

        response = self._make_request(prompt, 400, 0.3)

        if not response:
            return {"predicted_category": "UNKNOWN", "confidence": 0}

        try:
            content = response['choices'][0]['message']['content']
            return json.loads(content)
        except:
            return {"predicted_category": "UNKNOWN", "confidence": 0}

    # ------------------------------------------------------------------

    def generate_explanation(self,
                             text: str,
                             predicted_category: str,
                             ground_truth_category: Optional[str] = None) -> Dict[str, Any]:

        # ✅ MODIFIED PROMPT
        prompt = f"""
Explain why this climate event belongs to category "{predicted_category}"

Text: "{text}"

Actual Category: {ground_truth_category}

Respond in JSON:
{{
  "explanation": "",
  "key_phrases": [],
  "event_type": "",
  "impact": ""
}}
"""

        response = self._make_request(prompt, 500, 0.4)

        if not response:
            return {"explanation": "Failed"}

        try:
            content = response['choices'][0]['message']['content']
            return json.loads(content)
        except:
            return {"explanation": "Parsing error"}

    # ------------------------------------------------------------------

    def analyze_mismatch(self,
                         text: str,
                         predicted_category: str,
                         actual_category: str) -> Dict[str, Any]:

        # ✅ MODIFIED PROMPT
        prompt = f"""
sk-e39dd411a08042dfa0e53f29f3218399

Text: "{text}"
Predicted: {predicted_category}
Actual: {actual_category}

Respond in JSON:
{{
  "reason": "",
  "correction": ""
}}
"""

        response = self._make_request(prompt, 400, 0.5)

        if not response:
            return {"reason": "Failed"}

        try:
            content = response['choices'][0]['message']['content']
            return json.loads(content)
        except:
            return {"reason": "Parsing error"}


# ----------------------------------------------------------------------

def main():
    print("DeepSeek module ready")


if __name__ == "__main__":
    main()