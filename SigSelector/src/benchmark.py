import re
from openai import OpenAI
from groq import Groq

class BenchmarkRunner:
    def __init__(self, openai_api_key=None, groq_api_key=None):
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

    def _extract_summary_xml(self, text, tag="summary"):
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def get_summary_gpt(self, input_text, model="gpt-3.5-turbo"):
        if not self.openai_client:
            return "Skipped"
        prompt = (
            "Here are some news articles:\n" + input_text + "\n\n"
            "Please write a detailed and comprehensive summary of the provided articles. "
            "The summary should cover all key points and relevant details.\n\n"
            "Write your summary in <summary> XML tags."
         )

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=42,
                max_tokens=512
            )
            content = response.choices[0].message.content
            return self._extract_summary_xml(content)
        except Exception as e:
            print(f"Error OpenAI API: {e}")
            return ""

    def get_summary_groq(self, input_text, model="llama-3.1-8b-instant"):
        if not self.groq_client:
            return "Skipped"
        prompt = (
            "Here are some news articles:\n" + input_text + "\n\n"
            "Please write a detailed and comprehensive summary of the provided articles. "
            "The summary should cover all key points and relevant details.\n\n"
            "Write your summary in <summary> XML tags."
         )

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
                max_tokens=512,
            )
            content = chat_completion.choices[0].message.content
            return self._extract_summary_xml(content)
        except Exception as e:
            print(f"Error Groq API: {e}")
            return ""
