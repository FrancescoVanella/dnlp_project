import torch
import gc
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI

ZS_NAIVE_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
    "Please write a short summary for the article in 1-2 sentences.[/INST]",
    # others (arxiv, etc.)
}

ZS_NAIVE_PROMPT_STR_FOR_GPT = {
    "cnn": "Here is a news article:\n<text>\n\nPlease write a summary for the article in 2-3 sentences.",
    # others
}

class BenchmarkRunner:
  def __init__(self, openai_api_key=None, device=None):
    self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f" BenchmarkRunner on {self.device}")

  def _extract_summary_xml(self, text, tag="summary"):
    """Reply extract_xml_tag"""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
      return match.group(1).strip()
    return text.strip()

  def _prepare_prompt(self, model_name, input_text, dataset="cnn"):
    """Select correct prompt based on the model"""

    # MISTRAL
    if "mistral" in model_name.lower():
      template = ZS_NAIVE_PROMPT_STR_FOR_MISTRAL.get(dataset, ZS_NAIVE_PROMPT_STR_FOR_MISTRAL["cnn"])
      final_prompt = template.replace("<text>", input_text)
      return final_prompt, "mistral_style"

    # LLAMA / GPT
    else:
      template = ZS_NAIVE_PROMPT_STR_FOR_GPT.get(dataset, ZS_NAIVE_PROMPT_STR_FOR_GPT["cnn"])
      instruction = template.replace("<text>", input_text)
      # Add XML instruction
      final_prompt = instruction + "\n\nWrite your summary in <summary> XML tags."
      return final_prompt, "chat_style"

  def load_model(self, model_name):
    print(f"\n--- Loading model into VRAM: {model_name} ---")
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configuration Quantization 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    return tokenizer, model

  def get_summary_local(self, model, tokenizer, model_name, input_text, dataset="cnn"):
    try:
      # Prompt preparation
      prompt_str, style = self._prepare_prompt(model_name, input_text, dataset)

      # Tokenizzation
      if style == "chat_style":
        # For Llama 3 use chat template
        messages = [{"role": "user", "content": prompt_str}]
        if hasattr(tokenizer, "apply_chat_template"):
          final_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
          final_input = f"Human: {prompt_str}\n\nAssistant:"
      else:
        # For Mistral use string raw
        final_input = prompt_str

      inputs = tokenizer(final_input, return_tensors="pt").to(self.device)

      with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            #temperature=1.0,
            #top_p=0.8,
            #top_k=10,
            #do_sample=True,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
            )

      generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

      summary = ""
      if style == "mistral_style":
        if "[/INST]" in generated_text:
          summary = generated_text.split("[/INST]")[-1].strip()
        else:
          summary = generated_text[len(final_input):].strip()
      else:
        # XML estraction
        raw_answer = generated_text
        if "assistant" in generated_text:
          raw_answer = generated_text.split("assistant")[-1]
        summary = self._extract_summary_xml(raw_answer)
      return summary

    except Exception as e:
      print(f"Error con {model_name}: {e}")
      return ""

  def get_summary_gpt(self, input_text, dataset="cnn"):
    if not self.openai_client:
      return "Skipped"


    # GPT uses Claude + XML style
    prompt_str, _ = self._prepare_prompt("gpt", input_text, dataset)

    try:
      response = self.openai_client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[{"role": "user", "content": prompt_str}],
          temperature=0,
          #top_p=0.8,
          seed=42,
          max_tokens=512
      )
      content = response.choices[0].message.content
      return self._extract_summary_xml(content)

    except Exception as e:
      print(f"Error API OpenAI: {e}")
      return ""
