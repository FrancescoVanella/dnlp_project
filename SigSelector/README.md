
# SigSelector

SigSelector is a framework for efficient multi-document summarization. It uses SigExt to score and select the most relevant documents from a cluster before feeding them to an LLM.

## Structure
- `src/selector.py`: Logic for Top-k document selection.
- `src/benchmark.py`: LLM Interface (GPT-3.5, Llama 3.1).
- `prepare_sigext.py`: Pipeline to clone, train, and run SigExt inference.
- `main.py`: Main execution script.

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Data
```bash
python prepare_sigext.py
```

### 3. Run Benchmark

**Run Everything (Default: Top-1, Top-2, Full Text | GPT & Llama):**
```bash
export OPENAI_API_KEY="..."
export GROQ_API_KEY="..."
python main.py
```

**Run Specific Strategies (e.g., only Top-1 and Full Text):**
```bash
python main.py --strategies 1 full
```

**Run Specific Models (e.g., Llama only):**
```bash
python main.py --models llama
```

**Quick Test (First 5 documents only):**
```bash
python main.py --limit 5
```
