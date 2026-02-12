# SigCompressor

SigCompressor is a framework for benchmarking the impact of extractive text compression LLM summarization. By using signals extracted by SigExt from the text, this tool identifies key sentences and phrases to significantly reduce input token count in the prompt, while maintaining summary quality.

## Key Features
*   **Token Efficiency**: Reduce prompt size by 60-80% to lower API costs and latency.
*   **Dual Strategies**: Support for both sentence-level (extracting full keysentences) and phrase-level (extracting keyphrases) compression.
*   **Two LLM models**: Compatible with the local model Mistral and the API model GPT-3.5.
*   **Metrics**: Evaluates performance using ROUGE, BLEU, and BERTScore.

## Project Structure
*   `src/`: Core modules for compression, benchmarking, and metrics.
*   `prepare_sigext.py`: Automates the setup of the SigExt extractor and dataset preparation.
*   `main.py`: The main experiment runner.
*   `analyze_results.py`: Tools for visualizing compression vs metrics.

## Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/SigCompressor.git
    cd SigCompressor
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### 1. Prepare Data & Extractor
This script clones the original SigExt repo, trains the extractor on CNN/DailyMail, and runs inference to generate importance scores.
```bash
python prepare_sigext.py
```

### 2. Run the Benchmark
Execute the compression and summarization pipeline.
```bash
python main.py --input_file cnn_dataset_with_keyphrase/test.jsonl --openai_key YOUR_KEY
```

**Advanced Configuration**
You can specify the compressor type, custom ratios, and the LLM model:
```bash
python main.py \
  --input_file cnn_dataset_with_keyphrase/test.jsonl \
  --openai_key YOUR_KEY \
  --compressor phrase \
  --ratios 0.2 0.4 \
  --llm mistral
```
*   `--compressor`: `sentence`, `phrase`, or `all` (default).
*   `--ratios`: Space-separated list of retention ratios (e.g., `0.2 0.3`).
*   `--llm`: `gpt` (default), `mistral`, or `all`.

### 3. Analyze Results
Generate comparison plots and tables.
```bash
python analyze_results.py --results_dir results
```

## Output
Results are saved in the `results/` directory:
*   `*_summaries.json`: Generated summaries.
*   `*_metrics.csv`: ROUGE/BERTScore metrics.
*   `master_benchmark_results.csv`: Aggregated results.
*   `metrics_comparison.png`: Visualization of the trade-offs.

## Credits
This project utilizes the **SigExt** model. Please refer to the [original repository](https://github.com/amazon-science/SigExt) for details.
