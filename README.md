# DNLP project 

This repository contains two extensions for the DNLP project. Each subdirectory contains a self-contained module with its own documentation and source code.

## Modules

### 1. [SigCompressor](./SigCompressor)
A framework that uses the SigExt model to compress text prompts (at sentence or phrase level) before feeding them to LLMs, with the aim of reducing the number of tokens in the prompt while maintaining summary quality.

### 2. [SigSelector](./SigCompressor)
A framework that uses the SigExt model to rank and select the most relevant source documents based on keyphrase density before feeding them to LLMs. The aim is to optimize the input context for multi-document summarization by prioritizing high-value documents and filtering out less informative ones.

## Repository Organization
*   `SigCompressor/`: Contains the source code, scripts, and specific README for the first extension.
*   `SigSelector/`: Contains the source code, scripts, and specific README for the second extension.
