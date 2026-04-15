# GRACE: A GraphRAG Causal Benchmark

Welcome to the **GRACE** benchmark repository. This project aims to evaluate the robustness and causal reasoning capabilities of various GraphRAG (Retrieval-Augmented Generation over Knowledge Graphs) frameworks when faced with different graph perturbations.

## Installation & Setup

This project uses modern Python dependency management via `uv`.

```bash
# Cài đặt toàn bộ dependencies trong file pyproject.toml
uv sync
```

## Running Benchmarks

We are integrating multiple GraphRAG frameworks into our causal benchmark pipeline. Currently, we provide a starting benchmark script for **LightRAG**.

### 1. Configuration (Environment variables)

Before running the benchmark, you must configure the LLM endpoints via Environment Variables. We use a `.env` file for easy management.

```bash
# Copy the provided template
cp .env.template .env
```

Open the `.env` file and insert your API Key.
*   **OPENAI_API_KEY**: Required by LightRAG for LLM initialization.
*   **OPENAI_BASE_URL** *(Optional)*: Set this if you are using an OpenAI-compatible proxy (e.g. DeepSeek APIs, or local offline LLMs deployed through vLLM or Ollama).

### 2. Prepare Data (Wikidata Labels)

Before running the benchmark, you must fetch the English labels for all Wikidata entities and properties present in the subgraphs. This step ensures that the LLM understands the knowledge triples.

```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/data_generation/fetch_wikidata_labels.py
```
*This script will scan the data folders, query the Wikidata API, and cache the labels in `data/wikidata_labels.json`.*

### 3. Generate Dataset Statistics

It is highly recommended to run the statistical analyzer before benchmarking to verify the exact number of valid causal graphs and visualize query type distributions:

```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/dataset_statistics.py
```
*This will generate a detailed markdown report at `data/dataset_statistics.md` summarizing the distribution of query types (e.g., SELECT, ASK, COUNT) and perturbation variants.*

### 4. LightRAG Benchmark

After preparing the data, verifying statistics, and configuring the `.env` file, run the benchmark evaluation script for LightRAG:

```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/benchmark/benchmark_lightrag.py
```

*Note: The script will load a subgraph scenario from `data/test_perturbed_subgraphs/`, construct Knowledge Graphs dynamically using LightRAG for both the original (clean) and a perturbed variant, and then contrast the responses from the LLM.*

---

## Data Preparation & Analysis

The data pipeline scripts have been logically classified within the `scripts/` folder:
- **`scripts/data_generation/`**: Utilities extracting and perturbing source data.
- **`scripts/analysis/`**: Diagnostic utilities to verify metadata, logs, and diagram visualization.

If you want to learn how the datasets are exactly extracted from Wikidata, how causal perturbations are applied, and how they are statistically analyzed or visualized, please read our detailed methodology guide in:

👉 **[README_DATASET.md](README_DATASET.md)**
