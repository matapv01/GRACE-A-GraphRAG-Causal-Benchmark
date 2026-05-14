# GraphRAG-Benchmark

A Universal Diagnostic Subgraph Framework for evaluating Knowledge Graph Reasoning and Faithfulness 
of Large Language Models (LLMs).

## 🚀 Data Scraping Pipeline Instructions 

The project supports running background processes to construct the full benchmark data from Wikidata without interruption. The code features an auto-resume mechanism (skipping successfully parsed files) and **infinite retries** for limits like `429 Too Many Requests`.

### 1. Generate Benchmark Data Concurrently
Instead of running each script manually, the project provides an orchestration script that runs the entire sequence automatically: 
(1) Extract true graphs from Wikidata -> (2) Generate perturbed graphs -> (3) Fetch Entity Labels -> (4) Convert subgraphs to MCQ formats.

**Main Command:**
```bash
uv run python scripts/data_generation/prepare_benchmark_data.py
```
*(It is highly recommended to use Tmux or run in the background to prevent SSH disconnection).*

- Clean subgraphs are saved to: `data/test_clean_subgraphs/`
- Perturbed subgraphs (causally intervened) are saved to: `data/test_perturbed_subgraphs/`
- Label mappings are saved to: `data/wikidata_labels.json`

**Retry failed items:**
If Wikidata times out and some subgraphs receive an "error" tag inside their JSON instead of actual graph data, you do not need to delete everything. Simply activate the `--retry_errors` flag to overwrite and re-process the corrupted entries while instantly skipping the valid ones.

To backfill and retry across the *entire* pipeline (extraction, perturbation, labels, and MCQ conversion):
```bash
uv run python scripts/data_generation/prepare_benchmark_data.py --retry_errors
```

### 2. Data Filtering / Outlier Subgraphs Removal
Due to the nature of Wikidata Hub Nodes (e.g., highly connected entities like "United States", "Human"), some subgraphs extracted via 2-hop searches can explode to tens of thousands of nodes (up to 90,000+ edges), which will completely exceed the Context Window limits of most modern Large Language Models and cause API crashes or "Lost in the middle" problems during GraphRAG evaluation.

To ensure the Benchmark is fair, safe, and cost-effective, we isolate these extreme outliers (Graphs with **> 1000 nodes** or **> 1000 edges**).
**Command:**
```bash
uv run python scripts/analysis/check_subgraph_stats.py
```
*What it does:* Scans `data/test_*` directories and safely moves any questions violating the threshold constraint into `data/outliers_large_subgraphs/` isolation folder. (In a typical scrape of ~4118 valid LC-QuAD subgraphs, about ~33 outliers / 0.8% of the dataset will be isolated).

### 3. Dataset Health & Statistics Report
This tool scans the entire `data/test_clean_subgraphs` folder and strictly validates dataset integrity. It outputs a healthy Markdown report (`data/dataset_statistics.md`) proving dataset sanity, checking for missing files, and confirming the exact number of test samples with their 5 complete causal graph variants.

**Command:**
```bash
uv run python scripts/analysis/dataset_statistics.py
```

### 4. Extract Question Type Mapping
This script analyzes LC-QuAD's original layout to extract and map logic types and topological query structures (e.g., Simple, Multi-hop, Boolean, etc.). It generates `data/question_type_mapping.csv`, which acts as a crucial index that allows you to split benchmarking and visualization metrics by actual structure.

**Command:**
```bash
uv run python scripts/analysis/extract_question_types.py
```

### 5. Preview Data Samples
Displays detailed insights of the generated subgraphs (nodes, relations, wikidata mappings) for a quick health check.

**Command:**
```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/preview_samples.py
```

### 6. Visualize Graph Structure (Mermaid)
Use this script to output a Markdown Mermaid representation of a specified subgraph. Paste the code into a Mermaid live editor to visually verify Knowledge Graphs under different Causal States.

*(Requires `question_type_mapping.csv` to be spawned first from Step 4)*

Run on a random sample:
```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/visualize_mermaid.py
```

Target a specific topological graph mode (`--mode`) or a direct question ID (`--single_id`):

Available `--mode` types include:
- `"simple question right"` / `"simple question left"`: 1-hop simple retrieval.
- `"right-subgraph"` / `"left-subgraph"`: Multi-hop reasoning queries.
- `"center"`: Converging/diverging center subgraphs.
- `"statement_property"`: Qualifiers branch graph.
- `"two intentions"`: Conjunction queries.
- `"boolean"`: Yes/No graph queries.

```bash
# Preview a random multi-hop question layout
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/visualize_mermaid.py --mode "right-subgraph"

# Preview visually by an exact question ID
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/visualize_mermaid.py --single_id 10856
```

### 7. Emergency Kill
In case you executed the background scrape scripts by mistake, use this block to brutally kill all data extraction worker processes:

```bash
pkill -f extract_clean_subgraphs.py
pkill -f generate_perturbations.py
```
