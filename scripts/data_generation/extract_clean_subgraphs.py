import os
import time
import argparse
from rich.console import Console
from tqdm import tqdm
from graphrag_benchmark.infrastructure.dataset_loader import DatasetLoader
from graphrag_benchmark.interfaces.wikidata_api import WikidataClient
from graphrag_benchmark.use_cases.extract_gold_path import WikidataExtractor
from graphrag_benchmark.interfaces.embedding_api import EmbeddingClient
from graphrag_benchmark.use_cases.semantic_retriever import SemanticRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean knowledge graphs from Wikidata given a dataset."
    )
    parser.add_argument(
        "--retry_errors",
        action="store_true",
        help="If flagged, ignores missing valid JSON outputs and heavily retries previously failed extractions.",
    )
    args = parser.parse_args()

    console = Console()

    dataset_file = "data/lcquad_test.json"
    output_dir = "data/test_clean_subgraphs"

    os.makedirs(output_dir, exist_ok=True)

    console.print(
        f"[bold green]Loading Dataset Form: LC-QuAD 2.0 Test[/bold green]"
    )
    loader = DatasetLoader(dataset_file)
    questions = loader.load_dataset()

    console.print(f"Total questions loaded: {len(questions)}")

    # Initialize modules
    wd_client = WikidataClient()
    wd_extractor = WikidataExtractor(wd_client)
    embed_client = EmbeddingClient()
    retriever = SemanticRetriever(wd_client, embed_client)

    # Fetch previously extracted files to skip duplicates
    existing_files = {
        f.name for f in os.scandir(output_dir) if f.name.endswith(".json")
    }

    if args.retry_errors:
        console.print(
            "[red]--retry_errors enabled: Will FIND and RETRY previously failed queries (Skipping successful JSON extractions).[/red]"
        )

    console.print(
        "[bold yellow]Extracting Clean Subgraphs for all valid queries in background...[/bold yellow]"
    )

    processed_count = 0
    success_count = 0
    error_logs = []

    for q in tqdm(questions):
        try:
            file_name = f"{q.id}.json"
            if file_name in existing_files:
                # Skip if file already exists (Resume capability on restart)
                success_count += 1
                processed_count += 1
                continue

            # 1. Extract Gold Path (Ground Truth) from Wikidata SPARQL Endpoint
            gold_path = wd_extractor.extract_gold_path(q)

            # Skip questions with unparseable SPARQL or Endpoint errors (where Gold Path cannot be tracked)
            if not gold_path.triples:
                continue

            # 2. Generate Candidate Reasoning Space (Extra Context) via Semantic Retrieval
            subgraph = retriever.generate_subgraph(
                question_data=q, gold_path=gold_path, hop_count=1, top_k=5
            )

            # 3. Save unedited version (Clean/Unperturbed Subgraph) to JSON
            file_path = os.path.join(output_dir, f"{q.id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(subgraph.model_dump_json(indent=2))

            success_count += 1

            # Sleep 0.5s between loops to bypass Wikidata rate-limit bans
            time.sleep(0.5)

        except Exception as e:
            console.print(f"\n[red]Error processing Question {q.id}: {e}[/red]")
            time.sleep(1)  # Backoff if error

        processed_count += 1

    console.print(
        f"\n[bold green]Successfully extracted and saved {success_count} Clean Subgraphs[/bold green] out of {processed_count} processed requests."
    )
    console.print(f"Results saved in: [cyan]{os.path.abspath(output_dir)}/[/cyan]")


if __name__ == "__main__":
    main()
