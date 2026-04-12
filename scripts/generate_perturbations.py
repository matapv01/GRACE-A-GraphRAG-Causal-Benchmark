import os
import time
import json
from pathlib import Path
from rich.console import Console
from graphrag_benchmark.domain.models import ReasoningSubgraph
from graphrag_benchmark.interfaces.wikidata_api import WikidataClient
from graphrag_benchmark.use_cases.perturbation_module import PerturbationModule

def main():
    console = Console()
    input_dir = Path("data/clean_subgraphs")
    output_dir = Path("data/perturbed_subgraphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wd_client = WikidataClient()
    perturber = PerturbationModule(wd_client)
    
    console.print(f"[bold green]Starting Perturbation Worker in background...[/bold green]")
    console.print(f"Polling '{input_dir}' and saving to '{output_dir}'")
    
    processed_count = 0
    idle_ticks = 0
    
    while True:
        if not input_dir.exists():
            time.sleep(5)
            continue
            
        json_files = list(input_dir.glob("*.json"))
        new_files_processed_this_tick = 0
        
        for file_path in json_files:
            out_file_path = output_dir / file_path.name
            
            # Skip if already processed
            if out_file_path.exists():
                continue
                
            try:
                # Load the clean subgraph
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()
                    subgraph = ReasoningSubgraph.model_validate_json(data)
                    
                # Generate all 5 versions (Clean, Broken, Type Matching, Topological, Swapping)
                variants_dict = perturber.generate_all_variants(subgraph)
                
                # Combine outputs into one nested JSON file
                dump_data = {}
                for v_type, v_graph in variants_dict.items():
                    dump_data[v_type.value] = v_graph.model_dump(mode="json")
                    
                # Save to perturbed_subgraphs/
                with open(out_file_path, "w", encoding="utf-8") as f:
                    json.dump(dump_data, f, indent=2)
                    
                new_files_processed_this_tick += 1
                processed_count += 1
                console.print(f"[{time.strftime('%H:%M:%S')}] Perturbed: {file_path.name} | Total: {processed_count}")
                
            except Exception as e:
                console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                # Touch file with error so it doesn't block the loop forever
                with open(out_file_path, "w", encoding="utf-8") as f:
                    json.dump({"error": str(e)}, f)
                    
        # Sleep logic to chase the extractor task
        if new_files_processed_this_tick == 0:
            idle_ticks += 1
            if idle_ticks > 60: # 5 minutes (60 * 5s) without any new files -> auto exit
                console.print("\n[bold yellow]No new subgraphs extracted for 5 minutes. Assuming Extraction task is complete. Exiting Perturbation Worker.[/bold yellow]")
                break
            time.sleep(5)
        else:
            idle_ticks = 0

if __name__ == "__main__":
    main()
